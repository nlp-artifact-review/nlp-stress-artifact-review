from pathlib import Path
import gc
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = ROOT / ".checkpoints"

TRAIN_PATH = DATA_DIR / "dreaddit-train.csv"
TEST_PATH = DATA_DIR / "dreaddit-test.csv"

SEEDS = [42, 123, 2026]
CHECKPOINT = "sentence-transformers/all-mpnet-base-v2"
MAX_LENGTH = 256
EPOCHS = 3

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError(
        "Missing Dreaddit data files. Download the dataset attachment from "
        "https://aclanthology.org/D19-6213/ and place "
        "`dreaddit-train.csv` and `dreaddit-test.csv` in the `data/` folder."
    )

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)

train = train_raw.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
test_texts = set(test_raw["text"].dropna())
train = train.loc[~train["text"].isin(test_texts)].reset_index(drop=True)
test = test_raw.copy().reset_index(drop=True)

source = train[["text", "label"]].dropna(subset=["text", "label"]).reset_index(drop=True)
test_df = test[["text", "label"]].dropna(subset=["text", "label"]).reset_index(drop=True)

source["text"] = source["text"].astype(str)
test_df["text"] = test_df["text"].astype(str)
source["label"] = source["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

train_df, val_df = train_test_split(
    source,
    test_size=0.15,
    stratify=source["label"],
    random_state=42,
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

if (len(source), len(train_df), len(val_df), len(test_df)) != (2817, 2394, 423, 715):
    raise RuntimeError(
        "Unexpected Dreaddit sizes after preprocessing. "
        f"Observed {(len(source), len(train_df), len(val_df), len(test_df))}."
    )


def as_dataset(frame):
    return Dataset.from_pandas(
        frame.rename(columns={"label": "labels"}),
        preserve_index=False,
    )


tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)

raw = DatasetDict(
    {
        "train": as_dataset(train_df),
        "validation": as_dataset(val_df),
        "test": as_dataset(test_df),
    }
)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)


datasets = raw.map(tokenize, batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def metrics(labels, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "g_mean": float(np.sqrt(sensitivity * specificity)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_metrics(prediction):
    logits, labels = prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=1)
    probabilities = softmax(logits)[:, 1]
    return metrics(labels, predictions, probabilities)


def run(method, seed):
    set_all_seeds(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=2,
        id2label={0: "Non-stress", 1: "Stress/Stress-related"},
        label2id={"Non-stress": 0, "Stress/Stress-related": 1},
    )

    if method == "Full fine-tuning":
        learning_rate = 2e-5
        gradient_accumulation_steps = 2
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif method == "LoRA-large":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                target_modules=["q", "v"],
                modules_to_save=["classifier"],
            ),
        )
        learning_rate = 2e-4
        gradient_accumulation_steps = 1
    else:
        raise ValueError(method)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())

    run_dir = CHECKPOINT_DIR / f"{method.replace(' ', '_').replace('-', '_')}_{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)

    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.10,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="no",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        seed=seed,
        data_seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_output = trainer.predict(datasets["test"])

    logits = test_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    labels = test_output.label_ids
    predictions = np.argmax(logits, axis=1)
    probabilities = softmax(logits)[:, 1]

    row = {
        "training_method": method,
        "seed": seed,
        "best_validation_f1": float(trainer.state.best_metric),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / total,
        **metrics(labels, predictions, probabilities),
    }

    shutil.rmtree(run_dir, ignore_errors=True)
    del trainer
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


if not torch.cuda.is_available():
    print("Warning: CUDA GPU not detected. Reproduction will run substantially slower.")

rows = []

for seed in SEEDS:
    for method in ["Full fine-tuning", "LoRA-large"]:
        print(f"Running {method}, seed={seed}")
        rows.append(run(method, seed))

runs = pd.DataFrame(rows)
runs.to_csv(OUTPUT_DIR / "initial_three_seed_runs.csv", index=False)

summary = (
    runs.groupby("training_method")[
        ["accuracy", "g_mean", "f1", "roc_auc", "trainable_percentage"]
    ]
    .agg(["mean", "std"])
)

summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary = summary.reset_index()
summary.to_csv(OUTPUT_DIR / "initial_three_seed_summary.csv", index=False)

print("\nInitial three-seed summary (NOT the matched ten-seed headline):")
print(summary.to_string(index=False))
