from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
MODEL_CHECKPOINTS = {
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "DistilBERT": "distilbert-base-uncased",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
}
SEEDS = (42, 123, 2026, 7, 13, 101, 202, 404, 777, 1729)
METHODS = ("full_fine_tuning", "LoRA-large")


def load_data(data_dir):
    train_path = data_dir / "dreaddit-train.csv"
    test_path = data_dir / "dreaddit-test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Download official Dreaddit CSVs (see README.md), then place "
            f"dreaddit-train.csv and dreaddit-test.csv in {data_dir}"
        )
    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)
    train = raw_train.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    train = train.loc[~train["text"].isin(set(raw_test["text"].dropna()))].reset_index(drop=True)
    source = train[["text", "label"]].dropna(subset=["text", "label"]).copy()
    test = raw_test[["text", "label"]].dropna(subset=["text", "label"]).copy()
    source["text"] = source["text"].astype(str)
    test["text"] = test["text"].astype(str)
    source["label"] = source["label"].astype(int)
    test["label"] = test["label"].astype(int)
    train_df, val_df = train_test_split(
        source, test_size=.15, stratify=source["label"], random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if (len(source), len(train_df), len(val_df), len(test)) != (2817, 2394, 423, 715):
        raise RuntimeError("Unexpected Dreaddit row counts; check downloaded CSVs")
    hashes = {
        name: hashlib.sha256("\n".join(frame.text).encode()).hexdigest()
        for name, frame in (("train", train_df), ("validation", val_df), ("test", test))
    }
    expected = json.loads((ROOT / "reviewer_results" / "reviewer_run_design.json").read_text())
    for name, digest in hashes.items():
        if digest != expected[name + "_text_hash"]:
            raise RuntimeError(f"{name} text hash does not match the archived experiment")
    return {"train": train_df, "validation": val_df, "test": test}


def tokenized_data(checkpoint, frames):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    raw = DatasetDict({
        name: Dataset.from_pandas(frame.rename(columns={"label": "labels"}), preserve_index=False)
        for name, frame in frames.items()
    })
    encoded = raw.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=256),
        batched=True, remove_columns=["text"],
    )
    return tokenizer, encoded


def softmax_positive(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(z)
    return (probabilities / probabilities.sum(axis=1, keepdims=True))[:, 1]


def get_metrics(labels, logits):
    predictions = np.argmax(logits, axis=1)
    probabilities = softmax_positive(logits)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def trainer_metrics(evaluation_prediction):
    logits, labels = evaluation_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    return get_metrics(labels, np.asarray(logits))


def build_model(checkpoint, method):
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2,
        id2label={0: "Non-stress", 1: "Stress/Stress-related"},
        label2id={"Non-stress": 0, "Stress/Stress-related": 1},
    )
    if method == "LoRA-large":
        modules = {
            "bert": (["query", "value"], ["classifier"]),
            "mpnet": (["q", "v"], ["classifier"]),
            "distilbert": (["q_lin", "v_lin"], ["pre_classifier", "classifier"]),
        }
        targets, classification = modules[model.config.model_type]
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
            lora_dropout=.10, bias="none", target_modules=targets,
            modules_to_save=classification,
        ))
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return model


def run_one(backbone, method, seed, lr, frames, output_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer, encoded = tokenized_data(MODEL_CHECKPOINTS[backbone], frames)
    model = build_model(MODEL_CHECKPOINTS[backbone], method)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    batch = 8 if gpu_memory_gb >= 14 else 4
    accumulation = 8 // batch
    with tempfile.TemporaryDirectory(prefix="matched_training_", dir=output_dir) as temp_dir:
        args = TrainingArguments(
            output_dir=temp_dir, learning_rate=float(lr),
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=accumulation,
            per_device_eval_batch_size=16,
            num_train_epochs=5, weight_decay=.01,
            warmup_steps=int(math.ceil(.10 * 5 * math.ceil(len(encoded["train"]) / 8))),
            eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="f1",
            greater_is_better=True, save_total_limit=1,
            fp16=True, optim="adamw_torch", seed=int(seed), data_seed=int(seed),
            report_to=[], disable_tqdm=True,
        )
        trainer = Trainer(
            model=model, args=args, train_dataset=encoded["train"],
            eval_dataset=encoded["validation"], processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
            compute_metrics=trainer_metrics,
        )
        try:
            trainer.train()
            val = trainer.predict(encoded["validation"])
            test = trainer.predict(encoded["test"])
            val_logits = np.asarray(val.predictions[0] if isinstance(val.predictions, tuple)
                                    else val.predictions)
            test_logits = np.asarray(test.predictions[0] if isinstance(test.predictions, tuple)
                                     else test.predictions)
            epochs = [entry for entry in trainer.state.log_history if "eval_f1" in entry]
            best_epoch = (float(max(epochs, key=lambda entry: entry["eval_f1"])["epoch"])
                          if epochs else np.nan)
            record = {
                "backbone": backbone, "method": method, "lr": float(lr), "seed": int(seed),
                "train_batch": batch, "gradient_accumulation": accumulation,
                "effective_batch": 8, "max_epochs": 5, "best_epoch": best_epoch,
                "validation_f1": float(f1_score(val.label_ids, val_logits.argmax(axis=1),
                                                 zero_division=0)),
                "validation_roc_auc": float(roc_auc_score(val.label_ids,
                                                         softmax_positive(val_logits))),
                "trainable_parameters": trainable, "total_parameters": total,
                "trainable_fraction": trainable / total,
                **{"test_" + k: v for k, v in get_metrics(test.label_ids, test_logits).items()},
            }
            arrays = {
                "validation_logits": val_logits, "validation_labels": np.asarray(val.label_ids),
                "test_logits": test_logits, "test_labels": np.asarray(test.label_ids),
            }
            return record, arrays
        finally:
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Rerun the validation-selected matched models on a CUDA GPU."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "matched_reproduction")
    parser.add_argument("--backbone", choices=MODEL_CHECKPOINTS, default="MPNet")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--all-backbones", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Matched-model training requires a CUDA GPU. See README for CPU verification.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_data(args.data_dir)
    selected = pd.read_csv(ROOT / "reviewer_results" / "reviewer_selected_hyperparameters.csv")
    backbones = MODEL_CHECKPOINTS if args.all_backbones else [args.backbone]
    seeds = SEEDS if args.all_seeds else [args.seed]
    results_path = args.output_dir / "matched_retrained_by_seed.csv"
    for backbone in backbones:
        for method in METHODS:
            selection = selected[(selected.backbone == backbone) & (selected.method == method)]
            if len(selection) != 1:
                raise RuntimeError(f"Missing frozen validation learning rate for {backbone}/{method}")
            lr = float(selection.iloc[0].lr)
            for seed in seeds:
                previous = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
                logit_file = args.output_dir / f"{backbone}_{method}_seed{seed}.npz"
                if (not previous.empty and logit_file.exists() and
                    ((previous.backbone == backbone) & (previous.method == method) &
                     (previous.seed.astype(int) == seed)).any()):
                    print("Skipping completed run:", backbone, method, seed)
                    continue
                print("Training:", backbone, method, seed, "lr=", lr, flush=True)
                row, arrays = run_one(backbone, method, seed, lr, frames, args.output_dir)
                np.savez_compressed(logit_file, **arrays)
                pd.concat([previous, pd.DataFrame([row])], ignore_index=True).to_csv(results_path, index=False)
                print("Test F1:", row["test_f1"], "ROC-AUC:", row["test_roc_auc"], flush=True)
    print("Saved new GPU runs in", args.output_dir)
    print("The historical reviewer_results/ files have not been modified.")


if __name__ == "__main__":
    main()
