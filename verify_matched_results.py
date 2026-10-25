from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
BACKBONES = ("MPNet", "DistilBERT", "MiniLM")
METHODS = ("full_fine_tuning", "LoRA-large")
METRICS = ("test_f1", "test_roc_auc")
SEEDS = (42, 123, 2026, 7, 13, 101, 202, 404, 777, 1729)


def assert_close(actual, expected, description, atol=1e-7):
    if pd.isna(actual) and pd.isna(expected):
        return
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=0):
        raise AssertionError(f"{description}: calculated {actual}, archived {expected}")


def load_csv(folder, name):
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required results file: {path}")
    return pd.read_csv(path)


def softmax_positive(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials / exponentials.sum(axis=1, keepdims=True))[:, 1]


def ece(labels, probabilities, bins=10):
    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = (probabilities >= .5).astype(int) == labels
    edges = np.linspace(.5, 1., bins + 1)
    value = 0.
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            (confidence <= edges[index + 1]) if index == bins - 1
            else (confidence < edges[index + 1])
        )
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def verify_seed_logits(folder, runs):
    reference_labels = None
    for row in runs.itertuples(index=False):
        description = f"{row.backbone}/{row.method}/seed{row.seed}"
        path = folder / "seed_logits" / f"{row.backbone}_{row.method}_seed{row.seed}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as arrays:
            logits = arrays["test_logits"]
            labels = arrays["test_labels"].astype(int)
            validation_logits = arrays["validation_logits"]
            validation_labels = arrays["validation_labels"].astype(int)
        if logits.shape != (715, 2) or validation_logits.shape != (423, 2):
            raise AssertionError(f"Unexpected logit shapes for {description}")
        if labels.shape != (715,) or validation_labels.shape != (423,):
            raise AssertionError(f"Unexpected label shapes for {description}")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise AssertionError(f"Nonbinary test labels for {description}")
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise AssertionError(f"Official test label ordering differs in {description}")
        probabilities = softmax_positive(logits)
        predictions = np.argmax(logits, axis=1)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        computed = {
            "test_accuracy": accuracy_score(labels, predictions),
            "test_precision": precision_score(labels, predictions, zero_division=0),
            "test_recall": recall_score(labels, predictions, zero_division=0),
            "test_f1": f1_score(labels, predictions, zero_division=0),
            "test_roc_auc": roc_auc_score(labels, probabilities),
            "test_tn": tn, "test_fp": fp, "test_fn": fn, "test_tp": tp,
        }
        for metric, value in computed.items():
            assert_close(value, getattr(row, metric), f"{description}: {metric}")


def verify_summary(folder, runs):
    summary = load_csv(folder, "reviewer_matched_summary.csv")
    columns = ("test_accuracy", "test_f1", "test_roc_auc", "test_precision", "test_recall")
    if len(summary) != len(BACKBONES) * len(METHODS):
        raise AssertionError("Expected six summary rows")
    for row in summary.itertuples(index=False):
        group = runs[(runs.backbone == row.backbone) & (runs.method == row.method)]
        if len(group) != 10:
            raise AssertionError(f"Expected ten runs for {row.backbone}/{row.method}")
        for metric in columns:
            assert_close(group[metric].mean(), getattr(row, metric + "_mean"), metric + " mean")
            assert_close(group[metric].std(ddof=1), getattr(row, metric + "_std"), metric + " std")
    return summary


def verify_paired_statistics(folder, runs):
    archived = load_csv(folder, "reviewer_paired_statistics.csv")
    calculated = []
    for backbone in BACKBONES:
        for metric in METRICS:
            pivot = (runs[runs.backbone == backbone]
                     .pivot(index="seed", columns="method", values=metric)
                     .reindex(SEEDS))
            if pivot.isna().any().any():
                raise AssertionError(f"Missing paired observations for {backbone}/{metric}")
            diffs = (pivot["LoRA-large"] - pivot["full_fine_tuning"]).to_numpy()
            observed = float(diffs.mean())
            sign_patterns = np.asarray(list(product((-1, 1), repeat=10)))
            permutation_means = (sign_patterns * diffs).mean(axis=1)
            exact_p = float(np.mean(np.abs(permutation_means) >= abs(observed) - 1e-12))
            rng = np.random.default_rng(20261022)
            indices = rng.integers(0, 10, size=(20000, 10))
            ci = np.quantile(diffs[indices].mean(axis=1), (.025, .975))
            calculated.append({
                "backbone": backbone, "metric": metric,
                "n_paired_seeds": 10,
                "mean_lora_minus_full": observed,
                "95pct_seed_bootstrap_low": ci[0],
                "95pct_seed_bootstrap_high": ci[1],
                "unadjusted_exact_two_sided_p": exact_p,
            })
    frame = pd.DataFrame(calculated)
    ordered = frame["unadjusted_exact_two_sided_p"].sort_values().index.tolist()
    running_max = 0.
    for rank, index in enumerate(ordered):
        running_max = max(running_max, min(1., (len(ordered) - rank) *
            frame.loc[index, "unadjusted_exact_two_sided_p"]))
        frame.loc[index, "holm_adjusted_p"] = running_max
    for _, row in frame.iterrows():
        saved = archived[(archived.backbone == row["backbone"]) &
                         (archived.metric == row["metric"])]
        if len(saved) != 1:
            raise AssertionError(f"Missing or duplicate statistics for {row['backbone']}/{row['metric']}")
        for key in ("n_paired_seeds", "mean_lora_minus_full", "95pct_seed_bootstrap_low",
                    "95pct_seed_bootstrap_high", "unadjusted_exact_two_sided_p", "holm_adjusted_p"):
            assert_close(row[key], saved.iloc[0][key], f"{row['backbone']}/{row['metric']}/{key}")
    return frame


def verify_calibration(folder, runs):
    calibration = load_csv(folder, "reviewer_calibration_by_seed.csv")
    selective = load_csv(folder, "reviewer_selective_prediction_by_seed.csv")
    select_summary = load_csv(folder, "reviewer_selective_prediction_summary.csv")
    expected_rows = len(runs)
    if len(calibration) != expected_rows or len(selective) != expected_rows * 5:
        raise AssertionError("Unexpected number of calibration or selective prediction rows")
    for row in calibration.itertuples(index=False):
        description = f"{row.backbone}/{row.method}/seed{row.seed}"
        path = folder / "seed_logits" / f"{row.backbone}_{row.method}_seed{row.seed}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            logits = arrays["test_logits"]
            labels = arrays["test_labels"].astype(int)
        if row.temperature <= 0:
            raise AssertionError(f"Invalid fitted temperature in {description}")
        raw = softmax_positive(logits)
        calibrated = softmax_positive(logits / row.temperature)
        if not np.array_equal(raw >= .5, calibrated >= .5):
            raise AssertionError(f"Temperature scaling altered class predictions in {description}")
        for key, val in {
            "ece_before": ece(labels, raw), "ece_after": ece(labels, calibrated),
            "brier_before": brier_score_loss(labels, raw),
            "brier_after": brier_score_loss(labels, calibrated),
        }.items():
            assert_close(val, getattr(row, key), f"{description}/{key}")
        confidence = np.maximum(calibrated, 1 - calibrated)
        prediction = (calibrated >= .5).astype(int)
        for threshold in (.5, .6, .7, .8, .9):
            saved = selective[(selective.backbone == row.backbone) &
                              (selective.method == row.method) &
                              (selective.seed == row.seed) &
                              (np.isclose(selective.threshold, threshold))]
            if len(saved) != 1:
                raise AssertionError(f"Missing selective prediction row: {description}/{threshold}")
            saved = saved.iloc[0]
            keep = confidence >= threshold
            assert_close(keep.sum(), saved.retained_n, description + "/retained_n")
            assert_close(len(labels), saved.test_n, description + "/test_n")
            assert_close(keep.mean(), saved.coverage, description + "/coverage")
            if keep.any():
                predicted = prediction[keep]
                target = labels[keep]
                accuracy = float((predicted == target).mean())
                tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
                sensitivity = tp / (tp + fn) if tp + fn else np.nan
                specificity = tn / (tn + fp) if tn + fp else np.nan
                gmean = np.sqrt(sensitivity * specificity)
            else:
                accuracy = gmean = np.nan
            assert_close(accuracy, saved.retained_accuracy, description + "/retained_accuracy")
            assert_close(gmean, saved.retained_gmean, description + "/retained_gmean")
    for row in select_summary.itertuples(index=False):
        group = selective[(selective.backbone == row.backbone) &
                          (selective.method == row.method) &
                          np.isclose(selective.threshold, row.threshold)]
        if len(group) != 10:
            raise AssertionError("Selective prediction summary should aggregate ten seeds")
        for metric in ("coverage", "retained_accuracy", "retained_gmean"):
            assert_close(group[metric].mean(), getattr(row, metric + "_mean"), metric + " mean")
            assert_close(group[metric].std(ddof=1), getattr(row, metric + "_std"), metric + " std")


def main():
    parser = argparse.ArgumentParser(
        description="Verify the archived matched ten-seed results without GPU training."
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "reviewer_results")
    args = parser.parse_args()
    folder = args.results_dir
    runs = load_csv(folder, "reviewer_matched_test_by_seed.csv")
    if len(runs) != 60:
        raise AssertionError(f"Expected 60 matched training results; found {len(runs)}")
    actual = {(str(row.backbone), str(row.method), int(row.seed))
              for row in runs.itertuples(index=False)}
    expected = {(backbone, method, seed) for backbone in BACKBONES
                for method in METHODS for seed in SEEDS}
    if actual != expected or len(actual) != len(runs):
        raise AssertionError("Backbone/method/seed keys must form 60 unique matched runs")
    if not (runs["effective_batch"] == 8).all():
        raise AssertionError("Not all runs used effective batch size eight")
    verify_seed_logits(folder, runs)
    summary = verify_summary(folder, runs)
    statistics = verify_paired_statistics(folder, runs)
    verify_calibration(folder, runs)
    print("PASS: 60 archived model test predictions match reported metrics")
    print("PASS: all six mean/SD summary rows match")
    print("PASS: six paired sign-flip tests, 20,000-bootstrap intervals, and Holm p-values match")
    print("PASS: temperature-scaled test metrics and 0.50-0.90 confidence coverage match")
    print("\nMPNet ten-seed headline:")
    print(summary[summary.backbone == "MPNet"][[
        "method", "test_f1_mean", "test_f1_std", "test_roc_auc_mean", "test_roc_auc_std"
    ]].to_string(index=False))
    print("\nMPNet paired inference:")
    print(statistics[statistics.backbone == "MPNet"][[
        "metric", "mean_lora_minus_full", "holm_adjusted_p"
    ]].to_string(index=False))
    print("\nNote: This verifies recorded results, NOT independent GPU retraining.")


if __name__ == "__main__":
    main()
