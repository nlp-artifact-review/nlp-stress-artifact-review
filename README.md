# Anonymous stress-detection reproducibility artifact

Artifact for **Toward Parameter-Efficient and Trustworthy Transformers for Stress Detection in Reddit Posts**.

The paper includes **two distinct sets of experiments**. The initial comparisons use three seeds (42, 123, 2026) and method-specific training settings. The follow-up comparison uses **ten matched seeds**, an effective batch size of **eight**, validation-selected learning rates, and up to five epochs for both full fine-tuning and LoRA-large. **Do not mix results from these two designs.** The matched ten-seed MPNet result—not the original three-seed result—supports the paper's statistical comparison.

## Reviewer quickstart: verify the archived matched experiment

The repository includes the per-seed test/validation logits, labels, metrics, and calibration temperatures from the completed experiment. A CPU-only verification script recalculates the reported test metrics, means, standard deviations, bootstrap intervals, exact paired sign-flip p-values, Holm corrections, and calibrated confidence/coverage values. No raw Reddit text, pretrained models, or GPU is needed for this check.

With Python 3.11+ installed, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-verify.txt
python verify_matched_results.py
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\activate`. The statistics verification should report the following ten-seed result (mean ± sample standard deviation):

| MPNet method | F1-score | ROC-AUC | Parameters updated |
|---|---:|---:|---:|
| LoRA-large | 0.8473 ± 0.0045 | 0.9125 ± 0.0028 | 1.07% |
| Full fine-tuning | 0.8217 ± 0.0109 | 0.8941 ± 0.0159 | 100% |

Across the ten paired seeds, the mean LoRA-minus-full difference is **+0.02556 in F1** and **+0.01844 in ROC-AUC**. The Holm-adjusted exact permutation p-value is **0.01172** for each (six planned comparisons: three backbones × two metrics). These are results *for this dataset and training design*, not a universal superiority claim.

**Verification of stored model predictions is not the same as independently retraining the models.** To reproduce the training, follow the next section.

## Reproduce model training on a GPU

The source dataset is not redistributed. Cite the dataset as:

> Turcan, Elsbeth, and Kathy McKeown. 2019. “Dreaddit: A Reddit Dataset for Stress Analysis in Social Media.” *Proceedings of the Tenth International Workshop on Health Text Mining and Information Analysis*, 97–107. https://doi.org/10.18653/v1/D19-6213.

Get the released train/test CSV files from the [official ACL Anthology entry and attachment](https://aclanthology.org/D19-6213/) (the author's data archive can also be fetched automatically by the full notebook). Place the extracted files here:

```text
data/
  dreaddit-train.csv
  dreaddit-test.csv
```

Do not combine the two files, resplit the official test set, or omit text-overlap removal. The preprocessing removes **18 repeated-text training rows** and **3 rows overlapping the official test set**; it then fixes a stratified 85/15 training/validation split using random state 42. The sizes are **2,394 training**, **423 validation**, and **715 untouched official test** instances. `reproduce_matched.py` verifies the SHA-256 text hashes of all three modeling splits against the recorded design before training.

On a CUDA GPU with Python 3.11+, install the pinned model-training dependencies:

```bash
python -m pip install -r requirements.txt
python reproduce_matched.py --backbone MPNet --seed 42
```

The default runs **both MPNet methods for one seed**, using the learning rates chosen in the original validation-only sweep. Its new output is written to `outputs/matched_reproduction/`; the historical results in `reviewer_results/` are never overwritten. For the full 60-run matched comparison, run:

```bash
python reproduce_matched.py --all-backbones --all-seeds
```

Expect substantial GPU compute, model downloads, and some numerical variation between GPU/software environments. The rerun script uses the recorded validation-selected learning rates. To **repeat the hyperparameter sweep itself**, use the complete notebook described below.

### Complete notebook and validation-only sweep

`Dreaddit_experiments.ipynb` preserves the original three-seed experiments and contains the full reviewer addendum at the end, including:

1. the validation-only learning-rate sweep (seed 42, three rates for each model/method: 18 configurations);
2. matched-condition ten-seed retraining (three backbones × two methods × ten seeds = 60 runs);
3. exact paired sign-flip tests, seed-bootstrap 95% confidence intervals, Holm correction, temperature scaling on validation logits, and selective prediction analysis.

Open the notebook in **Google Colab with a CUDA GPU**, start from the first cell, and run sequentially. The first code cell installs and checks the notebook's dependencies, including handling the known incompatibility between older optional `torchao` packages and `peft`. `requirements-notebook.txt` additionally records the package versions captured during the completed A100 experiment (Python 3.13.16, PyTorch 2.11.0+cu130); the original `sentencepiece` version was not captured. The notebook is Colab-oriented and stores temporary experiment results under `/content/reviewer_revision_outputs/`, which is ephemeral unless you copy them to persistent storage. Executing the *entire* notebook runs many additional baseline experiments and may require an extended GPU session.

The notebook's **cell outputs, execution counters, widget state, and identifying collaborator reference have been removed** for anonymous review. This does not change the training code. Do not commit a personally linked, executed Colab notebook in place of this cleared copy.

### Initial three-seed replication (separate analysis)

`reproduce.py` is retained for the original, non-matched MPNet full-fine-tuning versus LoRA-large experiment:

```bash
python reproduce.py
```

It uses only seeds **42, 123, and 2026**, a maximum of **three epochs**, and *different effective batch sizes* between the two methods. Its results must **not** be cited as the matched ten-seed comparison. Results are stored separately in `outputs/initial_three_seed_*.csv`.

## Repository contents

| File or directory | Purpose |
|---|---|
| `Dreaddit_experiments.ipynb` | Full original experiment notebook **plus** reviewer follow-up; outputs cleared |
| `reproduce_matched.py` | Standalone CUDA runner for the matched, validation-selected configurations |
| `verify_matched_results.py` | CPU-only independent arithmetic check of archived predictions and summary claims |
| `reproduce.py` | Legacy three-seed MPNet experiment only |
| `reviewer_results/reviewer_validation_sweep.csv` | Validation-only learning-rate search records |
| `reviewer_results/reviewer_selected_hyperparameters.csv` | Frozen learning rates chosen before ten-seed testing |
| `reviewer_results/reviewer_matched_test_by_seed.csv` | Complete 60-row test results, trainable parameter counts, and run details |
| `reviewer_results/reviewer_matched_summary.csv` | Ten-seed means and sample standard deviations |
| `reviewer_results/reviewer_paired_statistics.csv` | Exact sign-flip tests, bootstrap CIs, Holm correction |
| `reviewer_results/reviewer_calibration_by_seed.csv` | Per-seed fitted temperatures, calibration errors, Brier scores |
| `reviewer_results/reviewer_selective_prediction_*.csv` | Per-seed and summarized confidence-threshold coverage/accuracy |
| `reviewer_results/seed_logits/*.npz` | All 60 sets of validation/test logits and binary labels; **no Reddit text** |
| `reviewer_results/reviewer_run_design.json` | Hyperparameters, seeds, split hashes |
| `reviewer_results/reviewer_data_cleaning_audit.csv` | Dataset leakage-removal audit |
| `reviewer_results/reviewer_published_dreaddit_baselines.csv` | Published external Dreaddit baseline figures, **not** rerun here |
| `reviewer_results/reviewer_mpnet_accuracy_coverage.png` | Figure from matched post-calibration confidence analysis |
| `requirements.txt` / `requirements-verify.txt` / `requirements-notebook.txt` | Matched GPU, CPU verification, and observed complete-notebook environment |
| `tests/` | Lightweight artifact integrity checks |

The source texts are obtained from the original dataset authors, not embedded in this repository. The public archived arrays contain prediction scores and binary labels only. The repository is intended for anonymous peer review; no author, institution, or reviewer identity is provided.
