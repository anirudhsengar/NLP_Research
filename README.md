# Interpretable and Bias-Aware AI-Text Detection

This repository contains the code, experiment pipeline, paper artifacts, and Gradio demo for the CSAI411 AI-text-detection project. It reproduces an HC3/DAIGT v2 TF-IDF logistic-regression baseline, extends it with a fairness-constrained three-way selective policy, evaluates out-of-domain robustness, and audits false positives on learner-English writing.

## Setup

Use `uv` with Python 3.11 or 3.12:

```bash
uv sync --extra dev
```

Optional transformer baselines:

```bash
uv sync --extra dev --extra baselines
```

Optional spaCy linguistic features:

```bash
uv sync --extra dev --extra spacy
uv run python -m spacy download en_core_web_sm
```

Optional GPT-2 perplexity / GLTR-style features use the `baselines` extra because they require
`torch` and `transformers`.

## Data

Keep the downloaded ICNALE files in the project root, as they are now:

- `ICNALE_Survey_202603.xlsx`
- `ICNALE_WE_2.6.zip`
- `ICNALE_WEP_0.7.zip`
- `ICNALE_WEUAE_1.0.zip`
- `ICNALE_WE_2.6/`
- `ICNALE_WEP_0.7/`
- `ICNALE_WEUAE_1.0/`

ICNALE text must not be redistributed. Raw and processed data paths are ignored by `.gitignore`.

The pipeline prefers the extracted directories. If only the encrypted ZIP archives are available, set the archive password in the environment:

```bash
export ICNALE_ZIP_PASSWORD='your-icnale-password'
```

HC3 and GPT-wiki-intro are fetched through Hugging Face `datasets`.

For the final paper-study reproduction, the default path uses the frozen merged
HC3/DAIGT CSV from the public anchor codebase:

```bash
git clone https://github.com/crusnix/ai_text_detector_final.git /tmp/ai_text_detector_final
# With Git LFS installed:
git -C /tmp/ai_text_detector_final lfs pull --include='data/merged_dataset(1).csv'
mkdir -p data/raw/anchor_merged
cp "/tmp/ai_text_detector_final/data/merged_dataset(1).csv" data/raw/anchor_merged/merged_dataset.csv
```

The expected columns are `text`, `label`, and `source`. Raw data paths are ignored
by git and should not be redistributed from this repository.

## Fast Reproduction Run

This runs a CPU-safe subset, trains the logistic-regression detector, and evaluates HC3 test, GPT-wiki-intro OOD, and ICNALE fairness:

```bash
uv run aidetect reproduce --sample-size 2000
```

If the ICNALE password is not available yet, run the public-data pipeline only:

```bash
uv run aidetect reproduce --sample-size 2000 --skip-icnale
```

Use full data for final classical results:

```bash
uv run aidetect reproduce --sample-size 0
```

Run optional transformer detector baselines after installing the `baselines` extra:

```bash
uv run aidetect baselines --max-samples 20000
```

Baseline runs now save the same artifact families as the classical model: metrics,
predictions, confusion matrices, subgroup FPR tables, error-analysis CSVs, and
`reports/results/baseline_summary_table.csv`. By default, baseline thresholds are
calibrated on the HC3 validation split at the configured target FPR.

Create a fixed-sample paper comparison for the saved classical model, and optionally
run baselines on exactly the same rows:

```bash
uv run aidetect compare --max-samples-per-set 2000
uv run aidetect compare --max-samples-per-set 2000 --run-baselines
```

The comparison command writes `comparison_sample_ids.csv`, classical comparison
metrics, optional baseline comparison metrics, `model_comparison_table.csv`, and
`paper_readiness_summary.json`. Use the sampled comparison for CPU-feasible paper
tables; full transformer inference over GPT-wiki-intro's 300k rows is possible but
usually not practical without GPU time.

Run feature-family ablations for the classical detector:

```bash
uv run aidetect ablations --max-eval-samples 5000
```

Include heavier linguistic or GPT-2/GLTR-style feature ablations only when those dependencies are installed:

```bash
uv run aidetect ablations --include-spacy --include-lm-stats --max-eval-samples 1000
```

Run the older binary-model calibration command when you need the legacy education-use threshold artifacts:

```bash
uv run aidetect calibrate
```

Run the final paper study. By default this uses the frozen anchor merged CSV and
the public notebook's fixed source split, compares sparse logistic-regression
feature families, calibrates a three-way selective policy, and writes the paper
tables and figures:

```bash
# Public-data smoke test without the frozen merged CSV.
uv run aidetect paper-study --sample-size 2000 --skip-daigt --bootstrap-iterations 20

# Anchor-compatible smoke test with the frozen merged CSV.
uv run aidetect paper-study --sample-size 2000 --bootstrap-iterations 20

# Full final run used for the paper artifacts.
uv run aidetect paper-study --sample-size 0 --bootstrap-iterations 1000
```

The `--skip-daigt` smoke command uses public HC3/GPT-wiki data without the frozen
anchor merge. The optional raw DAIGT v2 loader is still available at
`data/raw/daigt_v2/train_v2_drcat_02.csv`, but it is not the default paper-study
path because the anchor repository uses its own frozen merged CSV.

Regenerate the dataset provenance/count report from prepared files:

```bash
uv run aidetect profile
```

## Individual Commands

```bash
uv run aidetect prepare
uv run aidetect train
uv run aidetect eval
uv run aidetect demo
```

The demo command now defaults to the final paper-study model at
`artifacts/models/paper_study_selected.joblib`. Pass `--model-path
artifacts/models/classical_logreg.joblib` only when you intentionally want the
older binary-threshold model.

## Outputs

- Classical reproduction model: `artifacts/models/classical_logreg.joblib`
- Final selective paper-study model used by the demo: `artifacts/models/paper_study_selected.joblib`
- Metrics and prediction CSVs: `reports/results/`
- Consolidated result table: `reports/results/summary_table.csv`
- Baseline result table: `reports/results/baseline_summary_table.csv`
- Calibrated mitigation table: `reports/results/calibrated_summary_table.csv`
- Calibration policy/report: `reports/results/calibration_policy.json`, `reports/results/calibration_threshold_report.csv`
- Data/run provenance: `reports/results/data_profile.json`, `reports/results/run_manifest.json`
- Feature ablation table: `reports/results/ablation_metrics.csv`
- Grouped feature importance: `reports/results/logreg_feature_group_summary.csv`
- Fixed-sample model comparison: `reports/results/model_comparison_table.csv`
- Paper artifact checklist: `reports/results/paper_readiness_summary.json`
- Revised paper-study summary: `reports/results/paper_study_summary.csv`
- Three-way selective policy table: `reports/results/selective_policy_table.csv`
- Topic/source holdout table: `reports/results/topic_holdout_results.csv`
- Feature-family tradeoff table: `reports/results/feature_family_tradeoff.csv`
- Deterministic style-invariance table: `reports/results/style_invariance_results.csv`
- Per-evaluation error lists: `reports/results/*_errors.csv`
- Figures: `reports/figures/`
- Demo: <http://127.0.0.1:7860>

## Tests

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp uv run pytest
```

The explicit temp variables avoid pytest capture problems when WSL inherits Windows
`TMP`/`TEMP` paths.
