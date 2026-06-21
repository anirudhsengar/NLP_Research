# Interpretable and Bias-Aware AI-Text Detection

This repository contains the code and experiment pipeline for the CSAI411 project proposal. It builds an interpretable detector for AI-generated text, evaluates out-of-domain robustness, and audits false positives on learner-English writing.

The research paper is intentionally out of scope here. The deliverables are code, saved models, result tables, figures, and a Gradio demo.

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

Run the Alikhanov et al. anchor-paper comparison on the hydrated HC3+DAIGT
dataset:

```bash
uv run aidetect anchor-study
```

For a shorter artifact pass that still reproduces the exact M0 grid but samples
external audits and skips leave-one-domain retraining:

```bash
uv run aidetect anchor-study --max-external-samples 30000 --skip-leave-one-domain --no-grid-search-extensions
```

The command writes `reports/results/anchor_study_summary.csv`,
`anchor_grid_results.csv`, `anchor_threshold_metrics.csv`,
`anchor_stat_tests.csv`, `anchor_subgroup_fpr_ci.csv`, and
`anchor_calibration_metrics.csv`. The exact M0 reproduction can take several
minutes because it runs the paper's full 5-fold TF-IDF/logistic-regression grid.

Calibrate a conservative education-use threshold and evaluate the mitigation policy on held-out audit data:

```bash
uv run aidetect calibrate
```

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

## Outputs

- Trained model: `artifacts/models/classical_logreg.joblib`
- Metrics and prediction CSVs: `reports/results/`
- Consolidated result table: `reports/results/summary_table.csv`
- Baseline result table: `reports/results/baseline_summary_table.csv`
- Calibrated mitigation table: `reports/results/calibrated_summary_table.csv`
- Calibration policy/report: `reports/results/calibration_policy.json`, `reports/results/calibration_threshold_report.csv`
- Data/run provenance: `reports/results/data_profile.json`, `reports/results/run_manifest.json`
- Feature ablation table: `reports/results/ablation_metrics.csv`
- Grouped feature importance: `reports/results/logreg_feature_group_summary.csv`
- Fixed-sample model comparison: `reports/results/model_comparison_table.csv`
- Alikhanov anchor comparison: `reports/results/anchor_study_summary.csv`
- Paper artifact checklist: `reports/results/paper_readiness_summary.json`
- Per-evaluation error lists: `reports/results/*_errors.csv`
- Figures: `reports/figures/`
- Demo: <http://127.0.0.1:7860>

## Tests

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp uv run pytest
```

The explicit temp variables avoid pytest capture problems when WSL inherits Windows
`TMP`/`TEMP` paths.
