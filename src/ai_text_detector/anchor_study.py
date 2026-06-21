from __future__ import annotations

import json
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from ai_text_detector.calibration import split_icnale_calibration_audit
from ai_text_detector.features import feature_names, make_feature_pipeline
from ai_text_detector.metrics import (
    bootstrap_metric_intervals,
    calibration_diagnostics,
    compute_binary_metrics,
    compute_human_only_metrics,
    mcnemar_test,
    paired_bootstrap_auroc_difference,
    subgroup_fpr_intervals,
    threshold_for_target_fpr,
)
from ai_text_detector.schema import LABEL_HUMAN, ensure_canonical, read_jsonl
from ai_text_detector.splitting import split_by_group


ANCHOR_TRAIN_SOURCES = [
    "HC3_reddit_eli5",
    "HC3_finance",
    "DAIGT_v2_Distance learning",
    "DAIGT_v2_Seeking multiple opinions",
    "HC3_open_qa",
]

ANCHOR_VALIDATION_SOURCES = [
    "DAIGT_v2_Car-free cities",
    "DAIGT_v2_Does the electoral college work?",
    "DAIGT_v2_Facial action coding system",
    "DAIGT_v2_Mandatory extracurricular activities",
    "DAIGT_v2_Summer projects",
    "HC3_medicine",
    "DAIGT_v2_Driverless cars",
    "DAIGT_v2_Exploring Venus",
]

ANCHOR_TEST_SOURCES = [
    "DAIGT_v2_Cell phones at school",
    "DAIGT_v2_Grades for extracurricular activities",
    "DAIGT_v2_Community service",
    'DAIGT_v2_"A Cowboy Who Rode the Waves"',
    "DAIGT_v2_The Face on Mars",
    "HC3_wiki_csai",
    "DAIGT_v2_Phones and driving",
]

ANCHOR_EXPECTED_SPLIT_COUNTS = {
    "train": 85897,
    "validation": 24987,
    "test": 13311,
}

ANCHOR_HC3_DOMAINS = ["reddit_eli5", "finance", "open_qa", "medicine", "wiki_csai"]


@dataclass
class AnchorModel:
    name: str
    pipeline: Pipeline
    input_kind: str
    train_seconds: float
    best_params: dict[str, Any]
    validation_threshold: float
    high_confidence_threshold: float


def load_anchor_dataset(path: str | Path) -> pd.DataFrame:
    """Load the hydrated HC3+DAIGT dataset used by the anchor paper."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Anchor dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"text", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Anchor dataset is missing required columns: {sorted(missing)}")

    work = df[["text", "label", "source"]].copy()
    work["sample_id"] = [f"anchor:{idx}" for idx in range(len(work))]
    work["dataset"] = "anchor_hc3_daigt"
    work["split"] = ""
    work["domain"] = work["source"].astype(str)
    work["group_id"] = work["sample_id"]
    work["license_tag"] = "HC3/DAIGT source-dependent"
    return ensure_canonical(work)


def apply_anchor_source_split(
    df: pd.DataFrame,
    *,
    verify_expected_counts: bool = True,
) -> pd.DataFrame:
    """Apply the exact topic/source split used in Alikhanov et al.'s notebooks."""
    _validate_source_lists()
    split_map = {source: "train" for source in ANCHOR_TRAIN_SOURCES}
    split_map.update({source: "validation" for source in ANCHOR_VALIDATION_SOURCES})
    split_map.update({source: "test" for source in ANCHOR_TEST_SOURCES})

    work = df.copy()
    work["split"] = work["source"].map(split_map).fillna("")
    unassigned = sorted(work.loc[work["split"] == "", "source"].dropna().astype(str).unique())
    if unassigned:
        raise ValueError(f"Anchor source split leaves sources unassigned: {unassigned}")

    if verify_expected_counts:
        counts = work["split"].value_counts().to_dict()
        for split_name, expected in ANCHOR_EXPECTED_SPLIT_COUNTS.items():
            observed = int(counts.get(split_name, 0))
            if observed != expected:
                raise ValueError(
                    f"Anchor {split_name} split has {observed} rows; expected {expected}."
                )
    return work.reset_index(drop=True)


def make_alikhanov_baseline_pipeline(
    *,
    max_features: int | None = None,
    c_value: float = 1.0,
    penalty: str = "l2",
    random_seed: int = 42,
) -> Pipeline:
    """Exact word TF-IDF + logistic-regression baseline family from the anchor paper."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    random_state=random_seed,
                    solver="liblinear",
                    C=float(c_value),
                    penalty=penalty,
                    class_weight=None,
                ),
            ),
        ]
    )


def run_anchor_study(
    config: dict[str, Any],
    *,
    anchor_data_path: str | Path | None = None,
    models: list[str] | None = None,
    max_external_samples: int | None = None,
    bootstrap_iterations: int | None = None,
    paired_bootstrap_iterations: int = 300,
    run_leave_one_domain: bool = True,
    grid_search_extensions: bool = True,
    permutation_sample_size: int = 2000,
    verify_anchor_counts: bool = True,
) -> dict[str, Path]:
    """Run the paper-ready Alikhanov anchor comparison."""
    models = models or ["m0", "m1", "m2"]
    random_seed = int(config.get("random_seed", 42))
    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    figures_dir = Path(config["paths"]["reports_dir"]) / "figures"
    models_dir = Path(config["paths"]["artifacts_dir"]) / "models"
    paper_figures_dir = Path(config.get("_project_root", Path.cwd())) / "paper" / "figures"
    for directory in (results_dir, figures_dir, models_dir, paper_figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if anchor_data_path is None:
        anchor_data_path = Path(config.get("_project_root", Path.cwd())) / "data/raw/anchor_merged/merged_dataset.csv"
    anchor = apply_anchor_source_split(
        load_anchor_dataset(anchor_data_path),
        verify_expected_counts=verify_anchor_counts,
    )
    train_df = anchor[anchor["split"] == "train"].copy()
    validation_df = anchor[anchor["split"] == "validation"].copy()
    test_df = anchor[anchor["split"] == "test"].copy()

    eval_frames = _load_anchor_eval_frames(
        config,
        anchor_validation=validation_df,
        anchor_test=test_df,
        max_external_samples=max_external_samples,
        seed=random_seed,
    )
    icnale_calibration = eval_frames.pop("_icnale_calibration", pd.DataFrame())
    icnale_audit = eval_frames.pop("_icnale_audit", pd.DataFrame())
    if not icnale_audit.empty:
        eval_frames["icnale_fairness_audit"] = icnale_audit

    trained_models = []
    grid_rows = []
    for model_code in models:
        model = _train_anchor_model(
            model_code,
            train_df,
            validation_df,
            icnale_calibration=icnale_calibration,
            config=config,
            grid_search_extensions=grid_search_extensions,
        )
        trained_models.append(model)
        model_path = models_dir / f"anchor_{model.name}.joblib"
        joblib.dump(model, model_path)
        grid_rows.extend(_grid_rows_for_model(model.name, model.pipeline))

    grid_path = results_dir / "anchor_grid_results.csv"
    pd.DataFrame(grid_rows).to_csv(grid_path, index=False)

    score_cache: dict[tuple[str, str], np.ndarray] = {}
    threshold_rows = []
    subgroup_rows = []
    calibration_rows = []
    prediction_cache: dict[tuple[str, str, str], np.ndarray] = {}
    for model in trained_models:
        calibrators = _fit_score_calibrators(model, validation_df)
        for eval_name, frame in eval_frames.items():
            scores, infer_seconds = _timed_scores(model, frame, config)
            score_cache[(model.name, eval_name)] = scores
            thresholds = {
                "default_0.5": 0.5,
                "validation_target_fpr": model.validation_threshold,
                "conservative_review": model.high_confidence_threshold,
            }
            for threshold_policy, threshold in thresholds.items():
                metrics = _metrics_for_frame(
                    frame,
                    scores,
                    threshold=threshold,
                    target_fpr=float(config["model"].get("target_fpr", 0.01)),
                    bootstrap_iterations=int(
                        bootstrap_iterations
                        if bootstrap_iterations is not None
                        else config.get("evaluation", {}).get("bootstrap_iterations", 0)
                    ),
                    seed=random_seed,
                )
                prediction_cache[(model.name, eval_name, threshold_policy)] = (
                    scores >= threshold
                ).astype(int)
                threshold_rows.append(
                    {
                        "model": model.name,
                        "eval_set": eval_name,
                        "threshold_policy": threshold_policy,
                        "threshold": threshold,
                        "train_seconds": model.train_seconds,
                        "inference_seconds": infer_seconds,
                        "best_params": json.dumps(model.best_params, sort_keys=True),
                        **metrics,
                    }
                )
                if eval_name == "icnale_fairness_audit":
                    subgroup = subgroup_fpr_intervals(
                        frame,
                        scores,
                        threshold=threshold,
                        group_cols=["native_status", "cefr_level", "l1", "country", "domain", "length_bin"],
                        min_group_size=int(config["evaluation"].get("min_subgroup_size", 30)),
                    )
                    if not subgroup.empty:
                        subgroup.insert(0, "threshold_policy", threshold_policy)
                        subgroup.insert(0, "eval_set", eval_name)
                        subgroup.insert(0, "model", model.name)
                        subgroup_rows.extend(subgroup.to_dict(orient="records"))
            calibration_rows.extend(
                _calibration_rows_for_model(model, calibrators, eval_name, frame, scores)
            )

    threshold_path = results_dir / "anchor_threshold_metrics.csv"
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(threshold_path, index=False)

    summary_path = results_dir / "anchor_study_summary.csv"
    _summary_from_thresholds(threshold_df).to_csv(summary_path, index=False)

    subgroup_path = results_dir / "anchor_subgroup_fpr_ci.csv"
    pd.DataFrame(subgroup_rows).to_csv(subgroup_path, index=False)

    calibration_path = results_dir / "anchor_calibration_metrics.csv"
    pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)

    stat_path = results_dir / "anchor_stat_tests.csv"
    _paired_stat_rows(
        trained_models,
        eval_frames,
        score_cache,
        prediction_cache,
        paired_bootstrap_iterations=paired_bootstrap_iterations,
        seed=random_seed,
    ).to_csv(stat_path, index=False)

    leave_one_path = results_dir / "anchor_leave_one_domain_results.csv"
    if run_leave_one_domain:
        _leave_one_domain_results(
            trained_models,
            config,
            bootstrap_iterations=bootstrap_iterations or 0,
        ).to_csv(leave_one_path, index=False)
    else:
        pd.DataFrame().to_csv(leave_one_path, index=False)

    importance_path = results_dir / "anchor_grouped_permutation_importance.csv"
    contribution_path = results_dir / "anchor_icnale_false_positive_features.csv"
    _write_interpretability_outputs(
        trained_models,
        eval_frames,
        score_cache,
        config,
        importance_path=importance_path,
        contribution_path=contribution_path,
        permutation_sample_size=permutation_sample_size,
        seed=random_seed,
    )

    figure_paths = _save_anchor_figures(
        threshold_df,
        calibration_path=calibration_path,
        leave_one_path=leave_one_path,
        importance_path=importance_path,
        figures_dir=figures_dir,
        paper_figures_dir=paper_figures_dir,
    )

    readiness_path = results_dir / "anchor_study_readiness.json"
    _write_json(
        readiness_path,
        {
            "anchor_dataset": str(anchor_data_path),
            "split_counts": anchor["split"].value_counts().to_dict(),
            "models": [model.name for model in trained_models],
            "outputs": {
                "summary": str(summary_path),
                "grid": str(grid_path),
                "threshold_metrics": str(threshold_path),
                "stat_tests": str(stat_path),
                "subgroup_fpr_ci": str(subgroup_path),
                "calibration_metrics": str(calibration_path),
                "leave_one_domain": str(leave_one_path),
                "grouped_permutation_importance": str(importance_path),
                "icnale_false_positive_features": str(contribution_path),
                **{key: str(value) for key, value in figure_paths.items()},
            },
            "acceptance_note": _acceptance_note(threshold_df),
        },
    )

    return {
        "anchor_study_summary": summary_path,
        "anchor_grid_results": grid_path,
        "anchor_threshold_metrics": threshold_path,
        "anchor_stat_tests": stat_path,
        "anchor_subgroup_fpr_ci": subgroup_path,
        "anchor_calibration_metrics": calibration_path,
        "anchor_leave_one_domain_results": leave_one_path,
        "anchor_grouped_permutation_importance": importance_path,
        "anchor_icnale_false_positive_features": contribution_path,
        "anchor_study_readiness": readiness_path,
        **figure_paths,
    }


def _train_anchor_model(
    model_code: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    icnale_calibration: pd.DataFrame,
    config: dict[str, Any],
    grid_search_extensions: bool,
) -> AnchorModel:
    random_seed = int(config.get("random_seed", 42))
    canonical = model_code.lower().strip()
    if canonical in {"m0", "alikhanov", "alikhanov_style_word_tfidf_lr"}:
        name = "alikhanov_style_word_tfidf_lr"
        estimator = make_alikhanov_baseline_pipeline(random_seed=random_seed)
        param_grid = {
            "tfidf__max_features": [15000, 25000, 35000],
            "clf__C": [0.1, 1, 10],
            "clf__penalty": ["l1", "l2"],
        }
        fit_x = train_df["text"].astype(str)
        input_kind = "text"
        search = GridSearchCV(
            estimator,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
        )
    elif canonical in {"m1", "word_char", "word_char_tfidf_lr"}:
        name = "word_char_tfidf_lr"
        estimator = _feature_pipeline_model(
            config,
            use_word=True,
            use_char=True,
            use_stats=False,
            random_seed=random_seed,
        )
        fit_x = train_df
        input_kind = "frame"
        search = _extension_search(estimator) if grid_search_extensions else estimator
    elif canonical in {"m2", "word_char_stats", "word_char_stats_lr"}:
        name = "word_char_stats_lr"
        estimator = _feature_pipeline_model(
            config,
            use_word=True,
            use_char=True,
            use_stats=True,
            random_seed=random_seed,
        )
        fit_x = train_df
        input_kind = "frame"
        search = _extension_search(estimator) if grid_search_extensions else estimator
    else:
        raise ValueError(f"Unknown anchor model code: {model_code}")

    start = time.perf_counter()
    with _suppress_logistic_penalty_warnings():
        search.fit(fit_x, train_df["label"].astype(int))
    train_seconds = time.perf_counter() - start
    pipeline = search.best_estimator_ if isinstance(search, GridSearchCV) else search
    best_params = search.best_params_ if isinstance(search, GridSearchCV) else _params_from_pipeline(pipeline)
    validation_scores = _score_pipeline(pipeline, input_kind, validation_df)
    validation_threshold = threshold_for_target_fpr(
        validation_df["label"].astype(int),
        validation_scores,
        target_fpr=float(config["model"].get("target_fpr", 0.01)),
    )
    high_threshold = _conservative_threshold(
        validation_df,
        validation_scores,
        icnale_calibration=icnale_calibration,
        model_pipeline=pipeline,
        input_kind=input_kind,
        target_fpr=float(config["model"].get("target_fpr", 0.01)),
    )
    if isinstance(search, GridSearchCV):
        pipeline.anchor_cv_results_ = search.cv_results_
    return AnchorModel(
        name=name,
        pipeline=pipeline,
        input_kind=input_kind,
        train_seconds=train_seconds,
        best_params=dict(best_params),
        validation_threshold=float(validation_threshold),
        high_confidence_threshold=float(high_threshold),
    )


def _feature_pipeline_model(
    config: dict[str, Any],
    *,
    use_word: bool,
    use_char: bool,
    use_stats: bool,
    random_seed: int,
) -> Pipeline:
    feature_config = dict(config["features"])
    feature_config.update(
        {
            "use_word_tfidf": use_word,
            "use_char_tfidf": use_char,
            "use_basic_stats": use_stats,
            "use_spacy": False,
            "use_lm_stats": False,
        }
    )
    classifier = LogisticRegression(
        solver="liblinear",
        random_state=random_seed,
        class_weight=None,
        max_iter=int(config["model"].get("max_iter", 1000)),
    )
    return Pipeline([("features", make_feature_pipeline(feature_config)), ("clf", classifier)])


def _extension_search(estimator: Pipeline) -> GridSearchCV:
    return GridSearchCV(
        estimator,
        {
            "clf__C": [0.1, 1, 10],
            "clf__penalty": ["l1", "l2"],
        },
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )


def _score_pipeline(pipeline: Pipeline, input_kind: str, df: pd.DataFrame) -> np.ndarray:
    fit_x = df["text"].astype(str) if input_kind == "text" else df
    return pipeline.predict_proba(fit_x)[:, 1]


def _timed_scores(
    model: AnchorModel,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, float]:
    batch_size = int(config["evaluation"].get("batch_size", 5000))
    start = time.perf_counter()
    if not batch_size or len(frame) <= batch_size:
        scores = _score_pipeline(model.pipeline, model.input_kind, frame)
    else:
        scores = []
        for batch_start in range(0, len(frame), batch_size):
            batch = frame.iloc[batch_start : batch_start + batch_size]
            scores.append(_score_pipeline(model.pipeline, model.input_kind, batch))
        scores = np.concatenate(scores)
    return np.asarray(scores, dtype=float), time.perf_counter() - start


def _metrics_for_frame(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold: float,
    target_fpr: float,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    labels = frame["label"].astype(int)
    if set(labels.unique()) == {LABEL_HUMAN}:
        metrics = compute_human_only_metrics(scores, threshold=threshold, target_fpr=target_fpr)
    else:
        metrics = compute_binary_metrics(labels, scores, threshold=threshold, target_fpr=target_fpr)
    if bootstrap_iterations > 0:
        metrics.update(
            bootstrap_metric_intervals(
                labels,
                scores,
                threshold=threshold,
                target_fpr=target_fpr,
                iterations=bootstrap_iterations,
                seed=seed,
            )
        )
    return metrics


def _load_anchor_eval_frames(
    config: dict[str, Any],
    *,
    anchor_validation: pd.DataFrame,
    anchor_test: pd.DataFrame,
    max_external_samples: int | None,
    seed: int,
) -> dict[str, pd.DataFrame]:
    frames = {
        "anchor_validation": anchor_validation.copy(),
        "anchor_test": anchor_test.copy(),
    }
    processed_dir = Path(config["paths"]["processed_dir"])
    hc3_path = processed_dir / "hc3.jsonl"
    if hc3_path.exists():
        hc3 = read_jsonl(hc3_path)
        frames["current_hc3_group_test"] = hc3[hc3["split"] == "test"].copy()
    gptwiki_path = processed_dir / "gpt_wiki_intro.jsonl"
    if gptwiki_path.exists():
        frames["gpt_wiki_intro_ood"] = _sample_eval_frame(
            read_jsonl(gptwiki_path),
            max_samples=max_external_samples,
            seed=seed,
        )
    icnale_path = processed_dir / "icnale_fairness.jsonl"
    if icnale_path.exists():
        icnale = read_jsonl(icnale_path)
        calibration, audit = split_icnale_calibration_audit(
            icnale,
            calibration_fraction=float(config.get("calibration", {}).get("icnale_calibration_fraction", 0.3)),
            seed=seed,
        )
        frames["_icnale_calibration"] = calibration
        frames["_icnale_audit"] = _sample_eval_frame(audit, max_samples=max_external_samples, seed=seed)
    return {name: frame for name, frame in frames.items() if not frame.empty}


def _sample_eval_frame(
    frame: pd.DataFrame,
    *,
    max_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    if not max_samples or max_samples <= 0 or len(frame) <= max_samples:
        return frame.copy()
    if frame["label"].nunique(dropna=False) <= 1 and "native_status" in frame.columns:
        return _balanced_sample(frame, "native_status", max_samples=max_samples, seed=seed)
    return _balanced_sample(frame, "label", max_samples=max_samples, seed=seed)


def _balanced_sample(df: pd.DataFrame, column: str, *, max_samples: int, seed: int) -> pd.DataFrame:
    values = df[column].fillna("unknown").astype(str)
    per_group = max(1, max_samples // max(1, values.nunique()))
    sampled_indices = []
    for _, group in df.groupby(values):
        sampled_indices.extend(group.sample(min(len(group), per_group), random_state=seed).index.tolist())
    sampled = df.loc[sampled_indices]
    if len(sampled) < max_samples:
        remaining = df.drop(sampled.index, errors="ignore")
        if not remaining.empty:
            sampled = pd.concat(
                [sampled, remaining.sample(min(len(remaining), max_samples - len(sampled)), random_state=seed)]
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _conservative_threshold(
    validation_df: pd.DataFrame,
    validation_scores: np.ndarray,
    *,
    icnale_calibration: pd.DataFrame,
    model_pipeline: Pipeline,
    input_kind: str,
    target_fpr: float,
) -> float:
    thresholds = [
        threshold_for_target_fpr(validation_df["label"].astype(int), validation_scores, target_fpr=target_fpr)
    ]
    validation_human_scores = validation_scores[validation_df["label"].astype(int).to_numpy() == LABEL_HUMAN]
    if len(validation_human_scores):
        thresholds.append(
            threshold_for_target_fpr([LABEL_HUMAN] * len(validation_human_scores), validation_human_scores, target_fpr)
        )
    if not icnale_calibration.empty:
        icnale_scores = _score_pipeline(model_pipeline, input_kind, icnale_calibration)
        thresholds.append(
            threshold_for_target_fpr([LABEL_HUMAN] * len(icnale_scores), icnale_scores, target_fpr)
        )
        learner_scores = icnale_scores[
            icnale_calibration["native_status"].fillna("").astype(str).to_numpy() == "learner"
        ]
        if len(learner_scores):
            thresholds.append(
                threshold_for_target_fpr([LABEL_HUMAN] * len(learner_scores), learner_scores, target_fpr)
            )
    return float(max(thresholds))


def _grid_rows_for_model(model_name: str, pipeline: Pipeline) -> list[dict[str, Any]]:
    cv_results = getattr(pipeline, "anchor_cv_results_", None)
    if not cv_results:
        return [{"model": model_name, "grid_search": False}]
    rows = pd.DataFrame(cv_results).to_dict(orient="records")
    for row in rows:
        row["model"] = model_name
        row["grid_search"] = True
    return rows


def _summary_from_thresholds(threshold_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "eval_set",
        "threshold_policy",
        "n",
        "accuracy",
        "f1_macro",
        "human_f1",
        "ai_f1",
        "auroc",
        "aupr",
        "fpr",
        "fnr",
        "tpr",
        "train_seconds",
        "inference_seconds",
        "best_params",
    ]
    return threshold_df[[col for col in columns if col in threshold_df.columns]].copy()


def _paired_stat_rows(
    models: list[AnchorModel],
    eval_frames: dict[str, pd.DataFrame],
    score_cache: dict[tuple[str, str], np.ndarray],
    prediction_cache: dict[tuple[str, str, str], np.ndarray],
    *,
    paired_bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    if not models:
        return pd.DataFrame()
    baseline = next((model for model in models if model.name == "alikhanov_style_word_tfidf_lr"), models[0])
    rows = []
    for challenger in models:
        if challenger.name == baseline.name:
            continue
        for eval_name, frame in eval_frames.items():
            labels = frame["label"].astype(int).to_numpy()
            if len(np.unique(labels)) < 2:
                continue
            for policy in ("default_0.5", "validation_target_fpr", "conservative_review"):
                base_pred = prediction_cache.get((baseline.name, eval_name, policy))
                challenger_pred = prediction_cache.get((challenger.name, eval_name, policy))
                if base_pred is None or challenger_pred is None:
                    continue
                row = {
                    "eval_set": eval_name,
                    "threshold_policy": policy,
                    "baseline_model": baseline.name,
                    "challenger_model": challenger.name,
                    **mcnemar_test(labels, base_pred, challenger_pred),
                    **paired_bootstrap_auroc_difference(
                        labels,
                        score_cache[(challenger.name, eval_name)],
                        score_cache[(baseline.name, eval_name)],
                        iterations=paired_bootstrap_iterations,
                        seed=seed,
                    ),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _fit_score_calibrators(model: AnchorModel, validation_df: pd.DataFrame) -> dict[str, Any]:
    labels = validation_df["label"].astype(int).to_numpy()
    scores = _score_pipeline(model.pipeline, model.input_kind, validation_df)
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(scores.reshape(-1, 1), labels)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(scores, labels)
    return {"raw": None, "platt": platt, "isotonic": isotonic}


def _calibration_rows_for_model(
    model: AnchorModel,
    calibrators: dict[str, Any],
    eval_name: str,
    frame: pd.DataFrame,
    raw_scores: np.ndarray,
) -> list[dict[str, Any]]:
    labels = frame["label"].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        return []
    rows = []
    for calibration_method, calibrator in calibrators.items():
        if calibration_method == "raw":
            scores = raw_scores
        elif calibration_method == "platt":
            scores = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        else:
            scores = calibrator.predict(raw_scores)
        rows.append(
            {
                "model": model.name,
                "eval_set": eval_name,
                "calibration_method": calibration_method,
                **calibration_diagnostics(labels, scores, n_bins=10),
            }
        )
    return rows


def _leave_one_domain_results(
    models: list[AnchorModel],
    config: dict[str, Any],
    *,
    bootstrap_iterations: int,
) -> pd.DataFrame:
    hc3_path = Path(config["paths"]["processed_dir"]) / "hc3.jsonl"
    if not hc3_path.exists():
        return pd.DataFrame()
    hc3 = read_jsonl(hc3_path)
    rows = []
    for held_out in ANCHOR_HC3_DOMAINS:
        candidates = hc3[hc3["domain"].astype(str) != held_out].copy()
        held_out_df = hc3[hc3["domain"].astype(str) == held_out].copy()
        split_train_val = split_by_group(
            candidates,
            train_size=0.8,
            validation_size=0.1,
            test_size=0.1,
            seed=int(config.get("random_seed", 42)),
        )
        train_df = split_train_val[split_train_val["split"] == "train"].copy()
        validation_df = split_train_val[split_train_val["split"] == "validation"].copy()
        for template in models:
            pipeline = _clone_with_best_params(template, config)
            fit_x = train_df["text"].astype(str) if template.input_kind == "text" else train_df
            start = time.perf_counter()
            with _suppress_logistic_penalty_warnings():
                pipeline.fit(fit_x, train_df["label"].astype(int))
            train_seconds = time.perf_counter() - start
            validation_scores = _score_pipeline(pipeline, template.input_kind, validation_df)
            threshold = threshold_for_target_fpr(
                validation_df["label"].astype(int),
                validation_scores,
                target_fpr=float(config["model"].get("target_fpr", 0.01)),
            )
            scores = _score_pipeline(pipeline, template.input_kind, held_out_df)
            metrics = _metrics_for_frame(
                held_out_df,
                scores,
                threshold=threshold,
                target_fpr=float(config["model"].get("target_fpr", 0.01)),
                bootstrap_iterations=bootstrap_iterations,
                seed=int(config.get("random_seed", 42)),
            )
            rows.append(
                {
                    "held_out_domain": held_out,
                    "model": template.name,
                    "n_train": len(train_df),
                    "n_validation": len(validation_df),
                    "n_test": len(held_out_df),
                    "threshold": threshold,
                    "train_seconds": train_seconds,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _clone_with_best_params(template: AnchorModel, config: dict[str, Any]) -> Pipeline:
    params = template.best_params
    if template.input_kind == "text":
        return make_alikhanov_baseline_pipeline(
            max_features=params.get("tfidf__max_features"),
            c_value=float(params.get("clf__C", 1.0)),
            penalty=str(params.get("clf__penalty", "l2")),
            random_seed=int(config.get("random_seed", 42)),
        )
    use_stats = template.name == "word_char_stats_lr"
    pipeline = _feature_pipeline_model(
        config,
        use_word=True,
        use_char=True,
        use_stats=use_stats,
        random_seed=int(config.get("random_seed", 42)),
    )
    pipeline.set_params(
        **{
            key: value
            for key, value in params.items()
            if key in {"clf__C", "clf__penalty"}
        }
    )
    return pipeline


def _write_interpretability_outputs(
    models: list[AnchorModel],
    eval_frames: dict[str, pd.DataFrame],
    score_cache: dict[tuple[str, str], np.ndarray],
    config: dict[str, Any],
    *,
    importance_path: Path,
    contribution_path: Path,
    permutation_sample_size: int,
    seed: int,
) -> None:
    importance_rows = []
    contribution_rows = []
    eval_frame = eval_frames.get("anchor_test")
    icnale_frame = eval_frames.get("icnale_fairness_audit")
    for model in models:
        if model.input_kind != "frame":
            continue
        if eval_frame is not None and permutation_sample_size > 0:
            sample = _sample_eval_frame(eval_frame, max_samples=permutation_sample_size, seed=seed)
            importance_rows.extend(
                _grouped_permutation_importance(
                    model,
                    sample,
                    threshold=model.validation_threshold,
                    seed=seed,
                    target_fpr=float(config["model"].get("target_fpr", 0.01)),
                )
            )
        if icnale_frame is not None:
            scores = score_cache.get((model.name, "icnale_fairness_audit"))
            if scores is None:
                continue
            contribution_rows.extend(
                _false_positive_feature_contributions(
                    model,
                    icnale_frame,
                    scores,
                    threshold=model.validation_threshold,
                    top_sample_count=100,
                    top_feature_count=100,
                )
            )
    pd.DataFrame(importance_rows).to_csv(importance_path, index=False)
    pd.DataFrame(contribution_rows).to_csv(contribution_path, index=False)


def _grouped_permutation_importance(
    model: AnchorModel,
    frame: pd.DataFrame,
    *,
    threshold: float,
    seed: int,
    target_fpr: float,
) -> list[dict[str, Any]]:
    features = model.pipeline.named_steps["features"]
    classifier = model.pipeline.named_steps["clf"]
    labels = frame["label"].astype(int)
    original_blocks = []
    for name, transformer in features.transformer_list:
        block = transformer.transform(frame)
        original_blocks.append((name, block if sparse.issparse(block) else sparse.csr_matrix(block)))
    base_matrix = sparse.hstack([block for _, block in original_blocks]).tocsr()
    base_scores = classifier.predict_proba(base_matrix)[:, 1]
    base_metrics = compute_binary_metrics(labels, base_scores, threshold=threshold, target_fpr=target_fpr)
    rng = np.random.default_rng(seed)
    rows = []
    for group_name, _ in original_blocks:
        shuffled_blocks = []
        for name, block in original_blocks:
            if name == group_name:
                shuffled_blocks.append(block[rng.permutation(block.shape[0])])
            else:
                shuffled_blocks.append(block)
        permuted_matrix = sparse.hstack(shuffled_blocks).tocsr()
        permuted_scores = classifier.predict_proba(permuted_matrix)[:, 1]
        permuted_metrics = compute_binary_metrics(
            labels,
            permuted_scores,
            threshold=threshold,
            target_fpr=target_fpr,
        )
        rows.append(
            {
                "model": model.name,
                "eval_set": "anchor_test",
                "feature_family": group_name,
                "base_f1_macro": base_metrics["f1_macro"],
                "permuted_f1_macro": permuted_metrics["f1_macro"],
                "delta_f1_macro": base_metrics["f1_macro"] - permuted_metrics["f1_macro"],
                "base_auroc": base_metrics["auroc"],
                "permuted_auroc": permuted_metrics["auroc"],
                "delta_auroc": (
                    None
                    if base_metrics["auroc"] is None or permuted_metrics["auroc"] is None
                    else base_metrics["auroc"] - permuted_metrics["auroc"]
                ),
            }
        )
    return rows


def _false_positive_feature_contributions(
    model: AnchorModel,
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold: float,
    top_sample_count: int,
    top_feature_count: int,
) -> list[dict[str, Any]]:
    false_positive_mask = (frame["label"].astype(int).to_numpy() == LABEL_HUMAN) & (scores >= threshold)
    false_positives = frame.loc[false_positive_mask].copy()
    if false_positives.empty:
        return []
    false_positives["score_ai"] = scores[false_positive_mask]
    false_positives = false_positives.sort_values("score_ai", ascending=False).head(top_sample_count)
    features = model.pipeline.named_steps["features"]
    classifier = model.pipeline.named_steps["clf"]
    matrix = features.transform(false_positives)
    names = feature_names(features)
    coefs = classifier.coef_[0]
    contributions = matrix.multiply(coefs) if sparse.issparse(matrix) else np.asarray(matrix) * coefs
    mean_contribution = np.asarray(contributions.mean(axis=0)).ravel()
    if sparse.issparse(contributions):
        mean_abs = np.asarray(abs(contributions).mean(axis=0)).ravel()
        nonzero = np.asarray((contributions != 0).sum(axis=0)).ravel()
    else:
        mean_abs = np.abs(contributions).mean(axis=0)
        nonzero = (contributions != 0).sum(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_feature_count]
    rows = []
    for idx in order:
        rows.append(
            {
                "model": model.name,
                "feature": str(names[idx]),
                "feature_family": _feature_family(str(names[idx])),
                "mean_signed_contribution": float(mean_contribution[idx]),
                "mean_abs_contribution": float(mean_abs[idx]),
                "nonzero_count": int(nonzero[idx]),
                "false_positive_sample_count": int(len(false_positives)),
            }
        )
    return rows


def _feature_family(feature: str) -> str:
    if "__" in feature:
        return feature.split("__", 1)[0]
    if ":" in feature:
        return feature.split(":", 1)[0]
    return "other"


def _save_anchor_figures(
    threshold_df: pd.DataFrame,
    *,
    calibration_path: Path,
    leave_one_path: Path,
    importance_path: Path,
    figures_dir: Path,
    paper_figures_dir: Path,
) -> dict[str, Path]:
    outputs = {}
    outputs["anchor_model_comparison_figure"] = _save_model_comparison_figure(
        threshold_df,
        figures_dir / "anchor_model_comparison.png",
        paper_figures_dir / "anchor_model_comparison.png",
    )
    outputs["anchor_selective_workload_figure"] = _save_selective_workload_figure(
        threshold_df,
        figures_dir / "anchor_selective_workload.png",
        paper_figures_dir / "anchor_selective_workload.png",
    )
    outputs["anchor_calibration_figure"] = _save_calibration_figure(
        calibration_path,
        figures_dir / "anchor_calibration_metrics.png",
        paper_figures_dir / "anchor_calibration_metrics.png",
    )
    outputs["anchor_split_sensitivity_figure"] = _save_split_sensitivity_figure(
        leave_one_path,
        figures_dir / "anchor_split_sensitivity.png",
        paper_figures_dir / "anchor_split_sensitivity.png",
    )
    outputs["anchor_feature_family_figure"] = _save_importance_figure(
        importance_path,
        figures_dir / "anchor_feature_family_importance.png",
        paper_figures_dir / "anchor_feature_family_importance.png",
    )
    return outputs


def _save_model_comparison_figure(threshold_df: pd.DataFrame, report_path: Path, paper_path: Path) -> Path:
    plot = threshold_df[
        (threshold_df["eval_set"] == "anchor_test")
        & (threshold_df["threshold_policy"] == "default_0.5")
    ].copy()
    if plot.empty:
        return report_path
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(plot["model"], plot["accuracy"], color=["#4c78a8", "#f58518", "#54a24b"][: len(plot)])
    ax.axhline(0.8287, color="#111111", linestyle="--", linewidth=1, label="Alikhanov reported LR")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Anchor topic-split test accuracy")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_path, dpi=160)
    fig.savefig(paper_path, dpi=160)
    plt.close(fig)
    return report_path


def _save_selective_workload_figure(threshold_df: pd.DataFrame, report_path: Path, paper_path: Path) -> Path:
    plot = threshold_df[
        (threshold_df["eval_set"].isin(["anchor_test", "gpt_wiki_intro_ood", "icnale_fairness_audit"]))
        & (threshold_df["threshold_policy"].isin(["validation_target_fpr", "conservative_review"]))
    ].copy()
    if plot.empty:
        return report_path
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = plot["model"] + "\n" + plot["eval_set"] + "\n" + plot["threshold_policy"]
    ax.bar(np.arange(len(plot)), plot["fpr"], color="#b279a2")
    ax.set_ylabel("Human false-positive rate")
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(0, min(1.0, max(0.05, float(plot["fpr"].max()) * 1.2)))
    fig.tight_layout()
    fig.savefig(report_path, dpi=160)
    fig.savefig(paper_path, dpi=160)
    plt.close(fig)
    return report_path


def _save_calibration_figure(calibration_path: Path, report_path: Path, paper_path: Path) -> Path:
    if not calibration_path.exists():
        return report_path
    df = _read_csv_or_empty(calibration_path)
    if df.empty or "eval_set" not in df.columns:
        return report_path
    plot = df[df["eval_set"] == "anchor_test"].copy()
    if plot.empty:
        return report_path
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, group in plot.groupby("calibration_method"):
        ax.plot(group["model"], group["ece"], marker="o", label=method)
    ax.set_ylabel("Expected calibration error")
    ax.set_title("Anchor test calibration diagnostics")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_path, dpi=160)
    fig.savefig(paper_path, dpi=160)
    plt.close(fig)
    return report_path


def _save_split_sensitivity_figure(leave_one_path: Path, report_path: Path, paper_path: Path) -> Path:
    if not leave_one_path.exists():
        return report_path
    df = _read_csv_or_empty(leave_one_path)
    if df.empty or "f1_macro" not in df.columns:
        return report_path
    fig, ax = plt.subplots(figsize=(9, 4))
    for model, group in df.groupby("model"):
        ax.plot(group["held_out_domain"], group["f1_macro"], marker="o", label=model)
    ax.set_ylabel("Macro-F1")
    ax.set_title("HC3 leave-one-domain-out sensitivity")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_path, dpi=160)
    fig.savefig(paper_path, dpi=160)
    plt.close(fig)
    return report_path


def _save_importance_figure(importance_path: Path, report_path: Path, paper_path: Path) -> Path:
    if not importance_path.exists():
        return report_path
    df = _read_csv_or_empty(importance_path)
    if df.empty:
        return report_path
    plot = (
        df.groupby(["model", "feature_family"], as_index=False)["delta_f1_macro"]
        .mean()
        .sort_values("delta_f1_macro")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = plot["model"] + ": " + plot["feature_family"]
    ax.barh(labels, plot["delta_f1_macro"], color="#72b7b2")
    ax.set_xlabel("Macro-F1 drop after grouped permutation")
    ax.set_title("Feature-family grouped permutation importance")
    fig.tight_layout()
    fig.savefig(report_path, dpi=160)
    fig.savefig(paper_path, dpi=160)
    plt.close(fig)
    return report_path


def _params_from_pipeline(pipeline: Pipeline) -> dict[str, Any]:
    params = {}
    if "clf" in pipeline.named_steps:
        clf = pipeline.named_steps["clf"]
        params["clf__C"] = clf.C
        params["clf__penalty"] = clf.penalty
    if "tfidf" in pipeline.named_steps:
        params["tfidf__max_features"] = pipeline.named_steps["tfidf"].max_features
    return params


def _validate_source_lists() -> None:
    all_sources = ANCHOR_TRAIN_SOURCES + ANCHOR_VALIDATION_SOURCES + ANCHOR_TEST_SOURCES
    if len(all_sources) != len(set(all_sources)):
        raise ValueError("Anchor split source lists contain overlap.")


def _acceptance_note(threshold_df: pd.DataFrame) -> dict[str, Any]:
    row = threshold_df[
        (threshold_df["model"] == "alikhanov_style_word_tfidf_lr")
        & (threshold_df["eval_set"] == "anchor_test")
        & (threshold_df["threshold_policy"] == "default_0.5")
    ]
    if row.empty:
        return {"m0_reproduction_checked": False}
    accuracy = float(row.iloc[0].get("accuracy"))
    return {
        "m0_reproduction_checked": True,
        "reported_accuracy": 0.8287,
        "observed_accuracy": accuracy,
        "within_half_percentage_point": abs(accuracy - 0.8287) <= 0.005,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


@contextmanager
def _suppress_logistic_penalty_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'penalty' was deprecated.*")
        warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*")
        yield
