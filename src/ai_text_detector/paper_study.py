from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_text_detector.augmentation import transformed_frame
from ai_text_detector.baselines import run_transformer_baselines
from ai_text_detector.calibration import split_icnale_calibration_audit
from ai_text_detector.data import (
    load_anchor_merged,
    load_daigt_v2,
    load_gpt_wiki_intro,
    load_hc3,
    load_icnale,
)
from ai_text_detector.evaluation import save_feature_importance
from ai_text_detector.metrics import (
    bootstrap_metric_intervals,
    compute_binary_metrics,
    compute_human_only_metrics,
    compute_selective_metrics,
)
from ai_text_detector.model import DetectorBundle, save_bundle, train_detector
from ai_text_detector.reporting import save_data_profile, save_run_manifest
from ai_text_detector.schema import read_jsonl
from ai_text_detector.selective import (
    SelectivePolicy,
    calibrate_selective_policy,
    policy_satisfies_constraints,
)
from ai_text_detector.splitting import leave_one_domain_out_frames, split_by_anchor_source, split_by_topic


def run_paper_study(
    config: dict[str, Any],
    *,
    sample_size: int | None,
    skip_daigt: bool,
    skip_icnale: bool,
    bootstrap_iterations: int | None,
    run_baselines_flag: bool,
) -> dict[str, Path]:
    """Run the paper study and save report-ready artifacts."""
    cfg = copy.deepcopy(config)
    if bootstrap_iterations is not None:
        cfg.setdefault("evaluation", {})["bootstrap_iterations"] = int(bootstrap_iterations)
    requested_bootstrap_iterations = int(cfg.get("evaluation", {}).get("bootstrap_iterations", 0))
    loop_cfg = copy.deepcopy(cfg)
    loop_cfg.setdefault("evaluation", {})["bootstrap_iterations"] = 0
    _ensure_output_dirs(cfg)
    max_samples = None if sample_size in (None, 0) else int(sample_size)

    frames = _load_study_frames(cfg, max_samples=max_samples, skip_daigt=skip_daigt, skip_icnale=skip_icnale)
    profile_path = save_data_profile(frames, cfg)

    if "anchor_merged" in frames:
        hc3 = frames["anchor_merged"][frames["anchor_merged"]["dataset"] == "hc3"].copy()
        topic_frame = _build_topic_frame(frames["anchor_merged"], None, cfg)
    else:
        hc3 = frames["hc3"]
        daigt = frames.get("daigt_v2")
        topic_frame = _build_topic_frame(hc3, daigt, cfg)
    icnale_calibration, icnale_audit = _split_icnale_for_study(
        frames.get("icnale_fairness"),
        cfg=cfg,
        skip_icnale=skip_icnale,
    )
    eval_frames = {
        "topic_test": topic_frame[topic_frame["split"] == "test"].copy(),
        "gpt_wiki_intro_ood": frames["gpt_wiki_intro"].copy(),
    }
    if icnale_audit is not None and not icnale_audit.empty:
        eval_frames["icnale_fairness_audit"] = icnale_audit.copy()
    comparison_eval_frames = _comparison_eval_frames(cfg, eval_frames)

    variant_configs = _paper_variant_configs(cfg)
    bundles: dict[str, DetectorBundle] = {}
    policies: dict[str, SelectivePolicy] = {}
    summary_rows: list[dict[str, Any]] = []

    validation = topic_frame[topic_frame["split"] == "validation"].copy()
    human_calibration = _human_calibration_frame(validation, icnale_calibration)
    batch_size = int(cfg.get("evaluation", {}).get("batch_size", 5000))
    target_fpr = float(cfg.get("selective_policy", {}).get("target_fpr", 0.01))
    learner_target_fpr = float(cfg.get("selective_policy", {}).get("learner_target_fpr", target_fpr))

    for model_name, work_cfg in variant_configs.items():
        _progress(f"training {model_name}")
        bundle, validation_metrics = train_detector(topic_frame, work_cfg)
        bundles[model_name] = bundle
        _progress(f"calibrating selective policy for {model_name}")
        policy = calibrate_selective_policy(
            bundle,
            model_name=model_name,
            validation=validation,
            human_calibration=human_calibration,
            target_fpr=target_fpr,
            learner_target_fpr=learner_target_fpr,
            batch_size=batch_size,
        )
        policies[model_name] = policy
        summary_rows.append(
            {
                "model": model_name,
                "eval_set": "topic_validation",
                "split_mode": "topic",
                "selected_model": False,
                **_prefix("validation", validation_metrics),
                **policy.to_dict(),
            }
        )

    selected_model = _select_model(policies)
    _progress(f"selected model: {selected_model}")
    for model_name, bundle in bundles.items():
        policy = policies[model_name]
        frames_for_model = eval_frames if model_name == selected_model else comparison_eval_frames
        eval_cfg = copy.deepcopy(cfg if model_name == selected_model else loop_cfg)
        if model_name == selected_model:
            eval_cfg.setdefault("evaluation", {})["bootstrap_iterations"] = requested_bootstrap_iterations
        else:
            eval_cfg.setdefault("evaluation", {})["bootstrap_iterations"] = 0
        for eval_name, frame in frames_for_model.items():
            _progress(f"evaluating {model_name} on {eval_name} ({len(frame):,} rows)")
            summary_rows.append(
                _evaluation_row(
                    model_name=model_name,
                    eval_name=eval_name,
                    frame=frame,
                    bundle=bundle,
                    policy=policy,
                    cfg=eval_cfg,
                    split_mode="topic",
                    sampled=frame is comparison_eval_frames.get(eval_name),
                )
            )
    for row in summary_rows:
        row["selected_model"] = row.get("model") == selected_model
    selected_bundle = bundles[selected_model]
    selected_policy = policies[selected_model]
    selected_path = Path(cfg["paths"]["artifacts_dir"]) / "models" / "paper_study_selected.joblib"
    selected_bundle.metadata.update(
        {
            "selected_model": selected_model,
            "selective_policy": selected_policy.to_dict(),
            "paper_study": {
                "sample_size": sample_size,
                "skip_daigt": skip_daigt,
                "skip_icnale": skip_icnale,
                "split_mode": "topic",
            },
        }
    )
    save_bundle(selected_bundle, selected_path)
    save_feature_importance(selected_bundle, cfg)

    policy_rows = [
        {**policy.to_dict(), "constraints_satisfied": policy_satisfies_constraints(policy), "selected": name == selected_model}
        for name, policy in policies.items()
    ]
    policy_table = _write_csv(cfg, "selective_policy_table.csv", policy_rows)
    summary_table = _write_csv(cfg, "paper_study_summary.csv", summary_rows)
    feature_tradeoff = _write_csv(
        cfg,
        "feature_family_tradeoff.csv",
        [row for row in summary_rows if row.get("eval_set") != "topic_validation"],
    )
    _progress("running leave-one-HC3-source-out study")
    topic_holdout = _run_topic_holdout(hc3, variant_configs["word_char_stats_lr"], cfg)
    topic_holdout_table = _write_csv(cfg, "topic_holdout_results.csv", topic_holdout)
    _progress("running deterministic style-invariance study")
    style_rows = _run_style_invariance(
        bundles={name: bundles[name] for name in _style_model_names(selected_model, bundles)},
        policies=policies,
        eval_frames=eval_frames,
        cfg=cfg,
    )
    style_table = _write_csv(cfg, "style_invariance_results.csv", style_rows)
    _progress("writing paper-study figures")
    figures = _write_paper_study_figures(
        summary=pd.DataFrame(summary_rows),
        policies=pd.DataFrame(policy_rows),
        topic_holdout=pd.DataFrame(topic_holdout),
        style=pd.DataFrame(style_rows),
        cfg=cfg,
        selected_model=selected_model,
    )

    baseline_outputs: dict[str, Path] = {}
    if run_baselines_flag:
        baseline_outputs = run_transformer_baselines(
            eval_frames,
            cfg,
            max_samples=max_samples,
            output_prefix="paper_study_baseline",
            summary_filename="paper_study_baseline_summary_table.csv",
        )

    readiness_path = _write_readiness_summary(
        cfg,
        selected_model=selected_model,
        skip_daigt=skip_daigt,
        skip_icnale=skip_icnale,
        outputs={
            "paper_study_summary": summary_table,
            "selective_policy_table": policy_table,
            "topic_holdout_results": topic_holdout_table,
            "feature_family_tradeoff": feature_tradeoff,
            "style_invariance_results": style_table,
            **figures,
            **baseline_outputs,
        },
    )
    manifest_path = save_run_manifest(
        cfg,
        command="aidetect paper-study",
        outputs={
            "data_profile": profile_path,
            "selected_model": selected_path,
            "paper_study_summary": summary_table,
            "selective_policy_table": policy_table,
            "topic_holdout_results": topic_holdout_table,
            "feature_family_tradeoff": feature_tradeoff,
            "style_invariance_results": style_table,
            "paper_study_readiness": readiness_path,
            **figures,
            **baseline_outputs,
        },
        model_path=selected_path,
        model_metadata={
            "selected_model": selected_model,
            "selective_policy": selected_policy.to_dict(),
            "sample_size": sample_size,
            "skip_daigt": skip_daigt,
            "skip_icnale": skip_icnale,
        },
    )
    return {
        "data_profile": profile_path,
        "selected_model": selected_path,
        "paper_study_summary": summary_table,
        "selective_policy_table": policy_table,
        "topic_holdout_results": topic_holdout_table,
        "feature_family_tradeoff": feature_tradeoff,
        "style_invariance_results": style_table,
        "paper_study_readiness": readiness_path,
        "run_manifest": manifest_path,
        **figures,
        **baseline_outputs,
    }


def _load_study_frames(
    cfg: dict[str, Any],
    *,
    max_samples: int | None,
    skip_daigt: bool,
    skip_icnale: bool,
) -> dict[str, pd.DataFrame]:
    processed = Path(cfg["paths"]["processed_dir"])
    if bool(cfg.get("paper_study", {}).get("use_anchor_merged", False)) and not skip_daigt:
        frames = {
            "anchor_merged": load_anchor_merged(cfg, max_samples=max_samples),
            "gpt_wiki_intro": _load_prepared_or_raw(
                processed / "gpt_wiki_intro.jsonl",
                lambda: load_gpt_wiki_intro(cfg),
            ),
        }
        icnale_path = processed / "icnale_fairness.jsonl"
        if not skip_icnale:
            frames["icnale_fairness"] = _load_prepared_or_raw(icnale_path, lambda: load_icnale(cfg))
        if max_samples:
            frames = {
                name: _sample_for_study(frame, max_samples=max_samples, seed=int(cfg.get("random_seed", 42)))
                for name, frame in frames.items()
            }
        return frames

    frames = {
        "hc3": _load_prepared_or_raw(processed / "hc3.jsonl", lambda: load_hc3(cfg)),
        "gpt_wiki_intro": _load_prepared_or_raw(
            processed / "gpt_wiki_intro.jsonl",
            lambda: load_gpt_wiki_intro(cfg),
        ),
    }
    if not skip_daigt:
        frames["daigt_v2"] = load_daigt_v2(cfg)
    icnale_path = processed / "icnale_fairness.jsonl"
    if not skip_icnale:
        frames["icnale_fairness"] = _load_prepared_or_raw(icnale_path, lambda: load_icnale(cfg))
    if max_samples:
        frames = {
            name: _sample_for_study(frame, max_samples=max_samples, seed=int(cfg.get("random_seed", 42)))
            for name, frame in frames.items()
        }
    return frames


def _load_prepared_or_raw(path: Path, loader) -> pd.DataFrame:
    return read_jsonl(path) if path.exists() else loader()


def _split_icnale_for_study(
    icnale: pd.DataFrame | None,
    *,
    cfg: dict[str, Any],
    skip_icnale: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if skip_icnale or icnale is None or icnale.empty:
        return None, None
    return split_icnale_calibration_audit(
        icnale,
        calibration_fraction=float(cfg.get("calibration", {}).get("icnale_calibration_fraction", 0.3)),
        seed=int(cfg.get("random_seed", 42)),
    )


def _build_topic_frame(hc3: pd.DataFrame, daigt: pd.DataFrame | None, cfg: dict[str, Any]) -> pd.DataFrame:
    frames = [hc3.copy()]
    if daigt is not None and not daigt.empty:
        frames.append(daigt.copy())
    combined = pd.concat(frames, ignore_index=True)
    combined["topic_key"] = (
        combined["dataset"].astype(str)
        + ":"
        + combined["domain"].fillna("").astype(str).where(
            combined["domain"].fillna("").astype(str).str.len() > 0,
            combined["source"].fillna("").astype(str),
        )
    )
    split_cfg = cfg.get("splits", {})
    if str(split_cfg.get("mode", "")).lower() == "anchor_source":
        return split_by_anchor_source(combined)
    return split_by_topic(
        combined,
        topic_col="topic_key",
        train_size=float(split_cfg.get("train_size", 0.70)),
        validation_size=float(split_cfg.get("validation_size", 0.15)),
        test_size=float(split_cfg.get("test_size", 0.15)),
        seed=int(cfg.get("random_seed", 42)),
    )


def _human_calibration_frame(
    validation: pd.DataFrame,
    icnale_calibration: pd.DataFrame | None,
) -> pd.DataFrame:
    validation_human = validation[validation["label"].astype(int) == 0].copy()
    frames = [validation_human]
    if icnale_calibration is not None and not icnale_calibration.empty:
        frames.append(icnale_calibration.copy())
    return pd.concat(frames, ignore_index=True)


def _comparison_eval_frames(
    cfg: dict[str, Any],
    eval_frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    max_samples = int(cfg.get("paper_study", {}).get("comparison_eval_sample_size", 30000))
    seed = int(cfg.get("random_seed", 42))
    return {
        name: _sample_for_study(frame, max_samples=max_samples, seed=seed)
        for name, frame in eval_frames.items()
    }


def _paper_variant_configs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    def variant(
        name: str,
        *,
        word: bool,
        char: bool,
        stats: bool,
        penalty: str = "l2",
        l1_ratio: float | None = None,
        max_char_features: int | None = None,
        max_word_features: int | None = None,
        max_iter: int | None = None,
    ) -> dict[str, Any]:
        work = copy.deepcopy(cfg)
        work["features"].update(
            {
                "max_word_features": int(max_word_features)
                if max_word_features is not None
                else int(
                    cfg.get("paper_study", {}).get(
                        "max_word_features",
                        cfg["features"].get("max_word_features", 30000),
                    )
                ),
                "max_char_features": int(
                    cfg.get("paper_study", {}).get(
                        "max_char_features",
                        cfg["features"].get("max_char_features", 30000),
                    )
                ),
                "use_word_tfidf": word,
                "use_char_tfidf": char,
                "use_basic_stats": stats,
                "use_spacy": False,
                "use_lm_stats": False,
            }
        )
        if max_char_features is not None:
            work["features"]["max_char_features"] = int(max_char_features)
        work["model"].update(
            {
                "name": f"paper_study_{name}",
                "variant": name,
                "penalty": penalty,
                "l1_ratio": l1_ratio,
            }
        )
        if penalty == "elasticnet":
            work["model"]["max_iter"] = int(max_iter or 250)
            work["model"]["class_weight"] = None
        elif max_iter is not None:
            work["model"]["max_iter"] = int(max_iter)
        return work

    specs = {
        "alikhanov_style_word_tfidf_lr": variant(
            "alikhanov_style_word_tfidf_lr",
            word=True,
            char=False,
            stats=False,
        ),
        "word_char_stats_lr": variant("word_char_stats_lr", word=True, char=True, stats=True),
        "no_character_lr": variant("no_character_lr", word=True, char=False, stats=True),
        "stats_only_lr": variant("stats_only_lr", word=False, char=False, stats=True),
        "word_only_lr": variant("word_only_lr", word=True, char=False, stats=False),
        "char_capped_lr": variant(
            "char_capped_lr",
            word=True,
            char=True,
            stats=True,
            max_char_features=min(int(cfg["features"].get("max_char_features", 30000)), 5000),
        ),
    }
    if cfg.get("paper_study", {}).get("include_elastic_net", False):
        specs["elastic_net_lr"] = variant(
            "elastic_net_lr",
            word=True,
            char=True,
            stats=True,
            penalty="elasticnet",
            l1_ratio=0.5,
            max_word_features=5000,
            max_char_features=5000,
            max_iter=250,
        )
    return specs


def _evaluation_row(
    *,
    model_name: str,
    eval_name: str,
    frame: pd.DataFrame,
    bundle: DetectorBundle,
    policy: SelectivePolicy,
    cfg: dict[str, Any],
    split_mode: str,
    sampled: bool = False,
) -> dict[str, Any]:
    batch_size = int(cfg.get("evaluation", {}).get("batch_size", 5000))
    scores = bundle.score_frame(frame, batch_size=batch_size)
    labels = frame["label"].astype(int)
    target_fpr = float(cfg.get("model", {}).get("target_fpr", 0.01))
    if set(labels.unique()) == {0}:
        default_metrics = compute_human_only_metrics(scores, threshold=policy.low_threshold, target_fpr=target_fpr)
        high_metrics = compute_human_only_metrics(scores, threshold=policy.high_threshold, target_fpr=target_fpr)
    else:
        default_metrics = compute_binary_metrics(labels, scores, threshold=policy.low_threshold, target_fpr=target_fpr)
        high_metrics = compute_binary_metrics(labels, scores, threshold=policy.high_threshold, target_fpr=target_fpr)
    selective = compute_selective_metrics(
        labels,
        scores,
        low_threshold=policy.low_threshold,
        high_threshold=policy.high_threshold,
    )
    intervals = bootstrap_metric_intervals(
        labels,
        scores,
        threshold=policy.high_threshold,
        target_fpr=target_fpr,
        iterations=int(cfg.get("evaluation", {}).get("bootstrap_iterations", 0)),
        seed=int(cfg.get("evaluation", {}).get("bootstrap_seed", 42)),
        max_samples=cfg.get("evaluation", {}).get("bootstrap_sample_size"),
        n_jobs=cfg.get("evaluation", {}).get("bootstrap_n_jobs"),
    )
    fairness = _native_learner_fpr(frame, scores, high_threshold=policy.high_threshold)
    return {
        "model": model_name,
        "eval_set": eval_name,
        "split_mode": split_mode,
        "sampled_eval": bool(sampled),
        "selected_model": False,
        "n": int(len(frame)),
        **_prefix("default", default_metrics),
        **_prefix("high_confidence", high_metrics),
        **_prefix("selective", selective),
        **_prefix("high_confidence", intervals),
        **fairness,
    }


def _select_model(policies: dict[str, SelectivePolicy]) -> str:
    candidates = [
        policy
        for policy in policies.values()
        if policy_satisfies_constraints(policy)
    ] or list(policies.values())
    return max(
        candidates,
        key=lambda policy: (
            policy.validation_ai_recall_high_confidence
            if policy.validation_ai_recall_high_confidence is not None
            else -1.0
        ),
    ).model_name


def _run_topic_holdout(
    hc3: pd.DataFrame,
    model_cfg: dict[str, Any],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    holdout_cfg = copy.deepcopy(model_cfg)
    holdout_cfg["features"]["max_word_features"] = int(
        cfg.get("paper_study", {}).get("source_holdout_max_word_features", 8000)
    )
    holdout_cfg["features"]["max_char_features"] = int(
        cfg.get("paper_study", {}).get("source_holdout_max_char_features", 8000)
    )
    sample_size = int(cfg.get("paper_study", {}).get("source_holdout_sample_size", 30000))
    hc3 = _sample_by_columns(
        hc3,
        columns=["domain", "label"],
        max_samples=sample_size,
        seed=int(cfg.get("random_seed", 42)),
    )
    frames = leave_one_domain_out_frames(
        hc3,
        domain_col="domain",
        validation_size=float(cfg.get("splits", {}).get("validation_size", 0.15)),
        seed=int(cfg.get("random_seed", 42)),
    )
    for held_out, split_frame in frames.items():
        _progress(f"source holdout: {held_out}")
        bundle, validation_metrics = train_detector(split_frame, holdout_cfg)
        test = split_frame[split_frame["split"] == "test"].copy()
        scores = bundle.score_frame(test, batch_size=int(cfg.get("evaluation", {}).get("batch_size", 5000)))
        metrics = compute_binary_metrics(
            test["label"].astype(int),
            scores,
            threshold=bundle.threshold,
            target_fpr=float(cfg.get("model", {}).get("target_fpr", 0.01)),
        )
        rows.append(
            {
                "held_out_source": held_out,
                "model": "word_char_stats_lr",
                "n_train": int((split_frame["split"] == "train").sum()),
                "n_validation": int((split_frame["split"] == "validation").sum()),
                "n_test": int(len(test)),
                **_prefix("validation", validation_metrics),
                **_prefix("test", metrics),
            }
        )
    return rows


def _run_style_invariance(
    *,
    bundles: dict[str, DetectorBundle],
    policies: dict[str, SelectivePolicy],
    eval_frames: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    transforms = list(cfg.get("augmentation", {}).get("transforms", ["whitespace", "quotes", "punctuation"]))
    if "style_normalized" not in transforms:
        transforms.append("style_normalized")
    rows: list[dict[str, Any]] = []
    for eval_name, frame in eval_frames.items():
        sample = _sample_for_study(frame, max_samples=1000, seed=int(cfg.get("random_seed", 42)))
        for model_name, bundle in bundles.items():
            policy = policies[model_name]
            original_scores = bundle.score_frame(
                sample,
                batch_size=int(cfg.get("evaluation", {}).get("batch_size", 5000)),
            )
            for transform in transforms:
                changed = transformed_frame(sample, transform)
                changed_scores = bundle.score_frame(
                    changed,
                    batch_size=int(cfg.get("evaluation", {}).get("batch_size", 5000)),
                )
                shift = np.abs(changed_scores - original_scores)
                rows.append(
                    {
                        "model": model_name,
                        "eval_set": eval_name,
                        "transform": transform,
                        "n": int(len(sample)),
                        "mean_abs_score_shift": float(shift.mean()) if len(shift) else None,
                        "p95_abs_score_shift": float(np.percentile(shift, 95)) if len(shift) else None,
                        "default_decision_flip_rate": float(
                            (
                                (original_scores >= policy.low_threshold)
                                != (changed_scores >= policy.low_threshold)
                            ).mean()
                        )
                        if len(shift)
                        else None,
                        "high_confidence_decision_flip_rate": float(
                            (
                                (original_scores >= policy.high_threshold)
                                != (changed_scores >= policy.high_threshold)
                            ).mean()
                        )
                        if len(shift)
                        else None,
                    }
                )
    return rows


def _style_model_names(selected_model: str, bundles: dict[str, DetectorBundle]) -> list[str]:
    names = [selected_model]
    for candidate in ("no_character_lr", "char_capped_lr"):
        if candidate in bundles and candidate not in names:
            names.append(candidate)
    return names


def _sample_for_study(df: pd.DataFrame, *, max_samples: int, seed: int) -> pd.DataFrame:
    if not max_samples or len(df) <= max_samples:
        return df.copy().reset_index(drop=True)
    if "label" in df.columns and df["label"].nunique(dropna=False) > 1:
        return _balanced_sample_by_column(df, column="label", max_samples=max_samples, seed=seed)
    if (
        "native_status" in df.columns
        and df["native_status"].nunique(dropna=False) > 1
        and {"native", "learner"}.issubset(set(df["native_status"].astype(str)))
    ):
        return _balanced_sample_by_column(df, column="native_status", max_samples=max_samples, seed=seed)
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def _balanced_sample_by_column(
    df: pd.DataFrame,
    *,
    column: str,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    sampled_indices = []
    per_group = max(1, max_samples // df[column].nunique(dropna=False))
    for _, group in df.groupby(column, dropna=False):
        sampled_indices.extend(group.sample(min(len(group), per_group), random_state=seed).index.tolist())
    sampled = df.loc[sampled_indices]
    if len(sampled) < max_samples:
        remaining = df.drop(sampled.index, errors="ignore")
        if not remaining.empty:
            sampled = pd.concat(
                [sampled, remaining.sample(min(len(remaining), max_samples - len(sampled)), random_state=seed)]
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _sample_by_columns(
    df: pd.DataFrame,
    *,
    columns: list[str],
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    if not max_samples or len(df) <= max_samples:
        return df.copy().reset_index(drop=True)
    sampled_indices = []
    group_count = max(1, df.groupby(columns, dropna=False).ngroups)
    per_group = max(1, max_samples // group_count)
    for _, group in df.groupby(columns, dropna=False):
        sampled_indices.extend(group.sample(min(len(group), per_group), random_state=seed).index.tolist())
    sampled = df.loc[sampled_indices]
    if len(sampled) < max_samples:
        remaining = df.drop(sampled.index, errors="ignore")
        if not remaining.empty:
            sampled = pd.concat(
                [sampled, remaining.sample(min(len(remaining), max_samples - len(sampled)), random_state=seed)]
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _native_learner_fpr(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    high_threshold: float,
) -> dict[str, float | None]:
    if "native_status" not in frame.columns:
        return {}
    work = frame.copy()
    work["score_ai"] = scores
    work = work[work["label"].astype(int) == 0]
    rows = {}
    for status in ("native", "learner"):
        group = work[work["native_status"].astype(str) == status]
        rows[f"{status}_high_confidence_fpr"] = (
            float((group["score_ai"] >= high_threshold).mean()) if len(group) else None
        )
        rows[f"{status}_n_human"] = int(len(group))
    if rows.get("native_high_confidence_fpr") is not None and rows.get("learner_high_confidence_fpr") is not None:
        rows["fpr_gap_learner_minus_native"] = (
            rows["learner_high_confidence_fpr"] - rows["native_high_confidence_fpr"]
        )
    return rows


def _write_paper_study_figures(
    *,
    summary: pd.DataFrame,
    policies: pd.DataFrame,
    topic_holdout: pd.DataFrame,
    style: pd.DataFrame,
    cfg: dict[str, Any],
    selected_model: str,
) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    for base_dir_key, prefix in (("reports_dir", "reports"),):
        figures_dir = Path(cfg["paths"][base_dir_key]) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figures.update(_plot_all(figures_dir, summary, policies, topic_holdout, style, selected_model, prefix))
    paper_dir = Path(cfg.get("_project_root", Path.cwd())) / "paper" / "figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    figures.update(_plot_all(paper_dir, summary, policies, topic_holdout, style, selected_model, "paper"))
    return figures


def _plot_all(
    figures_dir: Path,
    summary: pd.DataFrame,
    policies: pd.DataFrame,
    topic_holdout: pd.DataFrame,
    style: pd.DataFrame,
    selected_model: str,
    prefix: str,
) -> dict[str, Path]:
    outputs = {
        f"{prefix}_model_tradeoff_figure": figures_dir / "paper_study_model_tradeoff.png",
        f"{prefix}_selective_policy_figure": figures_dir / "selective_policy_tradeoff.png",
        f"{prefix}_topic_holdout_figure": figures_dir / "topic_holdout_results.png",
        f"{prefix}_style_invariance_figure": figures_dir / "style_invariance_shift.png",
    }
    _plot_model_tradeoff(summary, outputs[f"{prefix}_model_tradeoff_figure"])
    _plot_selective_policy(summary, selected_model, outputs[f"{prefix}_selective_policy_figure"])
    _plot_topic_holdout(topic_holdout, outputs[f"{prefix}_topic_holdout_figure"])
    _plot_style_invariance(style, outputs[f"{prefix}_style_invariance_figure"])
    return outputs


def _plot_model_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    topic = summary[summary["eval_set"] == "topic_test"].copy()
    fairness = summary[summary["eval_set"] == "icnale_fairness_audit"].copy()
    if fairness.empty:
        fairness = summary[summary["eval_set"] == "gpt_wiki_intro_ood"].copy()
    merged = topic.merge(
        fairness[["model", "high_confidence_fpr"]],
        on="model",
        suffixes=("_topic", "_fairness"),
    )
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    if not merged.empty:
        ax.scatter(
            merged["high_confidence_fpr_fairness"],
            merged["high_confidence_tpr"],
            color="#4c78a8",
        )
        for _, row in merged.iterrows():
            ax.annotate(
                str(row["model"]).replace("_", "\n"),
                (row["high_confidence_fpr_fairness"], row["high_confidence_tpr"]),
                fontsize=6,
            )
    ax.set_xlabel("High-confidence human FPR")
    ax.set_ylabel("High-confidence AI recall")
    ax.set_title("Fairness-constrained model trade-off")
    ax.grid(alpha=0.25)
    _save(fig, path)


def _plot_selective_policy(summary: pd.DataFrame, selected_model: str, path: Path) -> None:
    rows = summary[(summary["model"] == selected_model) & (summary["eval_set"] != "topic_validation")].copy()
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    if not rows.empty:
        x = np.arange(len(rows))
        width = 0.35
        ax.bar(x - width / 2, rows["default_fpr"], width, label="Default FPR", color="#b23a48")
        ax.bar(x + width / 2, rows["high_confidence_fpr"], width, label="High-conf FPR", color="#2f6f9f")
        ax.set_xticks(x, rows["eval_set"].str.replace("_", "\n"))
        ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Three-way policy reduces accusations")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def _plot_topic_holdout(topic_holdout: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    if not topic_holdout.empty:
        rows = topic_holdout.sort_values("held_out_source")
        x = np.arange(len(rows))
        ax.bar(x, rows["test_f1_macro"], color="#54a24b")
        ax.set_xticks(x, rows["held_out_source"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 macro")
    ax.set_title("Leave-one-HC3-source-out robustness")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def _plot_style_invariance(style: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    if not style.empty:
        rows = style[
            (style["model"] == "word_char_stats_lr") & (style["transform"] == "style_normalized")
        ].copy()
        if rows.empty:
            rows = style.copy()
        eval_order = ["topic_test", "gpt_wiki_intro_ood", "icnale_fairness_audit"]
        label_map = {
            "topic_test": "Topic",
            "gpt_wiki_intro_ood": "GPT-wiki",
            "icnale_fairness_audit": "ICNALE",
        }
        rows["eval_order"] = pd.Categorical(rows["eval_set"], categories=eval_order, ordered=True)
        rows = rows.sort_values(["eval_order", "eval_set"])
        labels = rows["eval_set"].map(label_map).fillna(rows["eval_set"].str.replace("_", " "))
        ax.barh(labels, rows["mean_abs_score_shift"], color="#7f3c8d")
        max_shift = float(rows["mean_abs_score_shift"].max()) if not rows.empty else 0.0
        ax.set_xlim(0, max(0.02, max_shift * 1.25))
        ax.invert_yaxis()
    ax.set_xlabel("Mean absolute score shift")
    ax.set_title("Selected-model style sensitivity")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(pad=0.7)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_readiness_summary(
    cfg: dict[str, Any],
    *,
    selected_model: str,
    skip_daigt: bool,
    skip_icnale: bool,
    outputs: dict[str, Path],
) -> Path:
    payload = {
        "selected_model": selected_model,
        "skip_daigt": bool(skip_daigt),
        "skip_icnale": bool(skip_icnale),
        "headline_numbers_from_artifacts": True,
        "icnale_audit_used_for_selection": False,
        "sampled_transformer_baselines_must_be_labelled_sampled": True,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    path = Path(cfg["paths"]["reports_dir"]) / "results" / "paper_study_readiness.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
    return path


def _write_csv(cfg: dict[str, Any], filename: str, rows: list[dict[str, Any]]) -> Path:
    path = Path(cfg["paths"]["reports_dir"]) / "results" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _ensure_output_dirs(cfg: dict[str, Any]) -> None:
    Path(cfg["paths"]["reports_dir"], "results").mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["reports_dir"], "figures").mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["artifacts_dir"], "models").mkdir(parents=True, exist_ok=True)


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _progress(message: str) -> None:
    print(f"[paper-study] {message}", flush=True)
