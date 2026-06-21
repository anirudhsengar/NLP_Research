from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from ai_text_detector.baselines import run_transformer_baselines
from ai_text_detector.calibration import (
    calibrate_thresholds,
    save_calibration_report,
    split_icnale_calibration_audit,
)
from ai_text_detector.config import DEFAULT_CONFIG_PATH, ensure_dirs, load_config
from ai_text_detector.comparison import (
    create_comparison_frames,
    evaluate_classical_on_comparison,
    run_comparison_baselines,
    save_comparison_sample_ids,
    save_model_comparison_table,
    save_paper_readiness_summary,
)
from ai_text_detector.data import prepare_all
from ai_text_detector.demo import launch_demo
from ai_text_detector.evaluation import evaluate_bundle, save_evaluation_summary, save_feature_importance
from ai_text_detector.metrics import compute_binary_metrics, compute_human_only_metrics
from ai_text_detector.model import load_bundle, save_bundle, train_detector
from ai_text_detector.paper_study import run_paper_study
from ai_text_detector.reporting import save_data_profile, save_run_manifest
from ai_text_detector.schema import read_jsonl

app = typer.Typer(help="Run the AI-generated text detector research pipeline.")
console = Console()


@app.command()
def prepare(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    max_hc3_samples: int | None = typer.Option(None, help="Optional balanced HC3 sample cap."),
    max_gptwiki_samples: int | None = typer.Option(None, help="Optional balanced GPT-wiki sample cap."),
    include_optional_icnale: bool | None = typer.Option(
        None,
        help="Override config for WEP/WEUAE optional ICNALE modules.",
    ),
    skip_icnale: bool = typer.Option(False, help="Prepare only public HC3/GPT-wiki data."),
    include_daigt: bool = typer.Option(
        False,
        "--include-daigt",
        help="Also prepare the configured local DAIGT v2 CSV.",
    ),
):
    """Fetch/prepare HC3, GPT-wiki-intro, and local ICNALE data."""
    cfg = load_config(config)
    ensure_dirs(cfg)
    outputs = prepare_all(
        cfg,
        max_hc3_samples=max_hc3_samples,
        max_gptwiki_samples=max_gptwiki_samples,
        max_daigt_samples=max_hc3_samples,
        include_optional_icnale=include_optional_icnale,
        skip_icnale=skip_icnale,
        include_daigt=include_daigt,
    )
    for name, path in outputs.items():
        console.print(f"[green]prepared[/green] {name}: {path}")
    profile_path = _save_profile_from_output_paths(outputs, cfg)
    manifest_path = save_run_manifest(
        cfg,
        command="aidetect prepare",
        outputs={**outputs, "data_profile": profile_path},
    )
    console.print(f"[green]saved data profile[/green] {profile_path}")
    console.print(f"[green]saved run manifest[/green] {manifest_path}")


@app.command()
def train(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    model_path: Path | None = typer.Option(None, help="Where to save the trained model bundle."),
    use_spacy: bool | None = typer.Option(None, help="Override config spaCy feature setting."),
):
    """Train the interpretable logistic-regression detector on prepared HC3."""
    cfg = load_config(config)
    ensure_dirs(cfg)
    hc3 = read_jsonl(Path(cfg["paths"]["processed_dir"]) / "hc3.jsonl")
    bundle, metrics = train_detector(hc3, cfg, use_spacy=use_spacy)
    if model_path is None:
        model_path = Path(cfg["paths"]["artifacts_dir"]) / "models" / "classical_logreg.joblib"
    saved = save_bundle(bundle, model_path)
    save_feature_importance(bundle, cfg)
    manifest_path = save_run_manifest(
        cfg,
        command="aidetect train",
        outputs={"model": saved, "feature_importance": "reports/results/logreg_feature_coefficients.csv"},
        model_path=saved,
        model_metadata=bundle.metadata,
    )
    console.print(f"[green]saved model[/green] {saved}")
    console.print(f"[green]saved run manifest[/green] {manifest_path}")
    console.print(metrics)


@app.command(name="eval")
def evaluate(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    model_path: Path | None = typer.Option(None, help="Saved model bundle path."),
):
    """Evaluate the trained model on HC3 test, GPT-wiki OOD, and ICNALE fairness."""
    cfg = load_config(config)
    ensure_dirs(cfg)
    if model_path is None:
        model_path = Path(cfg["paths"]["artifacts_dir"]) / "models" / "classical_logreg.joblib"
    bundle = load_bundle(model_path)
    outputs = _evaluate_prepared(bundle, cfg)
    manifest_path = save_run_manifest(
        cfg,
        command="aidetect eval",
        outputs=outputs,
        model_path=model_path,
        model_metadata=bundle.metadata,
    )
    for name, paths in outputs.items():
        console.print(f"[green]evaluated[/green] {name}")
        for output_name, path in paths.items():
            console.print(f"  {output_name}: {path}")
    console.print(f"[green]saved run manifest[/green] {manifest_path}")


@app.command()
def baselines(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    max_samples: int | None = typer.Option(None, help="Max samples per eval set for CPU inference."),
):
    """Run optional transformer detector baselines on prepared evaluation data."""
    cfg = load_config(config)
    ensure_dirs(cfg)
    frames = _prepared_eval_frames(cfg)
    outputs = run_transformer_baselines(frames, cfg, max_samples=max_samples)
    manifest_path = save_run_manifest(
        cfg,
        command="aidetect baselines",
        outputs=outputs,
    )
    for name, path in outputs.items():
        console.print(f"[green]baseline[/green] {name}: {path}")
    console.print(f"[green]saved run manifest[/green] {manifest_path}")


@app.command()
def compare(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    model_path: Path | None = typer.Option(None, help="Saved classical model bundle path."),
    max_samples_per_set: int = typer.Option(
        2000,
        help="Fixed sampled rows per eval set; use 0 for full data.",
    ),
    run_baselines_flag: bool = typer.Option(
        False,
        "--run-baselines",
        help="Run GLTR/RoBERTa baselines on the same fixed sample.",
    ),
):
    """Create fixed-sample paper model-comparison artifacts."""
    cfg = load_config(config)
    ensure_dirs(cfg)
    if model_path is None:
        model_path = Path(cfg["paths"]["artifacts_dir"]) / "models" / "classical_logreg.joblib"
    bundle = load_bundle(model_path)
    sample_cap = None if max_samples_per_set == 0 else max_samples_per_set
    frames = create_comparison_frames(
        _prepared_eval_frames(cfg),
        max_samples_per_set=sample_cap,
        seed=int(cfg.get("random_seed", 42)),
    )
    sample_ids = save_comparison_sample_ids(frames, cfg)
    classical_outputs = evaluate_classical_on_comparison(bundle, frames, cfg)
    baseline_outputs = {}
    baseline_summary = None
    if run_baselines_flag:
        baseline_outputs = run_comparison_baselines(frames, cfg)
        baseline_summary = baseline_outputs.get("baseline_summary_table")
    comparison_table = save_model_comparison_table(classical_outputs, baseline_summary, cfg)
    readiness = save_paper_readiness_summary(
        cfg,
        comparison_table=comparison_table,
        sample_ids=sample_ids,
        baseline_outputs=baseline_outputs,
    )
    manifest_path = save_run_manifest(
        cfg,
        command="aidetect compare",
        outputs={
            "comparison_sample_ids": sample_ids,
            "model_comparison_table": comparison_table,
            "paper_readiness_summary": readiness,
            **classical_outputs,
            **baseline_outputs,
        },
        model_path=model_path,
        model_metadata=bundle.metadata,
    )
    console.print(f"[green]saved comparison sample ids[/green] {sample_ids}")
    console.print(f"[green]saved model comparison table[/green] {comparison_table}")
    console.print(f"[green]saved paper-readiness summary[/green] {readiness}")
    console.print(f"[green]saved run manifest[/green] {manifest_path}")


@app.command()
def calibrate(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    model_path: Path | None = typer.Option(None, help="Saved model bundle path."),
    calibration_fraction: float | None = typer.Option(
        None,
        help="Fraction of ICNALE participant groups used only for threshold calibration.",
    ),
):
    """Calibrate conservative threshold policies and evaluate mitigation on held-out data."""
    cfg = load_config(config)
    ensure_dirs(cfg)
    if model_path is None:
        model_path = Path(cfg["paths"]["artifacts_dir"]) / "models" / "classical_logreg.joblib"
    bundle = load_bundle(model_path)
    hc3 = read_jsonl(Path(cfg["paths"]["processed_dir"]) / "hc3.jsonl")
    icnale = read_jsonl(Path(cfg["paths"]["processed_dir"]) / "icnale_fairness.jsonl")
    fraction = (
        float(calibration_fraction)
        if calibration_fraction is not None
        else float(cfg.get("calibration", {}).get("icnale_calibration_fraction", 0.3))
    )
    icnale_calibration, icnale_audit = split_icnale_calibration_audit(
        icnale,
        calibration_fraction=fraction,
        seed=int(cfg.get("random_seed", 42)),
    )
    policy, report = calibrate_thresholds(
        bundle,
        hc3=hc3,
        icnale_calibration=icnale_calibration,
        config=cfg,
    )
    calibration_report, calibration_policy = save_calibration_report(
        bundle,
        policy=policy,
        report=report,
        config=cfg,
    )
    save_bundle(bundle, model_path)

    education_threshold = policy["thresholds"]["education_mitigated"]
    eval_frames = {
        "hc3_test": hc3[hc3["split"] == "test"].copy(),
        "gpt_wiki_intro_ood": read_jsonl(Path(cfg["paths"]["processed_dir"]) / "gpt_wiki_intro.jsonl"),
        "icnale_fairness_audit": icnale_audit,
    }
    outputs = {}
    for eval_name, frame in eval_frames.items():
        if frame.empty:
            continue
        outputs[f"{eval_name}_default"] = evaluate_bundle(
            bundle,
            frame,
            name=f"{eval_name}_default",
            config=cfg,
            threshold=bundle.threshold,
            threshold_policy="default_hc3_validation",
        )
        outputs[f"{eval_name}_education_mitigated"] = evaluate_bundle(
            bundle,
            frame,
            name=f"{eval_name}_education_mitigated",
            config=cfg,
            threshold=education_threshold,
            threshold_policy="education_mitigated",
        )
    summary = save_evaluation_summary(outputs, cfg, filename="calibrated_summary_table.csv")
    manifest_path = save_run_manifest(
        cfg,
        command="aidetect calibrate",
        outputs={**outputs, "calibration_report": calibration_report, "calibration_policy": calibration_policy},
        model_path=model_path,
        model_metadata=bundle.metadata,
    )
    console.print(f"[green]saved calibration report[/green] {calibration_report}")
    console.print(f"[green]saved calibration policy[/green] {calibration_policy}")
    console.print(f"[green]updated model[/green] {model_path}")
    console.print(f"[green]saved calibrated evaluation summary[/green] {summary}")
    console.print(f"[green]saved run manifest[/green] {manifest_path}")


@app.command()
def ablations(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    max_eval_samples: int | None = typer.Option(
        5000,
        help="Balanced eval sample cap per dataset; use 0 for full evaluation.",
    ),
    include_spacy: bool = typer.Option(False, help="Include a spaCy feature ablation."),
    include_lm_stats: bool = typer.Option(False, help="Include GPT-2/GLTR-style feature ablations."),
    skip_icnale: bool = typer.Option(False, help="Skip ICNALE fairness evaluation."),
):
    """Run feature-family ablations for the classical detector."""
    cfg = load_config(config)
    ensure_dirs(cfg)
    hc3 = read_jsonl(Path(cfg["paths"]["processed_dir"]) / "hc3.jsonl")
    eval_frames = _prepared_eval_frames(cfg, skip_icnale=skip_icnale)
    sample_cap = None if max_eval_samples == 0 else max_eval_samples
    rows = []

    for ablation_name, feature_config in _ablation_feature_configs(
        cfg["features"],
        include_spacy=include_spacy,
        include_lm_stats=include_lm_stats,
    ).items():
        work_cfg = copy.deepcopy(cfg)
        work_cfg["features"] = feature_config
        bundle, validation_metrics = train_detector(hc3, work_cfg)
        rows.append(
            {
                "ablation": ablation_name,
                "eval_set": "hc3_validation",
                **_prefixed_metrics(validation_metrics),
            }
        )
        for eval_name, frame in eval_frames.items():
            work = _sample_eval_frame(frame, max_samples=sample_cap, seed=int(cfg.get("random_seed", 42)))
            scores = bundle.score_frame(work, batch_size=int(cfg["evaluation"].get("batch_size", 5000)))
            labels = work["label"].astype(int)
            if set(labels.unique()) == {0}:
                metrics = compute_human_only_metrics(scores, threshold=bundle.threshold)
            else:
                metrics = compute_binary_metrics(labels, scores, threshold=bundle.threshold)
            rows.append(
                {
                    "ablation": ablation_name,
                    "eval_set": eval_name,
                    **_prefixed_metrics(metrics),
                }
            )
        console.print(f"[green]ablation complete[/green] {ablation_name}")

    output = Path(cfg["paths"]["reports_dir"]) / "results" / "ablation_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    manifest_path = save_run_manifest(
        cfg,
        command="aidetect ablations",
        outputs={"ablation_metrics": output},
    )
    console.print(f"[green]saved ablations[/green] {output}")
    console.print(f"[green]saved run manifest[/green] {manifest_path}")


@app.command()
def reproduce(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    sample_size: int | None = typer.Option(2000, help="CPU-safe balanced sample size; use 0 for full data."),
    run_baselines_flag: bool = typer.Option(False, "--run-baselines", help="Also run transformer baselines."),
    skip_icnale: bool = typer.Option(False, help="Skip ICNALE fairness data preparation/evaluation."),
):
    """Run a CPU-safe end-to-end experiment."""
    cfg = load_config(config)
    ensure_dirs(cfg)
    max_samples = None if sample_size == 0 else sample_size
    prepare_all(cfg, max_hc3_samples=max_samples, max_gptwiki_samples=max_samples, skip_icnale=skip_icnale)
    profile_path = save_data_profile(_prepared_all_frames(cfg, skip_icnale=skip_icnale), cfg)
    hc3 = read_jsonl(Path(cfg["paths"]["processed_dir"]) / "hc3.jsonl")
    bundle, metrics = train_detector(hc3, cfg)
    model_path = Path(cfg["paths"]["artifacts_dir"]) / "models" / "classical_logreg.joblib"
    save_bundle(bundle, model_path)
    save_feature_importance(bundle, cfg)
    outputs = _evaluate_prepared(bundle, cfg, skip_icnale=skip_icnale)
    manifest_path = save_run_manifest(
        cfg,
        command="aidetect reproduce",
        outputs={**outputs, "data_profile": profile_path},
        model_path=model_path,
        model_metadata=bundle.metadata,
    )
    console.print("[green]validation metrics[/green]")
    console.print(metrics)
    for name, paths in outputs.items():
        console.print(f"[green]evaluated[/green] {name}: {paths['metrics']}")
    console.print(f"[green]saved data profile[/green] {profile_path}")
    console.print(f"[green]saved run manifest[/green] {manifest_path}")
    if run_baselines_flag:
        baseline_outputs = run_transformer_baselines(_prepared_eval_frames(cfg), cfg, max_samples=max_samples)
        for name, path in baseline_outputs.items():
            console.print(f"[green]baseline[/green] {name}: {path}")


@app.command(name="paper-study")
def paper_study(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    sample_size: int | None = typer.Option(
        2000,
        help="In-memory balanced sample size per dataset; use 0 for full prepared data.",
    ),
    skip_daigt: bool = typer.Option(
        False,
        "--skip-daigt",
        help="Skip DAIGT v2; otherwise the configured local CSV is required.",
    ),
    skip_icnale: bool = typer.Option(False, "--skip-icnale", help="Skip ICNALE fairness audit."),
    bootstrap_iterations: int | None = typer.Option(
        None,
        "--bootstrap-iterations",
        help="Override evaluation.bootstrap_iterations for this run.",
    ),
    run_baselines_flag: bool = typer.Option(
        False,
        "--run-baselines",
        help="Run configured GLTR/RoBERTa baselines on the paper-study eval frames.",
    ),
):
    """Run the revised-paper HC3/DAIGT reproduction and selective-policy extension."""
    cfg = load_config(config)
    ensure_dirs(cfg)
    outputs = run_paper_study(
        cfg,
        sample_size=sample_size,
        skip_daigt=skip_daigt,
        skip_icnale=skip_icnale,
        bootstrap_iterations=bootstrap_iterations,
        run_baselines_flag=run_baselines_flag,
    )
    for name, path in outputs.items():
        console.print(f"[green]paper-study[/green] {name}: {path}")


@app.command()
def demo(
    model_path: Path = typer.Option(
        Path("artifacts/models/paper_study_selected.joblib"),
        help="Saved model bundle.",
    ),
    server_name: str = typer.Option("127.0.0.1"),
    server_port: int = typer.Option(7860),
):
    """Launch the Gradio demo."""
    launch_demo(model_path, server_name=server_name, server_port=server_port)


@app.command()
def profile(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    skip_icnale: bool = typer.Option(False, help="Skip ICNALE fairness data in the profile."),
):
    """Save a dataset provenance/count profile from prepared JSONL files."""
    cfg = load_config(config)
    ensure_dirs(cfg)
    output = save_data_profile(_prepared_all_frames(cfg, skip_icnale=skip_icnale), cfg)
    manifest_path = save_run_manifest(
        cfg,
        command="aidetect profile",
        outputs={"data_profile": output},
    )
    console.print(f"[green]saved data profile[/green] {output}")
    console.print(f"[green]saved run manifest[/green] {manifest_path}")


def _evaluate_prepared(bundle, cfg: dict, *, skip_icnale: bool = False) -> dict[str, dict[str, Path]]:
    frames = _prepared_eval_frames(cfg, skip_icnale=skip_icnale)
    outputs = {}
    for name, frame in frames.items():
        outputs[name] = evaluate_bundle(bundle, frame, name=name, config=cfg)
    save_evaluation_summary(outputs, cfg)
    return outputs


def _prepared_eval_frames(cfg: dict, *, skip_icnale: bool = False) -> dict[str, pd.DataFrame]:
    processed = Path(cfg["paths"]["processed_dir"])
    hc3 = read_jsonl(processed / "hc3.jsonl")
    frames = {
        "hc3_test": hc3[hc3["split"] == "test"].copy(),
        "gpt_wiki_intro_ood": read_jsonl(processed / "gpt_wiki_intro.jsonl"),
    }
    icnale_path = processed / "icnale_fairness.jsonl"
    if not skip_icnale and icnale_path.exists():
        frames["icnale_fairness"] = read_jsonl(icnale_path)
    return {name: frame for name, frame in frames.items() if not frame.empty}


def _prepared_all_frames(cfg: dict, *, skip_icnale: bool = False) -> dict[str, pd.DataFrame]:
    processed = Path(cfg["paths"]["processed_dir"])
    frames = {
        "hc3": read_jsonl(processed / "hc3.jsonl"),
        "gpt_wiki_intro": read_jsonl(processed / "gpt_wiki_intro.jsonl"),
    }
    icnale_path = processed / "icnale_fairness.jsonl"
    if not skip_icnale and icnale_path.exists():
        frames["icnale_fairness"] = read_jsonl(icnale_path)
    return frames


def _save_profile_from_output_paths(outputs: dict[str, Path], cfg: dict) -> Path:
    frames = {name: read_jsonl(path) for name, path in outputs.items()}
    return save_data_profile(frames, cfg)


def _ablation_feature_configs(
    base_features: dict,
    *,
    include_spacy: bool = False,
    include_lm_stats: bool = False,
) -> dict[str, dict]:
    def config_for(
        *,
        word: bool,
        char: bool,
        stats: bool,
        spacy: bool = False,
        lm_stats: bool = False,
    ) -> dict:
        feature_config = copy.deepcopy(base_features)
        feature_config.update(
            {
                "use_word_tfidf": word,
                "use_char_tfidf": char,
                "use_basic_stats": stats,
                "use_spacy": spacy,
                "use_lm_stats": lm_stats,
            }
        )
        return feature_config

    specs = {
        "word_tfidf": config_for(word=True, char=False, stats=False),
        "char_tfidf": config_for(word=False, char=True, stats=False),
        "basic_stats": config_for(word=False, char=False, stats=True),
        "word_char_tfidf": config_for(word=True, char=True, stats=False),
        "word_char_stats": config_for(word=True, char=True, stats=True),
    }
    if include_spacy:
        specs["word_char_stats_spacy"] = config_for(word=True, char=True, stats=True, spacy=True)
    if include_lm_stats:
        specs["lm_stats"] = config_for(word=False, char=False, stats=False, lm_stats=True)
        specs["word_char_stats_lm"] = config_for(word=True, char=True, stats=True, lm_stats=True)
    return specs


def _sample_eval_frame(df: pd.DataFrame, *, max_samples: int | None, seed: int) -> pd.DataFrame:
    if not max_samples or len(df) <= max_samples:
        return df
    sampled_indices = []
    per_label = max(1, max_samples // max(1, df["label"].nunique()))
    for _, group in df.groupby("label"):
        sampled_indices.extend(group.sample(min(len(group), per_label), random_state=seed).index.tolist())
    sampled = df.loc[sampled_indices]
    if len(sampled) < max_samples:
        remaining = df.drop(sampled.index, errors="ignore")
        if not remaining.empty:
            sampled = pd.concat(
                [sampled, remaining.sample(min(len(remaining), max_samples - len(sampled)), random_state=seed)]
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _prefixed_metrics(metrics: dict) -> dict:
    return {f"metric_{key}": value for key, value in metrics.items()}


if __name__ == "__main__":
    app()
