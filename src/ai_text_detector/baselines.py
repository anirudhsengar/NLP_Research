from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_text_detector.evaluation import evaluate_scores, save_evaluation_summary
from ai_text_detector.features import LanguageModelStatsTransformer
from ai_text_detector.metrics import threshold_for_target_fpr
from ai_text_detector.schema import read_jsonl


def run_transformer_baselines(
    eval_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    *,
    max_samples: int | None = None,
    output_prefix: str = "baseline",
    summary_filename: str = "baseline_summary_table.csv",
    threshold_max_samples: int | None = None,
) -> dict[str, Path]:
    baseline_cfg = config.get("baselines", {})
    models = baseline_cfg.get("transformer_models", {})
    batch_size = int(baseline_cfg.get("batch_size", 8))
    if max_samples is None:
        max_samples = baseline_cfg.get("max_samples_per_eval")

    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    all_eval_outputs = {}
    gltr_output, gltr_eval_outputs = _run_gltr_heuristic_baseline(
        eval_frames,
        config,
        max_samples=max_samples,
        output_prefix=output_prefix,
        threshold_max_samples=threshold_max_samples,
    )
    outputs.update(gltr_output)
    all_eval_outputs.update(gltr_eval_outputs)
    for baseline_name, spec in models.items():
        model_name = spec["model_name"]
        predictor = TransformerBaseline(model_name=model_name, batch_size=batch_size)
        output, eval_outputs = _run_predictor_baseline(
            baseline_name,
            predictor,
            eval_frames,
            config,
            max_samples=max_samples,
            model_name=model_name,
            output_prefix=output_prefix,
            threshold_max_samples=threshold_max_samples,
        )
        outputs[baseline_name] = output
        all_eval_outputs.update(eval_outputs)
    if all_eval_outputs:
        outputs["baseline_summary_table"] = save_evaluation_summary(
            all_eval_outputs,
            config,
            filename=summary_filename,
        )
    return outputs


class TransformerBaseline:
    def __init__(self, *, model_name: str, batch_size: int = 8):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._ai_index = None

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        self._load()
        scores = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
                probs = self._torch.softmax(logits, dim=-1).cpu().numpy()
            scores.extend(probs[:, self._ai_index].tolist())
        return np.asarray(scores, dtype=float)

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install baseline dependencies first: uv sync --extra baselines"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        self._ai_index = _detect_ai_class_index(self._model.config.id2label)


class GltrHeuristicBaseline:
    def __init__(self, *, model_name: str = "gpt2", max_length: int = 256):
        self.transformer = LanguageModelStatsTransformer(model_name=model_name, max_length=max_length)

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        stats = self.transformer.transform(texts).toarray()
        top10 = stats[:, 3]
        top100 = stats[:, 4]
        top1000 = stats[:, 5]
        over1000 = stats[:, 6]
        scores = 0.45 * top10 + 0.35 * top100 + 0.20 * top1000 - 0.35 * over1000
        return np.clip(scores, 0.0, 1.0)


def _run_gltr_heuristic_baseline(
    eval_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    *,
    max_samples: int | None,
    output_prefix: str,
    threshold_max_samples: int | None,
) -> tuple[dict[str, Path], dict[str, dict[str, Path]]]:
    spec = config.get("baselines", {}).get("gltr_heuristic", {})
    if not spec.get("enabled", False):
        return {}, {}

    predictor = GltrHeuristicBaseline(
        model_name=spec.get("model_name", "gpt2"),
        max_length=int(spec.get("max_length", 256)),
    )
    output, eval_outputs = _run_predictor_baseline(
        "gltr_heuristic",
        predictor,
        eval_frames,
        config,
        max_samples=max_samples,
        model_name=spec.get("model_name", "gpt2"),
        output_prefix=output_prefix,
        threshold_max_samples=threshold_max_samples,
    )
    return {"gltr_heuristic": output}, eval_outputs


def _run_predictor_baseline(
    baseline_name: str,
    predictor,
    eval_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    *,
    max_samples: int | None,
    model_name: str,
    output_prefix: str,
    threshold_max_samples: int | None,
) -> tuple[Path, dict[str, dict[str, Path]]]:
    results_dir = Path(config["paths"]["reports_dir"]) / "results"
    threshold, threshold_policy = _baseline_threshold(
        predictor,
        config,
        max_samples=max_samples,
        threshold_max_samples=threshold_max_samples,
    )

    all_rows = []
    eval_outputs = {}
    for eval_name, df in eval_frames.items():
        work = _sample_eval_frame(df, max_samples=max_samples, seed=int(config.get("random_seed", 42)))
        scores = predictor.predict_scores(work["text"].tolist())
        output_name = f"{output_prefix}_{baseline_name}_{eval_name}"
        paths = evaluate_scores(
            work,
            scores,
            name=output_name,
            config=config,
            threshold=threshold,
            threshold_policy=threshold_policy,
        )
        eval_outputs[output_name] = paths
        with paths["metrics"].open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics.update({"baseline": baseline_name, "model_name": model_name, "eval_set": eval_name})
        all_rows.append(metrics)
    output = results_dir / f"{output_prefix}_gltr_heuristic_metrics.json"
    if baseline_name != "gltr_heuristic":
        output = results_dir / f"{output_prefix}_{baseline_name}_metrics.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, sort_keys=True, allow_nan=False)
    return output, eval_outputs


def _baseline_threshold(
    predictor,
    config: dict[str, Any],
    *,
    max_samples: int | None,
    threshold_max_samples: int | None,
) -> tuple[float, str]:
    baseline_cfg = config.get("baselines", {})
    if not baseline_cfg.get("calibrate_threshold_on_hc3_validation", True):
        return 0.5, "fixed_0.5"
    processed_dir = Path(config["paths"]["processed_dir"])
    hc3_path = processed_dir / "hc3.jsonl"
    if not hc3_path.exists():
        return 0.5, "fixed_0.5_missing_hc3_validation"
    hc3 = read_jsonl(hc3_path)
    validation = hc3[hc3["split"] == "validation"].copy()
    if validation.empty:
        return 0.5, "fixed_0.5_missing_hc3_validation"
    if threshold_max_samples is None:
        threshold_max_samples = max_samples
    if not threshold_max_samples:
        threshold_max_samples = baseline_cfg.get("threshold_calibration_max_samples")
    work = _sample_eval_frame(
        validation,
        max_samples=threshold_max_samples,
        seed=int(config.get("random_seed", 42)),
    )
    scores = predictor.predict_scores(work["text"].tolist())
    threshold = threshold_for_target_fpr(
        work["label"].astype(int),
        scores,
        target_fpr=float(config["model"].get("target_fpr", 0.01)),
    )
    return float(threshold), "hc3_validation_target_fpr"


def _detect_ai_class_index(id2label: dict[int, str]) -> int:
    normalized = {int(idx): str(label).lower() for idx, label in id2label.items()}
    for idx, label in normalized.items():
        if any(token in label for token in ("fake", "generated", "chatgpt", "ai")):
            return idx
    for idx, label in normalized.items():
        if any(token in label for token in ("real", "human")):
            continue
    return 1 if len(normalized) > 1 else 0


def _sample_eval_frame(df: pd.DataFrame, *, max_samples: int | None, seed: int) -> pd.DataFrame:
    if not max_samples or len(df) <= max_samples:
        return df
    sampled_indices = []
    per_label = max(1, max_samples // max(1, df["label"].nunique()))
    for _, group in df.groupby("label"):
        sampled_indices.extend(group.sample(min(len(group), per_label), random_state=seed).index.tolist())
    return df.loc[sampled_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
