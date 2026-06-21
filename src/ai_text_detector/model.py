from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ai_text_detector.features import feature_names, make_feature_pipeline
from ai_text_detector.metrics import compute_binary_metrics, threshold_for_target_fpr


@dataclass
class DetectorBundle:
    pipeline: Pipeline
    threshold: float
    metadata: dict[str, Any]

    def score_texts(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        df = pd.DataFrame({"text": texts})
        return self.score_frame(df, batch_size=batch_size)

    def score_frame(self, df: pd.DataFrame, *, batch_size: int | None = None) -> np.ndarray:
        n_jobs = _feature_union_n_jobs(self.pipeline)
        if not batch_size or len(df) <= batch_size:
            with parallel_backend("threading", n_jobs=n_jobs):
                return self.pipeline.predict_proba(df)[:, 1]
        scores = []
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            with parallel_backend("threading", n_jobs=n_jobs):
                scores.append(self.pipeline.predict_proba(batch)[:, 1])
        return np.concatenate(scores)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        scores = self.score_frame(df)
        output = df.copy()
        output["score_ai"] = scores
        output["pred_label"] = (scores >= self.threshold).astype(int)
        return output

    def explain_text(self, text: str, *, top_k: int = 10) -> dict[str, Any]:
        df = pd.DataFrame({"text": [text]})
        score = float(self.score_frame(df)[0])
        names, contributions = self._feature_contributions(df)
        order_pos = np.argsort(contributions)[::-1][:top_k]
        order_neg = np.argsort(contributions)[:top_k]
        low_threshold = self.low_threshold
        high_threshold = self.high_confidence_threshold
        selective_prediction = self.selective_prediction(score)
        return {
            "score_ai": score,
            "threshold": self.threshold,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "education_threshold": high_threshold,
            "prediction": "ai_generated" if score >= self.threshold else "human_written",
            "selective_prediction": selective_prediction,
            "selective_decision": _display_decision(selective_prediction),
            "mitigated_prediction": selective_prediction,
            "policy_source": self.policy_source,
            "policy": self.selective_policy or {},
            "review_zone": {
                "low": low_threshold,
                "high": high_threshold,
                "active": high_threshold > low_threshold,
                "contains_score": high_threshold > low_threshold and low_threshold <= score < high_threshold,
            },
            "top_ai_features": [
                {"feature": str(names[idx]), "contribution": float(contributions[idx])}
                for idx in order_pos
                if contributions[idx] > 0
            ],
            "top_human_features": [
                {"feature": str(names[idx]), "contribution": float(contributions[idx])}
                for idx in order_neg
                if contributions[idx] < 0
            ],
            "feature_group_contributions": _group_feature_contributions(names, contributions, top_k=top_k),
        }

    def _feature_contributions(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        features_step = self.pipeline.named_steps["features"]
        classifier = self.pipeline.named_steps["classifier"]
        x = features_step.transform(df)
        coefs = classifier.coef_[0]
        if sparse.issparse(x):
            contributions = x.multiply(coefs).toarray()[0]
        else:
            contributions = np.asarray(x)[0] * coefs
        return feature_names(features_step), contributions

    @property
    def education_threshold(self) -> float:
        calibration = self.metadata.get("calibration", {})
        thresholds = calibration.get("thresholds", {})
        return float(thresholds.get("education_mitigated", self.threshold))

    @property
    def selective_policy(self) -> dict[str, Any] | None:
        policy = self.metadata.get("selective_policy")
        if not isinstance(policy, dict):
            return None
        if "low_threshold" not in policy or "high_threshold" not in policy:
            return None
        return policy

    @property
    def low_threshold(self) -> float:
        policy = self.selective_policy
        if policy is not None:
            return float(policy["low_threshold"])
        return float(self.threshold)

    @property
    def high_confidence_threshold(self) -> float:
        policy = self.selective_policy
        if policy is not None:
            return max(float(policy["high_threshold"]), self.low_threshold)
        return max(float(self.education_threshold), self.low_threshold)

    @property
    def policy_source(self) -> str:
        if self.selective_policy is not None:
            return "paper_study_selective_policy"
        if self.education_threshold > self.threshold:
            return "legacy_education_calibration"
        return "default_binary_threshold"

    def selective_prediction(self, score: float) -> str:
        low_threshold = self.low_threshold
        high_threshold = self.high_confidence_threshold
        if high_threshold > low_threshold and low_threshold <= score < high_threshold:
            return "manual_review"
        if score >= high_threshold:
            return "high_confidence_ai"
        return "human_written"


def train_detector(
    hc3: pd.DataFrame,
    config: dict[str, Any],
    *,
    use_spacy: bool | None = None,
) -> tuple[DetectorBundle, dict[str, float]]:
    train_df = hc3[hc3["split"] == "train"].copy()
    validation_df = hc3[hc3["split"] == "validation"].copy()
    if train_df.empty or validation_df.empty:
        raise ValueError("HC3 train and validation splits are required to train the detector.")

    feature_config = dict(config["features"])
    if use_spacy is not None:
        feature_config["use_spacy"] = use_spacy

    model_cfg = config["model"]
    penalty = model_cfg.get("penalty", "l2")
    solver = _solver_for_penalty(str(penalty))
    classifier_kwargs: dict[str, Any] = {
        "class_weight": model_cfg.get("class_weight", "balanced"),
        "max_iter": int(model_cfg.get("max_iter", 1000)),
        "solver": solver,
        "random_state": int(config.get("random_seed", 42)),
    }
    if penalty in {"elasticnet", "l1"}:
        classifier_kwargs["penalty"] = penalty
    if penalty == "elasticnet":
        classifier_kwargs["l1_ratio"] = float(model_cfg.get("l1_ratio", 0.5))
    classifier = LogisticRegression(**classifier_kwargs)
    pipeline = Pipeline(
        [
            ("features", make_feature_pipeline(feature_config)),
            ("classifier", classifier),
        ]
    )
    n_jobs = feature_config.get("n_jobs")
    with parallel_backend("threading", n_jobs=n_jobs):
        pipeline.fit(train_df, train_df["label"].astype(int))
        validation_scores = pipeline.predict_proba(validation_df)[:, 1]
    threshold = threshold_for_target_fpr(
        validation_df["label"].astype(int),
        validation_scores,
        target_fpr=float(config["model"].get("target_fpr", 0.01)),
    )
    metrics = compute_binary_metrics(
        validation_df["label"].astype(int),
        validation_scores,
        threshold=threshold,
        target_fpr=float(config["model"].get("target_fpr", 0.01)),
    )
    bundle = DetectorBundle(
        pipeline=pipeline,
        threshold=threshold,
        metadata={
            "model_name": config["model"].get("name", "classical_logreg"),
            "model_variant": config["model"].get("variant", "word_char_stats_lr"),
            "penalty": penalty,
            "target_fpr": float(config["model"].get("target_fpr", 0.01)),
            "feature_config": feature_config,
            "validation_metrics": metrics,
        },
    )
    return bundle, metrics


def save_bundle(bundle: DetectorBundle, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return output


def load_bundle(path: str | Path) -> DetectorBundle:
    return joblib.load(path)


def _solver_for_penalty(penalty: str) -> str:
    if penalty == "elasticnet":
        return "saga"
    if penalty == "l1":
        return "liblinear"
    return "liblinear"


def _feature_union_n_jobs(pipeline: Pipeline) -> int | None:
    try:
        return pipeline.named_steps["features"].n_jobs
    except Exception:
        return None


def _group_feature_contributions(
    names: np.ndarray,
    contributions: np.ndarray,
    *,
    top_k: int,
) -> list[dict[str, float | int | str]]:
    groups: dict[str, dict[str, float | int | str]] = {}
    for name, contribution in zip(names, contributions, strict=False):
        value = float(contribution)
        if value == 0.0:
            continue
        family = _feature_family(str(name))
        row = groups.setdefault(
            family,
            {
                "feature_family": family,
                "signed_contribution": 0.0,
                "absolute_contribution": 0.0,
                "positive_contribution": 0.0,
                "negative_contribution": 0.0,
                "n_active_features": 0,
            },
        )
        row["signed_contribution"] = float(row["signed_contribution"]) + value
        row["absolute_contribution"] = float(row["absolute_contribution"]) + abs(value)
        row["positive_contribution"] = float(row["positive_contribution"]) + max(value, 0.0)
        row["negative_contribution"] = float(row["negative_contribution"]) + min(value, 0.0)
        row["n_active_features"] = int(row["n_active_features"]) + 1
    return sorted(
        groups.values(),
        key=lambda item: float(item["absolute_contribution"]),
        reverse=True,
    )[:top_k]


def _feature_family(feature: str) -> str:
    if "__" in feature:
        family = feature.split("__", 1)[0]
    elif ":" in feature:
        family = feature.split(":", 1)[0]
    else:
        family = "other"
    labels = {
        "word_tfidf": "Word TF-IDF",
        "char_tfidf": "Character TF-IDF",
        "basic_stats": "Lexical/readability stats",
        "spacy_stats": "POS/NER/syntax stats",
        "lm_stats": "GPT-2/GLTR stats",
    }
    return labels.get(family, family)


def _display_decision(prediction: str) -> str:
    labels = {
        "human_written": "Human",
        "manual_review": "Manual review",
        "high_confidence_ai": "High-confidence AI",
    }
    return labels.get(prediction, prediction)
