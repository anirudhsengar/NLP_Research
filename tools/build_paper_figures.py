#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "paper" / "figures"

HUMAN = "#1f77b4"
AI = "#d62728"
DEFAULT = "#b23a48"
EDUCATION = "#2f6f9f"
NATIVE = "#4c78a8"
LEARNER = "#f58518"
NEUTRAL = "#58595b"


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_feature_family_weight()
    plot_hc3_confusion_matrix()
    plot_default_shift_rates()
    plot_score_distributions()
    plot_icnale_subgroup_policy()
    plot_education_tradeoff()
    plot_ablation_tradeoff()
    return 0


def plot_feature_family_weight() -> None:
    df = pd.read_csv(RESULTS / "logreg_feature_group_summary.csv")
    df = df.sort_values("coefficient_l1", ascending=True)
    total = df["coefficient_l1"].sum()
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    colors = ["#72b7b2", "#54a24b", "#e45756"]
    bars = ax.barh(clean_labels(df["feature_family"]), df["coefficient_l1"], color=colors[: len(df)])
    ax.set_xlabel("Sum of absolute coefficients")
    ax.set_title("Model reliance by feature family")
    ax.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, df["coefficient_l1"], strict=True):
        pct = 100 * value / total
        ax.text(value + total * 0.012, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center")
    save(fig, "feature_family_weight.png")


def plot_hc3_confusion_matrix() -> None:
    confusion = pd.read_csv(RESULTS / "hc3_test_confusion.csv", index_col=0)
    matrix = confusion.loc[["true_human", "true_ai"], ["pred_human", "pred_ai"]].to_numpy(dtype=int)
    row_totals = matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("HC3 test confusion matrix")
    ax.set_xticks([0, 1], ["Pred human", "Pred AI"])
    ax.set_yticks([0, 1], ["True human", "True AI"])
    max_value = matrix.max()
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            pct = value / max(row_totals[row, 0], 1)
            color = "white" if value > max_value * 0.55 else "#111111"
            ax.text(col, row, f"{value:,}\n{pct:.1%}", ha="center", va="center", color=color)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.text(
        0.5,
        -0.28,
        "Default HC3 threshold: FPR 1.28%, TPR 99.28%",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=6,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "hc3_confusion_matrix.png")


def plot_default_shift_rates() -> None:
    summary = pd.read_csv(RESULTS / "summary_table.csv")
    rows = summary.set_index("eval_set").loc[["hc3_test", "gpt_wiki_intro_ood", "icnale_fairness"]]
    labels = ["HC3 test", "GPT-wiki OOD", "ICNALE"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    fpr = rows["fpr"].to_numpy(dtype=float)
    tpr = rows["tpr"].to_numpy(dtype=float)
    fpr_bars = ax.bar(x - width / 2, fpr, width, label="Human FPR", color=DEFAULT)
    tpr_bars = ax.bar(x + width / 2, np.nan_to_num(tpr), width, label="AI recall", color=EDUCATION)
    annotate_bars(ax, fpr_bars, fpr)
    annotate_bars(ax, tpr_bars[:2], tpr[:2])
    ax.text(x[2] + width / 2, 0.03, "N/A", ha="center", va="bottom", color=NEUTRAL)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Rate")
    ax.set_title("Default threshold fails under shift")
    ax.set_xticks(x, labels)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.22)
    save(fig, "default_shift_rates.png")


def plot_score_distributions() -> None:
    default_threshold = float(pd.read_csv(RESULTS / "summary_table.csv")["threshold"].dropna().iloc[0])
    education_threshold = float(
        pd.read_csv(RESULTS / "calibrated_summary_table.csv")
        .query("threshold_policy == 'education_mitigated'")["threshold"]
        .dropna()
        .iloc[0]
    )
    bins = np.linspace(0, 1, 41)
    hc3 = pd.read_csv(RESULTS / "hc3_test_predictions.csv")
    gpt_wiki = sample_predictions("gpt_wiki_intro_ood_predictions.csv")
    icnale = pd.read_csv(RESULTS / "icnale_fairness_predictions.csv")
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    plot_hist_line(
        ax,
        hc3.loc[hc3["label"] == 0, "score_ai"].to_numpy(dtype=float),
        bins,
        "HC3 human",
        HUMAN,
    )
    plot_hist_line(
        ax,
        hc3.loc[hc3["label"] == 1, "score_ai"].to_numpy(dtype=float),
        bins,
        "HC3 AI",
        AI,
    )
    plot_hist_line(
        ax,
        gpt_wiki.loc[gpt_wiki["label"] == 0, "score_ai"].to_numpy(dtype=float),
        bins,
        "GPT-wiki human",
        LEARNER,
    )
    plot_hist_line(
        ax,
        icnale.loc[icnale["label"] == 0, "score_ai"].to_numpy(dtype=float),
        bins,
        "ICNALE human",
        "#7f3c8d",
    )
    ax.axvline(default_threshold, color="#111111", linewidth=1.0, linestyle="--")
    ax.axvline(education_threshold, color="#6f4e9b", linewidth=1.0, linestyle=":")
    ax.text(default_threshold + 0.012, 0.74, "default", rotation=90, va="top", fontsize=5.8)
    ax.text(education_threshold - 0.012, 0.74, "education", rotation=90, va="top", ha="right", fontsize=5.8)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Model AI score")
    ax.set_ylabel("Fraction per bin")
    ax.set_title("Score distributions under domain shift")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.22)
    save(fig, "score_distribution_shift.png")


def plot_icnale_subgroup_policy() -> None:
    summary = pd.read_csv(RESULTS / "calibrated_summary_table.csv")
    rows = summary[summary["eval_set"].str.startswith("icnale_fairness_audit")].copy()
    rows["policy"] = rows["threshold_policy"].map(
        {"default_hc3_validation": "Default", "education_mitigated": "Education"}
    )
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    native = rows["native_fpr"].to_numpy(dtype=float)
    learner = rows["learner_fpr"].to_numpy(dtype=float)
    native_bars = ax.bar(x - width / 2, native, width, label="Native", color=NATIVE)
    learner_bars = ax.bar(x + width / 2, learner, width, label="Learner", color=LEARNER)
    annotate_bars(ax, native_bars, native)
    annotate_bars(ax, learner_bars, learner)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("False-positive rate")
    ax.set_title("ICNALE subgroup false positives")
    ax.set_xticks(x, rows["policy"])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    save(fig, "icnale_subgroup_policy.png")


def plot_education_tradeoff() -> None:
    summary = pd.read_csv(RESULTS / "calibrated_summary_table.csv")
    summary["set"] = summary["eval_set"].map(
        {
            "hc3_test_default": "HC3",
            "hc3_test_education_mitigated": "HC3",
            "gpt_wiki_intro_ood_default": "GPT-wiki",
            "gpt_wiki_intro_ood_education_mitigated": "GPT-wiki",
            "icnale_fairness_audit_default": "ICNALE",
            "icnale_fairness_audit_education_mitigated": "ICNALE",
        }
    )
    summary["policy"] = summary["threshold_policy"].map(
        {"default_hc3_validation": "Default", "education_mitigated": "Education"}
    )
    sets = ["HC3", "GPT-wiki", "ICNALE"]
    x = np.arange(len(sets))
    width = 0.34
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.35), sharex=True)
    for ax, metric, title in [
        (axes[0], "fpr", "Human false positives"),
        (axes[1], "tpr", "AI recall"),
    ]:
        default_values = values_for(summary, sets, "Default", metric)
        education_values = values_for(summary, sets, "Education", metric)
        ax.bar(x - width / 2, np.nan_to_num(default_values), width, label="Default", color=DEFAULT)
        ax.bar(x + width / 2, np.nan_to_num(education_values), width, label="Education", color=EDUCATION)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=0.22)
        if metric == "tpr":
            ax.text(x[2], 0.04, "N/A", ha="center", va="bottom", color=NEUTRAL)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[1].set_xticks(x, sets)
    save(fig, "education_threshold_tradeoff.png")


def plot_ablation_tradeoff() -> None:
    df = pd.read_csv(RESULTS / "ablation_metrics.csv")
    rows = []
    for ablation in ["word_tfidf", "char_tfidf", "basic_stats", "word_char_tfidf", "word_char_stats"]:
        hc3 = df[(df["ablation"] == ablation) & (df["eval_set"] == "hc3_test")].iloc[0]
        icnale = df[(df["ablation"] == ablation) & (df["eval_set"] == "icnale_fairness")].iloc[0]
        rows.append(
            {
                "label": {
                    "word_tfidf": "word",
                    "char_tfidf": "char",
                    "basic_stats": "stats",
                    "word_char_tfidf": "word+char",
                    "word_char_stats": "full",
                }[ablation],
                "hc3_f1": float(hc3["metric_f1_macro"]),
                "icnale_fpr": float(icnale["metric_fpr"]),
            }
        )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.scatter(plot_df["hc3_f1"], plot_df["icnale_fpr"], s=42, color="#7f3c8d")
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["label"],
            (row["hc3_f1"], row["icnale_fpr"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6,
        )
    ax.set_xlabel("HC3 test F1-macro")
    ax.set_ylabel("ICNALE human FPR")
    ax.set_title("Accuracy and fairness do not move together")
    ax.set_xlim(0.74, 1.01)
    ax.set_ylim(0, 0.86)
    ax.grid(alpha=0.24)
    save(fig, "ablation_fairness_tradeoff.png")


def sample_predictions(filename: str, *, max_per_label: int = 25000) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / filename)
    sampled = []
    for _, group in df.groupby("label"):
        sampled.append(group.sample(min(len(group), max_per_label), random_state=42))
    return pd.concat(sampled, ignore_index=True)


def values_for(df: pd.DataFrame, sets: list[str], policy: str, metric: str) -> np.ndarray:
    values = []
    for name in sets:
        row = df[(df["set"] == name) & (df["policy"] == policy)]
        values.append(float(row[metric].iloc[0]) if not row.empty and pd.notna(row[metric].iloc[0]) else np.nan)
    return np.asarray(values, dtype=float)


def plot_hist_line(ax: plt.Axes, scores: np.ndarray, bins: np.ndarray, label: str, color: str) -> None:
    counts, edges = np.histogram(scores, bins=bins)
    fractions = counts / max(counts.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, fractions, drawstyle="steps-mid", linewidth=1.5, label=label, color=color)


def clean_labels(values: pd.Series) -> list[str]:
    return [str(value).replace("_", " ") for value in values]


def annotate_bars(ax: plt.Axes, bars, values: np.ndarray) -> None:
    for bar, value in zip(bars, values, strict=False):
        if pd.isna(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=5.8,
        )


def save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout(pad=0.6)
    fig.savefig(FIGURES / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
