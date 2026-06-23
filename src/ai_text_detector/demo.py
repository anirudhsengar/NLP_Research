from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai_text_detector.model import load_bundle


EXAMPLE_TEXTS = [
    [
        "Human-style reflection",
        (
            "When I moved into my first dorm room, I thought the hardest part would be sharing a "
            "small space with someone I had never met. It was actually the silence after my parents "
            "left that surprised me. I remember sitting on the floor because my sheets were still in "
            "a suitcase, eating crackers for dinner, and realizing that independence felt less like "
            "freedom and more like a set of tiny decisions nobody else would make for me."
        ),
    ],
    [
        "Generic explanatory answer",
        (
            "Artificial intelligence has become an important topic in modern education because it "
            "offers both opportunities and challenges. On one hand, AI tools can help students learn "
            "more efficiently, generate ideas, and receive personalized support. On the other hand, "
            "educators must consider academic integrity, fairness, and the responsible use of these "
            "technologies in classroom environments."
        ),
    ],
]

INITIAL_RESULTS_HTML = """
<section class="empty-state">
  <p class="eyebrow">Research demo</p>
  <h2>Paste a passage to see the detector's decision.</h2>
  <p>
    This demo estimates whether essay-style writing resembles AI-generated or
    human-written text, then shows the strongest signals behind that estimate.
  </p>
</section>
"""

CSS = """
html,
body,
.gradio-container {
  background: #f7f8fb !important;
  color: #172033 !important;
  color-scheme: light !important;
}

.gradio-container {
  max-width: 1180px !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container textarea,
.gradio-container input,
.gradio-container table {
  background: #ffffff !important;
  color: #172033 !important;
}

.gradio-container label,
.gradio-container th,
.gradio-container td,
.gradio-container .prose {
  color: #172033 !important;
}

.demo-header {
  border-bottom: 1px solid #d9dee8;
  margin-bottom: 18px;
  padding: 10px 0 18px;
}

.demo-header h1 {
  color: #172033;
  font-size: 2rem;
  line-height: 1.1;
  margin: 4px 0 10px;
}

.demo-header p {
  color: #4f5f76;
  font-size: 1rem;
  line-height: 1.55;
  margin: 0;
  max-width: 880px;
}

.eyebrow {
  color: #1d6f78;
  font-size: 0.78rem;
  font-weight: 700;
  margin: 0 0 6px;
  text-transform: uppercase;
}

.result-shell,
.empty-state {
  background: #ffffff;
  border: 1px solid #d9dee8;
  border-radius: 8px;
  padding: 18px;
}

.empty-state h2 {
  color: #172033;
  font-size: 1.2rem;
  margin: 0 0 8px;
}

.empty-state p {
  color: #59667a;
  line-height: 1.5;
  margin: 0;
}

.status-row {
  align-items: flex-start;
  display: flex;
  gap: 14px;
  justify-content: space-between;
  margin-bottom: 16px;
}

.status-label {
  border-radius: 999px;
  color: #ffffff;
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 700;
  padding: 6px 10px;
}

.status-label.ai {
  background: #b42318;
}

.status-label.human {
  background: #19734d;
}

.status-label.review {
  background: #9a5b0a;
}

.status-label.error {
  background: #5b6472;
}

.status-copy h2 {
  color: #172033;
  font-size: 1.35rem;
  margin: 8px 0 6px;
}

.status-copy p {
  color: #566276;
  line-height: 1.5;
  margin: 0;
}

.score-card {
  min-width: 180px;
  text-align: right;
}

.score-value {
  color: #172033;
  font-size: 2rem;
  font-weight: 800;
  line-height: 1;
}

.score-caption {
  color: #677386;
  font-size: 0.84rem;
  margin-top: 4px;
}

.score-track {
  background: linear-gradient(90deg, #2f855a 0%, #d79b2f 55%, #b42318 100%);
  border-radius: 999px;
  height: 12px;
  margin: 12px 0 18px;
  position: relative;
}

.score-fill {
  background: #ffffff;
  border: 2px solid #172033;
  border-radius: 999px;
  height: 18px;
  position: absolute;
  top: -3px;
  transform: translateX(-50%);
  width: 18px;
  z-index: 2;
}

.threshold-marker {
  background: #172033;
  border-radius: 999px;
  height: 20px;
  position: absolute;
  top: -4px;
  width: 3px;
}

.threshold-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-top: 12px;
}

.metric {
  background: #f7f8fb;
  border: 1px solid #e4e8f0;
  border-radius: 8px;
  padding: 11px 12px;
}

.metric span {
  color: #687589;
  display: block;
  font-size: 0.78rem;
  margin-bottom: 4px;
}

.metric strong {
  color: #172033;
  display: block;
  font-size: 1rem;
}

.warning-note {
  background: #fff7ed;
  border: 1px solid #fed7aa;
  border-radius: 8px;
  color: #83470c;
  line-height: 1.45;
  margin-top: 14px;
  padding: 10px 12px;
}

.interpretation-note {
  color: #566276;
  font-size: 0.92rem;
  line-height: 1.5;
  margin: 10px 0 0;
}

@media (max-width: 760px) {
  .demo-header h1 {
    font-size: 1.55rem;
  }

  .status-row {
    display: block;
  }

  .score-card {
    margin-top: 14px;
    text-align: left;
  }

  .threshold-grid {
    grid-template-columns: 1fr;
  }
}
"""


def launch_demo(model_path: str | Path, *, server_name: str = "127.0.0.1", server_port: int = 7860):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before launching the demo.") from exc

    model_path = str(Path(model_path))

    def predict(text: str):
        if not text or not text.strip():
            return _error_html("Enter text to classify."), [], []
        bundle = _load_demo_bundle(model_path)
        explanation = bundle.explain_text(text, top_k=8)
        word_count = len(text.split())
        summary = _summary_html(explanation, word_count=word_count)
        group_rows = _group_rows(explanation["feature_group_contributions"])
        signal_rows = _signal_rows(explanation)
        return summary, group_rows, signal_rows

    theme = _light_theme(gr)

    with gr.Blocks(title="Interpretable AI-Text Detector") as app:
        gr.HTML(
            """
            <header class="demo-header">
              <p class="eyebrow">Interpretable AI-text detection</p>
              <h1>Classify a passage and inspect the evidence.</h1>
              <p>
                This research demo is designed for education-style writing. It reports an AI score,
                applies a stricter review threshold to reduce false positives, and explains which
                feature families pushed the decision toward AI-generated or human-written.
              </p>
            </header>
            """
        )
        with gr.Row():
            with gr.Column(scale=5):
                text = gr.Textbox(
                    label="Passage",
                    lines=12,
                    placeholder="Paste at least 80 words of essay-style text for the most reliable result.",
                    buttons=["copy"],
                )
                with gr.Row():
                    button = gr.Button("Classify text", variant="primary")
                    clear = gr.ClearButton(value="Clear", components=[text])
                gr.Examples(
                    examples=[[example[1]] for example in EXAMPLE_TEXTS],
                    inputs=text,
                    label="Examples",
                    examples_per_page=2,
                )
            with gr.Column(scale=4):
                summary = gr.HTML(value=INITIAL_RESULTS_HTML, label="Decision")

        gr.Markdown(
            "Positive contribution values push the model toward AI-generated; negative values push it "
            "toward human-written. Larger absolute values mean stronger evidence for this passage."
        )
        with gr.Tab("Evidence by Feature Family"):
            group_contributions = gr.Dataframe(
                headers=[
                    "Leans",
                    "Feature family",
                    "Net effect",
                    "Relative weight",
                    "Active signals",
                ],
                datatype=["str", "str", "number", "str", "number"],
                label="Grouped evidence",
                interactive=False,
                wrap=True,
            )
        with gr.Tab("Top Individual Signals"):
            contributions = gr.Dataframe(
                headers=["Leans", "Signal", "Contribution"],
                datatype=["str", "str", "number"],
                label="Strongest signals",
                interactive=False,
                wrap=True,
            )

        button.click(predict, inputs=[text], outputs=[summary, group_contributions, contributions])
        text.submit(predict, inputs=[text], outputs=[summary, group_contributions, contributions])
        clear.click(
            lambda: (INITIAL_RESULTS_HTML, [], []),
            inputs=None,
            outputs=[summary, group_contributions, contributions],
        )

    app.launch(server_name=server_name, server_port=server_port, theme=theme, css=CSS)


def _light_theme(gr):
    return gr.themes.Soft(primary_hue="teal", neutral_hue="gray").set(
        body_background_fill="#f7f8fb",
        body_background_fill_dark="#f7f8fb",
        body_text_color="#172033",
        body_text_color_dark="#172033",
        body_text_color_subdued="#566276",
        body_text_color_subdued_dark="#566276",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#f7f8fb",
        background_fill_secondary_dark="#f7f8fb",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color="#d9dee8",
        block_border_color_dark="#d9dee8",
        block_label_background_fill="#f7f8fb",
        block_label_background_fill_dark="#f7f8fb",
        block_label_text_color="#172033",
        block_label_text_color_dark="#172033",
        border_color_primary="#d9dee8",
        border_color_primary_dark="#d9dee8",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_background_fill_focus="#ffffff",
        input_background_fill_focus_dark="#ffffff",
        input_border_color="#cfd6e3",
        input_border_color_dark="#cfd6e3",
        input_border_color_focus="#1d6f78",
        input_border_color_focus_dark="#1d6f78",
        input_placeholder_color="#778396",
        input_placeholder_color_dark="#778396",
        table_text_color="#172033",
        table_text_color_dark="#172033",
        table_border_color="#d9dee8",
        table_border_color_dark="#d9dee8",
        table_even_background_fill="#ffffff",
        table_even_background_fill_dark="#ffffff",
        table_odd_background_fill="#f7f8fb",
        table_odd_background_fill_dark="#f7f8fb",
        panel_background_fill="#ffffff",
        panel_background_fill_dark="#ffffff",
        panel_border_color="#d9dee8",
        panel_border_color_dark="#d9dee8",
        button_primary_background_fill="#1d6f78",
        button_primary_background_fill_dark="#1d6f78",
        button_primary_background_fill_hover="#155e65",
        button_primary_background_fill_hover_dark="#155e65",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_dark="#ffffff",
        button_secondary_background_fill_hover="#eef3f6",
        button_secondary_background_fill_hover_dark="#eef3f6",
        button_secondary_border_color="#cfd6e3",
        button_secondary_border_color_dark="#cfd6e3",
        button_secondary_text_color="#172033",
        button_secondary_text_color_dark="#172033",
    )


@lru_cache(maxsize=4)
def _load_demo_bundle(model_path: str):
    return load_bundle(model_path)


def _summary_html(explanation: dict[str, Any], *, word_count: int) -> str:
    score = float(explanation["score_ai"])
    standard_threshold = float(explanation["threshold"])
    education_threshold = float(explanation["education_threshold"])
    mitigated_prediction = str(explanation["mitigated_prediction"])
    raw_prediction = str(explanation["prediction"])
    label, status_class, reason = _decision_copy(mitigated_prediction)
    score_pct = _format_percent(score)
    warning = ""
    if word_count < 80:
        warning = (
            '<div class="warning-note">'
            "Short text warning: reliability is lower below about 80 words. Treat this result as a "
            "screening signal, not a final judgment."
            "</div>"
        )

    review_zone = explanation.get("review_zone", {})
    review_active = bool(review_zone.get("active", False))
    review_text = (
        "Manual-review band is active"
        if review_active and standard_threshold <= score < education_threshold
        else "Outside manual-review band"
    )
    raw_label = _format_decision(raw_prediction)
    policy_label = _format_decision(mitigated_prediction)

    return f"""
    <section class="result-shell">
      <div class="status-row">
        <div class="status-copy">
          <span class="status-label {status_class}">{html.escape(label)}</span>
          <h2>{html.escape(policy_label)}</h2>
          <p>{html.escape(reason)}</p>
        </div>
        <div class="score-card">
          <div class="score-value">{score_pct}</div>
          <div class="score-caption">AI score</div>
        </div>
      </div>
      <div class="score-track" aria-label="AI score scale">
        <div class="score-fill" title="AI score" style="left: {_marker_position(score)}%;"></div>
        <div class="threshold-marker" title="Standard threshold"
          style="left: {_marker_position(standard_threshold)}%;"></div>
        <div class="threshold-marker" title="Education threshold"
          style="left: {_marker_position(education_threshold)}%;"></div>
      </div>
      <div class="threshold-grid">
        <div class="metric">
          <span>Raw model decision</span>
          <strong>{html.escape(raw_label)}</strong>
        </div>
        <div class="metric">
          <span>Standard threshold</span>
          <strong>{_format_percent(standard_threshold)}</strong>
        </div>
        <div class="metric">
          <span>Education threshold</span>
          <strong>{_format_percent(education_threshold)}</strong>
        </div>
        <div class="metric">
          <span>Word count</span>
          <strong>{word_count}</strong>
        </div>
        <div class="metric">
          <span>Review status</span>
          <strong>{html.escape(review_text)}</strong>
        </div>
        <div class="metric">
          <span>Recommended reading</span>
          <strong>{html.escape(_recommended_reading(mitigated_prediction))}</strong>
        </div>
      </div>
      <p class="interpretation-note">
        Scores above the education threshold are treated as stronger AI evidence. Scores between the
        standard and education thresholds should be reviewed by a person, especially for learner
        writing or short passages.
      </p>
      {warning}
    </section>
    """


def _error_html(message: str) -> str:
    return f"""
    <section class="result-shell">
      <span class="status-label error">Needs text</span>
      <div class="status-copy">
        <h2>No passage entered</h2>
        <p>{html.escape(message)}</p>
      </div>
    </section>
    """


def _group_rows(items: list[dict[str, Any]]) -> list[list[Any]]:
    total = sum(abs(float(item["absolute_contribution"])) for item in items) or 1.0
    rows = []
    for item in items:
        signed = float(item["signed_contribution"])
        absolute = abs(float(item["absolute_contribution"]))
        rows.append(
            [
                _lean_label(signed),
                item["feature_family"],
                round(signed, 3),
                _format_percent(absolute / total),
                int(item["n_active_features"]),
            ]
        )
    return rows


def _signal_rows(explanation: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for item in explanation["top_ai_features"]:
        rows.append(
            [
                "AI-generated",
                _clean_feature_name(str(item["feature"])),
                round(float(item["contribution"]), 3),
            ]
        )
    for item in explanation["top_human_features"]:
        rows.append(
            [
                "Human-written",
                _clean_feature_name(str(item["feature"])),
                round(float(item["contribution"]), 3),
            ]
        )
    return sorted(rows, key=lambda row: abs(float(row[2])), reverse=True)


def _decision_copy(decision: str) -> tuple[str, str, str]:
    if decision == "ai_generated":
        return (
            "Likely AI-generated",
            "ai",
            "The AI score clears the stricter education threshold, so the model sees strong AI-like evidence.",
        )
    if decision == "manual_review":
        return (
            "Manual review",
            "review",
            "The raw model leans AI, but the conservative education policy keeps this in the review band.",
        )
    return (
        "Likely human-written",
        "human",
        "The AI score is below the model threshold, so the passage looks more human-written to this detector.",
    )


def _format_decision(decision: str) -> str:
    labels = {
        "ai_generated": "AI-generated",
        "manual_review": "Manual review",
        "human_written": "Human-written",
    }
    return labels.get(decision, decision.replace("_", " ").title())


def _recommended_reading(decision: str) -> str:
    if decision == "ai_generated":
        return "Strong AI signal"
    if decision == "manual_review":
        return "Do not treat as proof"
    return "Low AI signal"


def _lean_label(value: float) -> str:
    if value > 0:
        return "AI-generated"
    if value < 0:
        return "Human-written"
    return "Neutral"


def _format_percent(value: float) -> str:
    return f"{max(0.0, min(1.0, value)) * 100:.1f}%"


def _marker_position(value: float) -> str:
    return f"{max(0.0, min(1.0, value)) * 100:.1f}"


def _clean_feature_name(feature: str) -> str:
    stats_labels = {
        "avg_word_len": "Average word length",
        "burstiness_index": "Sentence-length burstiness",
        "char_count": "Character count",
        "flesch_kincaid_grade": "Flesch-Kincaid grade level",
        "flesch_reading_ease": "Flesch reading ease",
        "hapax_ratio": "One-off word ratio",
        "sentence_count": "Sentence count",
        "sentence_len_mean": "Average sentence length",
        "sentence_len_p10": "Short sentence length",
        "sentence_len_p90": "Long sentence length",
        "stopword_ratio": "Stopword ratio",
        "type_token_ratio": "Vocabulary diversity",
        "unigram_repeat_ratio": "Repeated word ratio",
        "word_count": "Word count",
        "word_frequency_entropy": "Word-frequency entropy",
        "word_len_std": "Word-length variation",
    }
    if feature.startswith("basic_stats__stats:"):
        key = feature.rsplit(":", 1)[-1]
        return stats_labels.get(key, _titleize_feature(key))

    if "__" in feature:
        family, value = feature.split("__", 1)
        family_label = {
            "word_tfidf": "Word pattern",
            "char_tfidf": "Character pattern",
            "spacy_stats": "Linguistic feature",
            "lm_stats": "Language-model feature",
        }.get(family, _titleize_feature(family))
        return f"{family_label}: {_titleize_feature(value)}"

    return _titleize_feature(feature)


def _titleize_feature(value: str) -> str:
    value = re.sub(r"^stats:", "", value)
    value = value.replace("_", " ").replace(":", ": ")
    value = re.sub(r"\s+", " ", value).strip()
    return value[:1].upper() + value[1:]
