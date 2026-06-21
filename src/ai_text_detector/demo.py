from __future__ import annotations

from pathlib import Path

from ai_text_detector.model import load_bundle


def launch_demo(model_path: str | Path, *, server_name: str = "127.0.0.1", server_port: int = 7860):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before launching the demo.") from exc

    bundle = load_bundle(model_path)

    def predict(text: str):
        if not text or not text.strip():
            return {"error": "Enter text to classify."}, [], []
        explanation = bundle.explain_text(text, top_k=8)
        warning = ""
        if len(text.split()) < 80:
            warning = "Short text warning: detector reliability is lower below about 80 words."
        summary = {
            "prediction": explanation["prediction"],
            "mitigated_prediction": explanation["mitigated_prediction"],
            "score_ai": round(explanation["score_ai"], 4),
            "threshold": round(explanation["threshold"], 4),
            "education_threshold": round(explanation["education_threshold"], 4),
            "review_zone": explanation["review_zone"],
            "warning": warning,
        }
        group_rows = []
        for item in explanation["feature_group_contributions"]:
            direction = "AI" if item["signed_contribution"] > 0 else "Human"
            group_rows.append(
                [
                    direction,
                    item["feature_family"],
                    item["signed_contribution"],
                    item["absolute_contribution"],
                    item["n_active_features"],
                ]
            )
        rows = []
        for item in explanation["top_ai_features"]:
            rows.append(["AI", item["feature"], item["contribution"]])
        for item in explanation["top_human_features"]:
            rows.append(["Human", item["feature"], item["contribution"]])
        return summary, group_rows, rows

    with gr.Blocks(title="Interpretable AI-Text Detector") as app:
        gr.Markdown("# Interpretable AI-Text Detector")
        text = gr.Textbox(label="Text", lines=10)
        button = gr.Button("Classify")
        summary = gr.JSON(label="Prediction")
        group_contributions = gr.Dataframe(
            headers=["Direction", "Feature family", "Signed contribution", "Absolute contribution", "Active features"],
            datatype=["str", "str", "number", "number", "number"],
            label="Grouped feature contributions",
        )
        contributions = gr.Dataframe(
            headers=["Direction", "Feature", "Contribution"],
            datatype=["str", "str", "number"],
            label="Raw top feature contributions",
        )
        button.click(predict, inputs=[text], outputs=[summary, group_contributions, contributions])

    app.launch(server_name=server_name, server_port=server_port)
