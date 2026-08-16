import gradio as gr
import sys
from pathlib import Path

# Add project root to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predict import SentimentPredictor
from app.config import config
from app.utils import (
    clean_text,
    truncate_text,
    format_prediction,
    PredictionHistory,
)


# ──────────────────────────────────────────────────────────
# Initialize predictor and history
# ──────────────────────────────────────────────────────────
print("🚀 Loading model...")
predictor = SentimentPredictor(config.model_path)
history = PredictionHistory(max_size=config.max_history)


# ──────────────────────────────────────────────────────────
# Core prediction functions
# ──────────────────────────────────────────────────────────
def predict_single(text: str, threshold: float):
    """Predict sentiment for a single text."""
    if not text or not text.strip():
        return "⚠️ Please enter some text.", history.to_dataframe()

    text, was_truncated = truncate_text(clean_text(text), config.max_text_length)
    if was_truncated:
        text += " [truncated]"

    result = predictor.predict(text)
    history.add(text, result)
    formatted = format_prediction(result, threshold)

    return formatted, history.to_dataframe()


def predict_batch(texts: str, threshold: float):
    """Predict sentiment for multiple texts (one per line)."""
    if not texts or not texts.strip():
        return "⚠️ Please enter at least one line.", ""

    lines = [ln.strip() for ln in texts.split("\n") if ln.strip()]
    if not lines:
        return "⚠️ No valid lines found.", ""

    if len(lines) > 50:
        return f"⚠️ Too many lines ({len(lines)}). Maximum is 50.", ""

    # Run predictions
    results = []
    for line in lines:
        clean = clean_text(line)
        if not clean:
            continue
        pred = predictor.predict(clean)
        label, score = max(pred.items(), key=lambda x: x[1])
        flag = " ⚠️" if score < threshold else ""
        results.append(f"{line[:60]}{'...' if len(line) > 60 else ''}\n → {label} ({score:.1%}){flag}\n")

    summary = f"✅ Processed {len(results)} review(s)"
    return summary, "\n".join(results)


def clear_history():
    """Clear the prediction history."""
    history.clear()
    return "", history.to_dataframe()


def load_example(example_text: str):
    """Load an example into the input box."""
    return example_text


# ──────────────────────────────────────────────────────────
# Build the Gradio interface
# ──────────────────────────────────────────────────────────
def build_app():
    with gr.Blocks(
        title=config.title,
        theme=gr.themes.Soft(primary_hue="indigo"),
    ) as demo:

        gr.Markdown(f"# {config.title}")
        gr.Markdown(config.description)

        # Threshold slider at the top (shared)
        with gr.Row():
            threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=config.confidence_threshold_default,
                step=0.05,
                label="🎯 Confidence Threshold",
                info="Predictions below this are flagged as potentially unreliable.",
            )

        # ─── Tabs: Single / Batch / About ─────────────────────
        with gr.Tabs():

            # ── Tab 1: Single Prediction ──────────────────────
            with gr.Tab("🔍 Single Review"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Movie Review",
                            placeholder="Type or paste a movie review here...",
                            lines=6,
                        )
                        with gr.Row():
                            predict_btn = gr.Button("🔮 Predict", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear")
                        gr.Examples(
                            examples=[
                                ["This film was an absolute masterpiece. The acting, the cinematography, the score — everything was perfect. A must-watch!"],
                                ["What a complete waste of two hours. The plot made no sense, the characters were wooden, and the ending was insultingly bad."],
                                ["It was okay. Nothing special, but not the worst either. I'd watch it on a lazy Sunday."],
                                ["I LOVED the action scenes but HATED the dialogue. Mixed feelings overall."],
                                ["The best movie of 2024. Can't wait for the sequel!"],
                            ],
                            inputs=text_input,
                            label="📚 Try these examples",
                        )

                    with gr.Column(scale=1):
                        output_text = gr.Markdown(label="Prediction")
                        history_table = gr.Dataframe(
                            headers=["Time", "Text", "Label", "Score"],
                            label="📜 Session History (last 10)",
                            interactive=False,
                            wrap=True,
                        )

                predict_btn.click(
                    fn=predict_single,
                    inputs=[text_input, threshold_slider],
                    outputs=[output_text, history_table],
                )
                clear_btn.click(
                    fn=clear_history,
                    outputs=[output_text, history_table],
                )

            # ── Tab 2: Batch Prediction ──────────────────────
            with gr.Tab("📦 Batch Mode"):
                gr.Markdown(
                    "Paste multiple reviews (one per line). Max **50 reviews** at a time."
                )
                with gr.Row():
                    with gr.Column():
                        batch_input = gr.Textbox(
                            label="Reviews (one per line)",
                            placeholder="Review 1\nReview 2\nReview 3...",
                            lines=10,
                        )
                        batch_btn = gr.Button("🔮 Predict All", variant="primary")
                    with gr.Column():
                        batch_summary = gr.Markdown(label="Summary")
                        batch_output = gr.Textbox(
                            label="Detailed Results",
                            lines=12,
                            interactive=False,
                        )

                batch_btn.click(
                    fn=predict_batch,
                    inputs=[batch_input, threshold_slider],
                    outputs=[batch_summary, batch_output],
                )

                gr.Examples(
                    examples=[
                        [[
                            "Absolutely loved it!\n",
                            "Worst movie ever.\n",
                            "It was fine, nothing special.\n",
                            "Brilliant performances and stunning visuals.\n",
                            "Boring and predictable.\n",
                        ]],
                    ],
                    inputs=batch_input,
                    label="📚 Batch example",
                )

            # ── Tab 3: About / Help ──────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    f"""
                    ## About this app

                    **Model**: `{config.model_path}`

                    **Architecture**: DistilBERT (66M parameters) fine-tuned on the IMDB dataset.

                    ### Features
                    - 🔍 **Single Review** — Analyze one review at a time with full confidence analysis
                    - 📦 **Batch Mode** — Process up to 50 reviews at once
                    - 🎯 **Threshold Filter** — Flag low-confidence predictions for manual review
                    - 📜 **Session History** — Track your last 10 predictions
                    - 📚 **Examples** — Pre-loaded test cases including edge cases (sarcasm, mixed opinions)

                    ### Confidence Levels
                    - 🟢 **Very High** (≥ 90%)
                    - 🟢 **High** (≥ 75%)
                    - 🟡 **Medium** (at threshold)
                    - 🔴 **Low** (below threshold) — flagged for review

                    ### Limits
                    - Max text length: **{config.max_text_length} characters** (longer text is auto-truncated)
                    - Batch size: **50 reviews** max
                    - History: last **{config.max_history}** predictions

                    ---
                    💡 **Tip**: Try the model with sarcasm, mixed reviews, or non-English text to see how it handles edge cases.
                    """
                )

    return demo


# ──────────────────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
