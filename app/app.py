import gradio as gr
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predict import SentimentPredictor

# Load from Hub when deployed, or local path during dev
MODEL_PATH = "RousingSea7309/sentiment-distilbert-imdb"  # HF Hub ID
predictor = SentimentPredictor(MODEL_PATH)


def analyze(text):
    return predictor.predict(text)


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=4, placeholder="Type a movie review..."),
    outputs=gr.Label(num_top_classes=2),
    title="🎬 Movie Review Sentiment Analyzer",
    description="Powered by DistilBERT fine-tuned on IMDB",
    examples=[
        ["This movie was absolutely incredible, I loved every second!"],
        ["Terrible plot, bad acting, complete waste of time."],
    ],
)

if __name__ == "__main__":
    demo.launch()
