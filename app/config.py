import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Configuration for the Gradio app."""
    model_path: str = os.getenv("MODEL_PATH", "RousingSea7309/sentiment-distilbert-imdb")
    title: str = "🎬 Movie Review Sentiment Analyzer"
    description: str = """
    **DistilBERT fine-tuned on IMDB reviews** | Try single reviews, batch mode, or test edge cases!
    """
    max_history: int = 10
    confidence_threshold_default: float = 0.7
    max_text_length: int = 2000  # chars, to prevent abuse


config = AppConfig()
