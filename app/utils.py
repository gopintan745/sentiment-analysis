"""Utility functions for the Gradio app."""
import re
from datetime import datetime
from typing import List, Dict, Tuple


def clean_text(text: str) -> str:
    """Basic text cleanup."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    return text


def truncate_text(text: str, max_chars: int) -> Tuple[str, bool]:
    """Truncate text and return a flag indicating if truncation occurred."""
    if len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def format_prediction(result: Dict[str, float], threshold: float) -> str:
    """Format a single prediction with confidence analysis."""
    label, score = max(result.items(), key=lambda x: x[1])

    # Build emoji + label
    emoji_map = {
        "LABEL_0": "😞",
        "LABEL_1": "😊",
        "NEGATIVE": "😞",
        "POSITIVE": "😊",
        "negative": "😞",
        "positive": "😊",
    }
    emoji = emoji_map.get(label, "🤔")

    # Confidence level
    if score >= 0.9:
        confidence = "🟢 Very High"
    elif score >= 0.75:
        confidence = "🟢 High"
    elif score >= threshold:
        confidence = "🟡 Medium"
    else:
        confidence = "🔴 Low"

    below_threshold = score < threshold
    warning = "⚠️ Below threshold — prediction may be unreliable\n\n" if below_threshold else ""

    return f"{warning}{emoji} **{label}**\nConfidence: {score:.1%}\nLevel: {confidence}"


class PredictionHistory:
    """Manages a list of past predictions for the current session."""

    def __init__(self, max_size: int = 10):
        self.history: List[Dict] = []
        self.max_size = max_size

    def add(self, text: str, result: Dict[str, float]):
        """Add a prediction to the history."""
        if not result:
            return

        label, score = max(result.items(), key=lambda x: x[1])
        self.history.insert(0, {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "text_preview": text[:80] + ("..." if len(text) > 80 else ""),
            "label": label,
            "score": score,
        })
        # Keep only the most recent N
        self.history = self.history[:self.max_size]

    def clear(self):
        self.history = []

    def to_dataframe(self):
        """Return history as a list of lists for Gradio Dataframe."""
        if not self.history:
            return [["—", "No predictions yet", "—", "—"]]
        return [
            [item["timestamp"], item["text_preview"], item["label"], f"{item['score']:.1%}"]
            for item in self.history
        ]
