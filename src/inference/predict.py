from transformers import pipeline


class SentimentPredictor:
    """Wrapper for easy inference."""

    def __init__(self, model_path: str):
        print(f"🔮 Loading model from: {model_path}")
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_path,
            top_k=None,
        )

    def predict(self, text: str) -> dict:
        """Return label and score."""
        result = self.classifier(text)[0]
        # Convert to {label: score} format
        return {r["label"]: r["score"] for r in result}


if __name__ == "__main__":
    predictor = SentimentPredictor("distilbert-base-uncased")
    print(predictor.predict("This movie was absolutely fantastic!"))
