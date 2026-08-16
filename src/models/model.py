from transformers import AutoModelForSequenceClassification
from configs.config import config


def load_model():
    """Load pre-trained model with classification head."""
    print(f"🧠 Loading model: {config.model.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name,
        num_labels=config.model.num_labels,
        cache_dir=config.model.cache_dir,
    )
    return model
