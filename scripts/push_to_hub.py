import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import login
from configs.config import config


def main():
    # Login
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("❌ HF_TOKEN not set in environment")
    login(token=token)

    # Push
    model_path = "models/final"
    repo_id = config.hub.repo_id

    print(f"☁️ Pushing model to: {repo_id}")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print(f"✅ Model available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
