import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import load_and_prepare_data
from src.models.model import load_model
from src.training.trainer import build_trainer


def main():
    print("🚀 Starting training pipeline\n")

    # 1. Data
    train_ds, eval_ds, tokenizer = load_and_prepare_data()

    # 2. Model
    model = load_model()

    # 3. Trainer
    trainer = build_trainer(model, train_ds, eval_ds, tokenizer)

    # 4. Train!
    print("\n🏋️ Training...")
    trainer.train()

    # 5. Final evaluation
    print("\n📊 Final evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    # 6. Save locally
    save_path = Path("models/final")
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ Model saved to: {save_path}")


if __name__ == "__main__":
    main()
