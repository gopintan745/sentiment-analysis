from datasets import load_dataset
from transformers import AutoTokenizer
from configs.config import config


def load_and_prepare_data():
    """Load IMDB dataset and apply tokenization."""
    print(f"📥 Loading dataset: {config.data.dataset_name}")
    raw_datasets = load_dataset(config.data.dataset_name)

    print(f"🔤 Loading tokenizer: {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        cache_dir=config.model.cache_dir,
    )

    def tokenize(batch):
        return tokenizer(
            batch[config.data.text_column],
            padding="max_length",
            truncation=True,
            max_length=config.data.max_length,
        )

    tokenized = raw_datasets.map(tokenize, batched=True)

    # Use subsets for faster training (great for learning!)
    train_ds = tokenized["train"].select(range(config.data.train_subset))
    eval_ds = tokenized["test"].select(range(config.data.eval_subset))

    print(f"✅ Train size: {len(train_ds)} | Eval size: {len(eval_ds)}")
    return train_ds, eval_ds, tokenizer
