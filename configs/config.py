from dataclasses import dataclass, field
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    dataset_name: str = "imdb"
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 256
    train_subset: int = 5000   # Use subset for faster training
    eval_subset: int = 1000


@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    cache_dir: str = str(PROJECT_ROOT / "models" / "base")


@dataclass
class TrainingConfig:
    output_dir: str = str(PROJECT_ROOT / "checkpoints")
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    fp16: bool = True  # Use mixed precision on GPU


@dataclass
class HubConfig:
    repo_id: str = f"{os.getenv('HF_USERNAME', 'your-username')}/sentiment-distilbert-imdb"
    private: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hub: HubConfig = field(default_factory=HubConfig)


config = Config()
