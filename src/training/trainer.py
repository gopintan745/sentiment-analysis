from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from configs.config import config
from training.evaluate import compute_metrics


def build_trainer(model, train_dataset, eval_dataset, tokenizer):
    """Build the HuggingFace Trainer."""
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        evaluation_strategy=config.training.evaluation_strategy,
        fp16=config.training.fp16,
        report_to="none",  # set to "wandb" if you want to log
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
