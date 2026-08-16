import evaluate
import numpy as np


def compute_metrics(eval_pred):
    """Compute accuracy and F1 score."""
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }
