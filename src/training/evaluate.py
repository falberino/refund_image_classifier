from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def save_training_history(history, plot_path: str | Path) -> None:
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    hist = history.history
    plt.figure(figsize=(8, 4))
    plt.plot(hist.get("accuracy", []), label="train_acc")
    plt.plot(hist.get("val_accuracy", []), label="val_acc")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def evaluate_model(model, test_ds, class_names: List[str]) -> Dict[str, object]:
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()

    return {
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def save_metrics(metrics: Dict[str, object], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(metrics["classification_report"], f, indent=2)

    with (output_dir / "confusion_matrix.json").open("w", encoding="utf-8") as f:
        json.dump(metrics["confusion_matrix"], f, indent=2)
