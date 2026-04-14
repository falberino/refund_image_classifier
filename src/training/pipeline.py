from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.data.dataset import load_datasets
from src.training.evaluate import evaluate_model, save_metrics, save_training_history
from src.training.model import build_classifier
from src.utils.logging_utils import setup_logger


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config["paths"]
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    logger = setup_logger("training", Path(paths["logs_dir"]) / "training.log")
    logger.info("Starting training pipeline")

    image_size = tuple(dataset_cfg["image_size"])
    train_ds, val_ds, test_ds, class_names = load_datasets(
        data_dir=dataset_cfg["data_dir"],
        image_size=image_size,
        batch_size=training_cfg["batch_size"],
        seed=training_cfg["seed"],
        validation_split=training_cfg["validation_split"],
        test_split=training_cfg["test_split"],
        class_names=dataset_cfg.get("class_names") or None,
    )

    logger.info("Detected classes: %s", class_names)

    model = build_classifier(
        image_size=image_size,
        num_classes=len(class_names),
        learning_rate=training_cfg["learning_rate"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_cfg["epochs"],
    )

    eval_metrics = evaluate_model(model, test_ds, class_names)

    model_path = Path(paths["model_file"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info("Saved model to %s", model_path)

    class_map_path = Path(paths["class_map_file"])
    class_map_path.parent.mkdir(parents=True, exist_ok=True)
    class_map = {str(idx): name for idx, name in enumerate(class_names)}
    with class_map_path.open("w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    save_training_history(history, paths["history_plot_file"])
    save_metrics(eval_metrics, paths["metrics_dir"])

    final_val_acc = history.history.get("val_accuracy", [None])[-1]
    results = {
        "model_path": str(model_path),
        "class_map_path": str(class_map_path),
        "final_val_accuracy": final_val_acc,
        "classification_report_path": str(Path(paths["metrics_dir"]) / "classification_report.json"),
        "confusion_matrix_path": str(Path(paths["metrics_dir"]) / "confusion_matrix.json"),
    }

    logger.info("Training pipeline completed")
    return results
