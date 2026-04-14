from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from src.inference.predictor import ImagePredictor, is_supported_image
from src.utils.logging_utils import setup_logger


def _load_manifest(manifest_path: Path) -> Set[str]:
    if not manifest_path.exists():
        return set()
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("processed", []))


def _save_manifest(manifest_path: Path, processed: Set[str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"processed": sorted(processed)}, f, indent=2)


def run_batch_inference(config: Dict[str, Any]) -> Path:
    paths = config["paths"]
    batch_input = Path(paths["batch_input_dir"])
    batch_output = Path(paths["batch_output_dir"])
    archive_dir = Path(paths["archive_dir"])
    manifest_path = Path(paths["processed_manifest_file"])

    logger = setup_logger("batch", Path(paths["logs_dir"]) / "batch.log")

    batch_output.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    batch_input.mkdir(parents=True, exist_ok=True)

    predictor = ImagePredictor(
        model_path=paths["model_file"],
        class_map_path=paths["class_map_file"],
        image_size=tuple(config["dataset"]["image_size"]),
    )

    processed = _load_manifest(manifest_path)
    candidates = [p for p in batch_input.iterdir() if p.is_file() and is_supported_image(p)]
    to_process = [p for p in candidates if p.name not in processed]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = batch_output / f"predictions_{ts}.csv"

    logger.info("Batch run started. Found %d new images.", len(to_process))

    rows: List[Dict[str, object]] = []
    for image_path in to_process:
        try:
            pred = predictor.predict_bytes(image_path.read_bytes())
            row = {
                "filename": image_path.name,
                "predicted_label": pred["predicted_label"],
                "predicted_index": pred["predicted_index"],
                "probabilities_json": json.dumps(pred["probabilities"]),
            }
            rows.append(row)

            archived_path = archive_dir / image_path.name
            image_path.rename(archived_path)
            processed.add(image_path.name)
        except Exception as exc:
            logger.exception("Failed image %s: %s", image_path, exc)

    fieldnames = ["filename", "predicted_label", "predicted_index", "probabilities_json"]
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _save_manifest(manifest_path, processed)
    logger.info("Batch run completed. Wrote %d predictions to %s", len(rows), output_file)
    return output_file
