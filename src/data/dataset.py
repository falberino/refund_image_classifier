from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf


def load_datasets(
    data_dir: str | Path,
    image_size: Tuple[int, int],
    batch_size: int,
    seed: int,
    validation_split: float,
    test_split: float,
    class_names: Optional[list[str]] = None,
):
    """Load train/val/test tf datasets from a class-folder image dataset."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    holdout_split = validation_split + test_split
    if not 0 < holdout_split < 1:
        raise ValueError("validation_split + test_split must be between 0 and 1")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names if class_names else None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=holdout_split,
        subset="training",
    )

    holdout_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names if class_names else None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=holdout_split,
        subset="validation",
    )
    detected_class_names = list(train_ds.class_names)

    holdout_batches = tf.data.experimental.cardinality(holdout_ds).numpy()
    val_ratio_in_holdout = validation_split / holdout_split
    val_batches = max(1, int(holdout_batches * val_ratio_in_holdout))

    val_ds = holdout_ds.take(val_batches)
    test_ds = holdout_ds.skip(val_batches)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, detected_class_names
