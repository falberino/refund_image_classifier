from src.training.pipeline import run_training
from src.utils.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    results = run_training(cfg)
    print("Training completed:")
    for k, v in results.items():
        print(f"- {k}: {v}")
