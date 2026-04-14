from src.batch.run_batch import run_batch_inference
from src.utils.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    output_path = run_batch_inference(cfg)
    print(f"Batch predictions saved to: {output_path}")
