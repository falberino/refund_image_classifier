from src.utils.config import load_config


def test_load_config_has_required_sections():
    cfg = load_config()
    for section in ["dataset", "training", "paths", "api"]:
        assert section in cfg
