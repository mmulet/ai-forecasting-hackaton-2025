# pyright: standard
from pathlib import Path

def get_config_path(config_name: str) -> Path:
    """Get the path to the configuration directory."""
    return Path(__file__).resolve().parent.parent / "mas_configs" / f"{config_name}.yaml"