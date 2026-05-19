"""
Config loader utility.
Loads YAML config files and resolves paths relative to the project root.
"""

import os
from pathlib import Path
import yaml

# Project root is two levels above this file: src/cfpb_assistant/utils/ -> root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_config(config_name: str) -> dict:
    """
    Load a YAML config file from the configs/ directory.

    Args:
        config_name: filename without extension, e.g. "preprocessing"

    Returns:
        dict with config values
    """
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / relative_path


def ensure_dir(path) -> Path:
    """Create directory if it does not exist and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
