"""Small helpers shared by the training, experiment, and monitoring scripts."""

import copy
import hashlib
from pathlib import Path

import yaml


def load_config(path):
    """Read a YAML config file and return it as a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or invalid: {path}")
    return config


def merge_config(base_config, overrides):
    """Return a copy of base_config with one experiment's overrides applied.

    Each top-level section in overrides (for example "model") is updated key by key.
    A nested value such as model.params is replaced as a whole, so switching model
    types never leaves behind parameters from the previous model.
    """
    merged = copy.deepcopy(base_config)
    for section, values in (overrides or {}).items():
        if section not in merged:
            raise KeyError(f"Unknown config section in overrides: {section}")
        merged[section].update(copy.deepcopy(values))
    return merged


def file_md5(path):
    """Return the MD5 hash of a file. This matches the hash DVC uses for the file."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()