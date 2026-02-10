"""
Environment and configuration utilities.

Usage:
    from src.common.utils import load_env, load_config, get_api_key
"""

import os
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.common.path import PROJECT_ROOT, CONFIGS_DIR


def load_env() -> Dict[str, str]:
    """
    Load .env file from project root and return key environment variables.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "NVIDIA_API_KEY": os.getenv("NVIDIA_API_KEY", ""),
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "WANDB_DISABLED": os.getenv("WANDB_DISABLED", "true"),
    }


def get_api_key(key_name: str = "NVIDIA_API_KEY", required: bool = True) -> str:
    """
    Load .env and return a specific API key.

    Args:
        key_name: Environment variable name
        required: If True, raise ValueError when key is missing

    Raises:
        ValueError: If required=True and key is not found
    """
    load_dotenv(PROJECT_ROOT / ".env")
    value = os.getenv(key_name, "")
    if required and not value:
        raise ValueError(
            f"{key_name} not found in environment. "
            f"Please set it in {PROJECT_ROOT / '.env'}"
        )
    return value


def load_config(
    path: Optional[str] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load YAML config with optional profile merging.

    Args:
        path: Path to YAML config (defaults to configs/default.yaml)
        profile: Profile name to merge on top of defaults
                 (e.g. "llama_generator", "mistral_evaluator")

    Returns:
        Merged config dict. If profile is given, defaults are updated
        with profile-specific overrides (deep merge one level).

    Example:
        cfg = load_config(profile="mistral_evaluator")
        lr = cfg["train"]["learning_rate"]
    """
    if path is None:
        path = str(CONFIGS_DIR / "default.yaml")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if profile is None:
        return raw

    defaults = raw.get("defaults", {})
    profiles = raw.get("profiles", {})

    if profile not in profiles:
        available = list(profiles.keys())
        raise KeyError(f"Profile '{profile}' not found. Available: {available}")

    overrides = profiles[profile]

    # Shallow-deep merge: for each top-level key in overrides,
    # if both sides are dicts, merge them; otherwise override.
    merged = {}
    for key in set(list(defaults.keys()) + list(overrides.keys())):
        d_val = defaults.get(key)
        o_val = overrides.get(key)
        if isinstance(d_val, dict) and isinstance(o_val, dict):
            merged[key] = {**d_val, **o_val}
        elif o_val is not None:
            merged[key] = o_val
        else:
            merged[key] = d_val

    # Add top-level scalars from overrides (base_model, task, output_dir, etc.)
    for key in overrides:
        if key not in defaults:
            merged[key] = overrides[key]

    # Attach project-level settings
    merged["project"] = raw.get("project", {})

    return merged
