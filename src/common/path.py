"""
Project path resolution.

Usage:
    from src.common.path import PROJECT_ROOT, DATA_DIR
"""

from pathlib import Path
import os


def get_project_root() -> Path:
    """
    Resolve project root directory.

    Priority:
        1. PROJECT_ROOT environment variable (if set)
        2. Walk up from this file: src/common/path.py -> common -> src -> project
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    return Path(__file__).resolve().parents[2]


def find_project_root_from_cwd() -> Path:
    """
    Walk up from cwd until we find a directory containing 'src/'.
    Useful in Jupyter notebooks where __file__ is not available.

    Usage (in notebooks):
        from src.common.path import find_project_root_from_cwd
        PROJECT_ROOT = find_project_root_from_cwd()
    """
    p = Path.cwd().resolve()
    while not (p / "src").exists() and p.parent != p:
        p = p.parent
    if not (p / "src").exists():
        raise FileNotFoundError("Could not find project root (no 'src/' directory found)")
    return p


PROJECT_ROOT = get_project_root()

DATA_DIR       = PROJECT_ROOT / "data"
LABELED_DIR    = DATA_DIR / "labeled_data"
CONFIGS_DIR    = PROJECT_ROOT / "configs"
OUTPUTS_DIR    = PROJECT_ROOT / "outputs"
