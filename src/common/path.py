from pathlib import Path
import os

def get_project_root() -> Path:
    # 1️⃣ ENV hat Priorität
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root)

    # 2️⃣ Fallback: Datei-Lage
    # src/utils/path.py → utils → src → project
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
