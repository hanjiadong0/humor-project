import os, yaml
from dotenv import load_dotenv

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_env():
    load_dotenv()
    return {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "WANDB_DISABLED": os.getenv("WANDB_DISABLED", "true"),
    }
