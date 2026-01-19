from __future__ import annotations

import re
import argparse
from pathlib import Path
import pandas as pd


def clean_text(s: str) -> str:
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def unify_one_csv(path: Path, source: str, text_col: str, label_col: str | None):
    df = pd.read_csv(path)

    # text
    if text_col not in df.columns:
        raise ValueError(f"{path.name}: missing text_col='{text_col}'. columns={list(df.columns)}")
    text = df[text_col].astype(str).map(clean_text)

    # label (optional)
    if label_col is None:
        label = pd.Series([None] * len(df))
    else:
        if label_col not in df.columns:
            raise ValueError(f"{path.name}: missing label_col='{label_col}'. columns={list(df.columns)}")
        label = df[label_col]

    out = pd.DataFrame({
        "id": [f"{source}-{i}" for i in range(len(df))],
        "text": text,
        "label": label,
        "source": source,
    })

    # drop empty texts
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data", help="folder that contains your datasets")
    parser.add_argument("--out", default="data/unified.csv", help="output unified csv")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)

    # 🔧 CONFIG: define your datasets here (file -> mapping)
    DATASETS = [
        # Example:
        # {"file": "dataset1.csv", "source": "dataset1", "text_col": "tweet", "label_col": "humor"},
        # {"file": "dataset2.csv", "source": "dataset2", "text_col": "text",  "label_col": None},
    ]

    if not DATASETS:
        raise SystemExit(
            "Please fill DATASETS config list inside the script with your file names and column mappings."
        )

    parts = []
    for spec in DATASETS:
        path = in_dir / spec["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

        parts.append(
            unify_one_csv(
                path=path,
                source=spec["source"],
                text_col=spec["text_col"],
                label_col=spec.get("label_col"),
            )
        )

    unified = pd.concat(parts, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} with {len(unified)} rows")


if __name__ == "__main__":
    main()
