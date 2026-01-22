from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from src.utils.utils import load_config


SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet"}


# =========================
# 1) Data loading
# =========================
@dataclass
class DataLoader:
    raw_dir: Path

    def list_files(self) -> list[Path]:
        files: list[Path] = []
        for f in self.raw_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(f)
        return files

    def load(self, path: Path) -> pd.DataFrame:
        suf = path.suffix.lower()
        if suf == ".csv":
            return pd.read_csv(path)
        if suf in [".jsonl", ".json"]:
            return pd.read_json(path, lines=(suf == ".jsonl"))
        if suf == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file: {path}")


# =========================
# 2) Transform / normalize
# =========================
@dataclass
class Normalizer:
    # candidates
    text_candidates: list[str]
    cls_candidates: list[str]
    reg_candidates: list[str]

    # label config
    reg_min: float = 0.0
    reg_max: float = 5.0
    cls_from_reg_threshold: float = 3.0

    # robust scaling config
    reg_quantile: float = 0.95

    # learned (fit)
    q_ref: Optional[float] = None

    def pick_col(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # ---------- CLS ----------
    def normalize_cls_value(self, x) -> int:
        if pd.isna(x):
            return -1
        if isinstance(x, (bool, np.bool_)):
            return int(x)
        s = str(x).strip().lower()
        if s in ["1", "true", "yes", "y", "humor", "funny"]:
            return 1
        if s in ["0", "false", "no", "n", "not humor", "not funny"]:
            return 0
        try:
            v = float(s)
            if v in [0.0, 1.0]:
                return int(v)
        except:
            pass
        return -1

    # ---------- REG (robust scaling with q_ref) ----------
    def normalize_reg_value(self, x) -> float:
        if pd.isna(x):
            return -1.0
        try:
            v = float(x)
        except:
            return -1.0
        if v < 0 or self.q_ref is None or self.q_ref <= 0:
            return -1.0
        v = v * self.reg_max / self.q_ref
        v = max(self.reg_min, min(self.reg_max, v))
        return float(v)

    def fit_quantile(self, loader: DataLoader, files: Iterable[Path]) -> None:
        """
        Fit step: compute q_ref from all valid reg values >=0 across files.
        (Use TRAIN set only if you have a split.)
        """
        vals = []

        for f in files:
            df = loader.load(f)

            text_col = self.pick_col(df, self.text_candidates)
            if text_col is None:
                continue

            reg_col = self.pick_col(df, self.reg_candidates)
            if reg_col is None:
                continue

            s = pd.to_numeric(df[reg_col], errors="coerce")
            s = s[s >= 0]
            if len(s) > 0:
                vals.append(s.to_numpy())

        if not vals:
            self.q_ref = None
            return

        all_vals = np.concatenate(vals, axis=0)
        if all_vals.size == 0:
            self.q_ref = None
            return

        self.q_ref = float(np.quantile(all_vals, self.reg_quantile))

    def transform_file(self, loader: DataLoader, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Transform one file into unified schema: id, text, y_cls, y_reg, source
        Returns None if cannot find text column or file yields no valid rows.
        """
        df = loader.load(file_path)

        text_col = self.pick_col(df, self.text_candidates)
        if text_col is None:
            return None

        cls_col = self.pick_col(df, self.cls_candidates)
        reg_col = self.pick_col(df, self.reg_candidates)

        # --- clean text (vectorized) ---
        text = df[text_col].astype(str).str.strip()
        mask = text.notna() & (text.str.len() > 0)
        if mask.sum() == 0:
            return None

        df2 = df.loc[mask].copy()
        text2 = text.loc[mask]

        source = file_path.stem
        ids = (source + "_" + df2.index.astype(str))

        # --- y_cls ---
        if cls_col is not None:
            y_cls = df2[cls_col].apply(self.normalize_cls_value).astype(int)
            y_cls = y_cls.where(y_cls.isin([0, 1]), -1)
        else:
            y_cls = pd.Series([-1] * len(df2), index=df2.index, dtype=int)

        # --- y_reg ---
        if reg_col is not None and self.q_ref is not None and self.q_ref > 0:
            # use vectorized path for speed
            s = pd.to_numeric(df2[reg_col], errors="coerce")
            y_reg = pd.Series([-1.0] * len(df2), index=df2.index, dtype=float)

            valid = s.notna() & (s >= 0)
            scaled = (s[valid] * self.reg_max / self.q_ref).clip(self.reg_min, self.reg_max)
            y_reg.loc[valid] = scaled.astype(float)
        else:
            y_reg = pd.Series([-1.0] * len(df2), index=df2.index, dtype=float)

        # --- derive cls from reg only if cls missing ---
        missing_cls = (y_cls == -1) & (y_reg >= 0)
        y_cls.loc[missing_cls] = (y_reg.loc[missing_cls] >= self.cls_from_reg_threshold).astype(int)

        return pd.DataFrame(
            {
                "id": ids.values,
                "text": text2.values,
                "y_cls": y_cls.values,
                "y_reg": y_reg.values,
                "source": source,
            }
        )


# =========================
# 3) Build unified dataset
# =========================
@dataclass
class DatasetBuilder:
    loader: DataLoader
    normalizer: Normalizer
    out_path: Path

    def build(self) -> pd.DataFrame:
        files = self.loader.list_files()
        if not files:
            raise RuntimeError(f"No supported files in {self.loader.raw_dir}")

        # Fit robust reference (q95)
        self.normalizer.fit_quantile(self.loader, files)
        print(f"[fit] q_ref (q={self.normalizer.reg_quantile}): {self.normalizer.q_ref}")

        # Transform all files
        parts = []
        skipped = 0
        for f in files:
            out = self.normalizer.transform_file(self.loader, f)
            if out is None:
                skipped += 1
                continue
            parts.append(out)

        if not parts:
            raise RuntimeError("No rows produced. Check your column candidates and raw data.")

        unified = pd.concat(parts, ignore_index=True)

        # de-dup by text (keep first)
        unified = unified.drop_duplicates(subset=["text"], keep="first")
        print(f"[info] skipped files: {skipped}, total rows after dedup: {len(unified)}")

        return unified

    def save(self, df: pd.DataFrame) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.out_path, index=False)
        print("Saved:", self.out_path)


def main(config_path="configs/default.yaml"):
    cfg = load_config(config_path)

    raw_dir = Path(cfg["data"]["raw_dir"])
    out_path = Path(cfg["data"]["processed_path"])

    text_candidates = cfg["data"]["text_col_candidates"]
    cls_candidates = cfg["data"]["cls_label_candidates"]
    reg_candidates = cfg["data"]["reg_label_candidates"]

    labels_cfg = cfg.get("labels", {})
    normalizer = Normalizer(
        text_candidates=text_candidates,
        cls_candidates=cls_candidates,
        reg_candidates=reg_candidates,
        reg_min=float(labels_cfg.get("reg_min", 0.0)),
        reg_max=float(labels_cfg.get("reg_max", 5.0)),
        cls_from_reg_threshold=float(labels_cfg.get("cls_from_reg_threshold", 3.0)),
        reg_quantile=float(labels_cfg.get("reg_quantile", 0.95)),
    )

    loader = DataLoader(raw_dir=raw_dir)
    builder = DatasetBuilder(loader=loader, normalizer=normalizer, out_path=out_path)

    df = builder.build()
    builder.save(df)
    print(df.head())


if __name__ == "__main__":
    main()
