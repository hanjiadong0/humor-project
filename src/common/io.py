"""
File I/O utilities for CSV, JSON, and JSONL.

Usage:
    from src.common.io import read_jsonl, write_jsonl, read_csv_safe
"""

import csv
import json
from typing import List, Dict, Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file (one JSON object per line)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CSV / TSV
# ---------------------------------------------------------------------------

def read_csv_safe(
    path: str,
    sep: str = ",",
    drop_na_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read a CSV/TSV file with optional NaN dropping.

    Args:
        path: File path
        sep: Delimiter ("," for CSV, "\\t" for TSV)
        drop_na_cols: Drop rows where these columns have NaN
    """
    df = pd.read_csv(path, sep=sep)
    if drop_na_cols:
        df = df.dropna(subset=drop_na_cols)
    return df


class IncrementalCSVWriter:
    """
    Context manager for writing CSV rows incrementally with flush.

    Usage:
        with IncrementalCSVWriter(path, headers) as w:
            w.writerow([...])   # flushed immediately
    """

    def __init__(self, path: str, headers: List[str]):
        self.path = path
        self.headers = headers
        self._file = None
        self._writer = None

    def __enter__(self):
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.headers)
        self._file.flush()
        return self

    def writerow(self, row: List[Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
        return False


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def read_json(path: str) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any, indent: int = 2) -> None:
    """Write data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
