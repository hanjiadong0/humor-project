"""
Score normalization utilities.

The Mistral evaluator outputs humor ratings on a 0-4 scale (trained on
HaHackathon data, mean=2.26, std=0.57). This module provides the canonical
piecewise-linear mapping to a 0-10 scale used throughout the pipeline.

Usage:
    from src.common.scoring import normalize_mistral_score
"""

from typing import List, Tuple

# Calibrated to hahackathon_train.csv distribution:
#   Range: 0.1-4.0, Mean: 2.26, SD: 0.57
#   ~90% of scores fall between 1.5 and 3.0
MISTRAL_BREAKPOINTS: List[Tuple[float, float]] = [
    (0.0,  0.0),   # no humor
    (1.0,  2.0),   # weak humor
    (2.0,  4.5),   # below average
    (2.26, 5.0),   # average (training mean)
    (3.0,  7.0),   # good humor
    (3.5,  8.5),   # very good
    (4.0, 10.0),   # exceptional
]


def normalize_mistral_score(raw: float) -> float:
    """
    Map a Mistral regression score (0-4) to a 0-10 scale
    using piecewise linear interpolation.

    Args:
        raw: Raw Mistral humor_rating output (0-4)

    Returns:
        Normalized score on 0-10 scale
    """
    raw = max(0.0, min(4.0, raw))

    for i in range(len(MISTRAL_BREAKPOINTS) - 1):
        x0, y0 = MISTRAL_BREAKPOINTS[i]
        x1, y1 = MISTRAL_BREAKPOINTS[i + 1]
        if raw <= x1:
            t = (raw - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return 10.0
