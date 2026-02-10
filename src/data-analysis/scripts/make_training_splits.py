import os
import pandas as pd

CSV_PATH = "hahackathon_train.csv"
OUT_DIR = "outputs/hahackathon"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# numeric
for c in ["is_humor","humor_rating","humor_controversy","offense_rating"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["text"] = df["text"].astype(str)
df["text_norm"] = df["text"].str.replace(r"\s+"," ",regex=True).str.strip()

# ---- 1) detector set: all rows with is_humor present (here it's all)
detector = df.dropna(subset=["is_humor"]).copy()
detector = detector[["id","text_norm","is_humor","offense_rating"]]
detector.to_csv(os.path.join(OUT_DIR, "train_detector.csv"), index=False)

# ---- 2) scorer set: only rows with humor_rating + controversy present
scorer = df.dropna(subset=["humor_rating","humor_controversy","is_humor","offense_rating"]).copy()
scorer["humor_clip"] = scorer["humor_rating"].clip(0,5)
scorer["humor_round"] = scorer["humor_clip"].round().astype(int).clip(0,5)

# 3-bucket label (low/mid/high)
def to_bucket3(x):
    if x <= 1: return "low"
    if x == 2: return "mid"
    return "high"
scorer["humor_bucket3"] = scorer["humor_round"].map(to_bucket3)

scorer = scorer[["id","text_norm","is_humor","humor_rating","humor_controversy","offense_rating",
                 "humor_clip","humor_round","humor_bucket3"]]
scorer.to_csv(os.path.join(OUT_DIR, "train_scorer.csv"), index=False)

# ---- 3) multitask set: keep all rows, add mask flags for missing ratings
multi = df.copy()
multi["has_humor_score"] = (~multi["humor_rating"].isna()) & (~multi["humor_controversy"].isna())
multi = multi[["id","text_norm","is_humor","humor_rating","humor_controversy","offense_rating","has_humor_score"]]
multi.to_csv(os.path.join(OUT_DIR, "train_multitask.csv"), index=False)

# ---- small README for teammates
readme = f"""TRAINING DATA EXPORTS

1) train_detector.csv
   - Use for humor detection (binary): predict is_humor from text_norm.
   - Rows: {len(detector)}

2) train_scorer.csv
   - Use for humor scoring on humor subset only (is_humor should be 1 here).
   - Predict humor_rating (regression) or humor_bucket3 (classification).
   - Rows: {len(scorer)}

3) train_multitask.csv
   - Use for multi-head training.
   - Always train is_humor and offense_rating heads.
   - Train humor_rating / humor_controversy heads ONLY when has_humor_score==True (mask loss).
   - Rows: {len(multi)}
"""
with open(os.path.join(OUT_DIR, "README_training_exports.txt"), "w", encoding="utf-8") as f:
    f.write(readme)

print("DONE. Wrote exports to:", OUT_DIR)
