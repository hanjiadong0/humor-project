import os, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "hahackathon_train.csv"
OUT_DIR = "outputs/hahackathon"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ---- columns ----
COLS = ["id","text","is_humor","humor_rating","humor_controversy","offense_rating"]
missing = [c for c in COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing cols {missing}. Actual: {list(df.columns)}")

d = df.copy()
d["text"] = d["text"].astype(str)
d["text_norm"] = d["text"].str.replace(r"\s+", " ", regex=True).str.strip()
for c in ["is_humor","humor_rating","humor_controversy","offense_rating"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

def approx_token_count(s: str) -> int:
    parts = re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)
    return len(parts)

d["tok_len"] = d["text_norm"].map(approx_token_count)
d["char_len"] = d["text_norm"].str.len()

# ==========================================================
# Step1: missing analysis
# ==========================================================
label_cols = ["is_humor","humor_rating","humor_controversy","offense_rating"]

miss_rate = d[label_cols].isna().mean().sort_values(ascending=False)
miss_rate.to_csv(os.path.join(OUT_DIR, "missing_rate_per_col.csv"), header=["missing_rate"])

pattern = d[label_cols].isna().astype(int)
pattern["pattern"] = pattern.apply(lambda r: "".join(r.astype(str)), axis=1)
pattern_counts = pattern["pattern"].value_counts()
pattern_counts.head(30).to_csv(os.path.join(OUT_DIR, "missing_patterns_top30.csv"))

tmp = d[["id"] + label_cols].copy()
tmp["missing_any"] = tmp[label_cols].isna().any(axis=1)
bins = pd.qcut(tmp["id"], 10, duplicates="drop")
by_bin = tmp.groupby(bins)["missing_any"].mean()
by_bin.to_csv(os.path.join(OUT_DIR, "missing_any_by_id_decile.csv"), header=["missing_any_rate"])

# ==========================================================
# Step2: define trainable dv
# ==========================================================
dv = d[d["text_norm"].str.len() > 0].copy()
dv["missing_any_label"] = dv[label_cols].isna().any(axis=1)
dv = dv[~dv["missing_any_label"]].copy()

dv["humor_clip"] = dv["humor_rating"].clip(0, 5)
dv["offense_clip"] = dv["offense_rating"].clip(0, 5)
dv["humor_round"] = dv["humor_clip"].round().astype(int).clip(0, 5)
dv["offense_round"] = dv["offense_clip"].round().astype(int).clip(0, 5)

dv.to_csv(os.path.join(OUT_DIR, "dv_clean.csv"), index=False)

# ==========================================================
# Step3: score distribution
# ==========================================================
counts = dv["humor_round"].value_counts().sort_index()
counts.to_csv(os.path.join(OUT_DIR, "humor_round_counts.csv"), header=["count"])

plt.figure()
plt.hist(dv["humor_clip"], bins=30)
plt.title("humor_rating distribution (clipped 0-5)")
plt.xlabel("humor_rating"); plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "humor_hist.png"), dpi=160)
plt.close()

plt.figure()
plt.bar(counts.index.astype(str), counts.values)
plt.title("humor_rating counts (rounded)")
plt.xlabel("score"); plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "humor_round_bar.png"), dpi=160)
plt.close()

# ==========================================================
# Step4: label conflicts
# ==========================================================
conflict_A = dv[(dv["is_humor"] == 0) & (dv["humor_clip"] >= 3.5)].copy()
conflict_B = dv[(dv["is_humor"] == 1) & (dv["humor_clip"] <= 0.8)].copy()

conflict_A.to_csv(os.path.join(OUT_DIR, "conflict_is0_highhumor.csv"), index=False)
conflict_B.to_csv(os.path.join(OUT_DIR, "conflict_is1_lowhumor.csv"), index=False)

# ==========================================================
# Step5: correlations + heatmap
# ==========================================================
corr_hum_off = float(np.corrcoef(dv["humor_clip"], dv["offense_clip"])[0,1])
corr_hum_con = float(np.corrcoef(dv["humor_clip"], dv["humor_controversy"])[0,1])

with open(os.path.join(OUT_DIR, "corr.json"), "w", encoding="utf-8") as f:
    json.dump({"corr_humor_offense": corr_hum_off, "corr_humor_controversy": corr_hum_con}, f, indent=2)

cnt = np.zeros((6,6), dtype=int)
for h, o in zip(dv["humor_round"].values, dv["offense_round"].values):
    cnt[int(h), int(o)] += 1

plt.figure()
plt.imshow(cnt, aspect="auto")
plt.colorbar(label="count")
plt.xticks(range(6), [str(i) for i in range(6)])
plt.yticks(range(6), [str(i) for i in range(6)])
plt.xlabel("offense (rounded)")
plt.ylabel("humor (rounded)")
plt.title("Count heatmap: humor bucket vs offense bucket")
for i in range(6):
    for j in range(6):
        if cnt[i,j] > 0:
            plt.text(j, i, str(cnt[i,j]), ha="center", va="center", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "humor_vs_offense_heat.png"), dpi=160)
plt.close()

# ==========================================================
# Step6: high humor low offense subsets (fixed thresholds)
# ==========================================================
subset_round = dv[(dv["humor_round"] >= 4) & (dv["offense_round"] <= 1)].copy()
subset_round.to_csv(os.path.join(OUT_DIR, "highHumor_lowOffense_round.csv"), index=False)

subset_cont = dv[(dv["humor_clip"] >= 3.5) & (dv["offense_clip"] <= 1.5)].copy()
subset_cont.to_csv(os.path.join(OUT_DIR, "highHumor_lowOffense_cont.csv"), index=False)

# ==========================================================
# Step7: 3-bucket label
# ==========================================================
def to_bucket3(x):
    if x <= 1: return "low"
    if x == 2: return "mid"
    return "high"

dv["humor_bucket3"] = dv["humor_round"].map(to_bucket3)
dv["humor_bucket3"].value_counts().to_csv(os.path.join(OUT_DIR, "humor_bucket3_counts.csv"), header=["count"])
dv.to_csv(os.path.join(OUT_DIR, "dv_with_buckets.csv"), index=False)

# ==========================================================
# A) use the 3068 "missing rating" rows (still useful)
# ==========================================================
has_score = d["humor_rating"].notna() & d["humor_controversy"].notna()
no_score = ~has_score

aux_stats = {
    "n_total": int(len(d)),
    "n_has_score": int(has_score.sum()),
    "n_no_score": int(no_score.sum()),
    "is_humor_rate_has_score": float(d.loc[has_score, "is_humor"].mean()),
    "is_humor_rate_no_score": float(d.loc[no_score, "is_humor"].mean()),
    "offense_mean_has_score": float(d.loc[has_score, "offense_rating"].mean()),
    "offense_mean_no_score": float(d.loc[no_score, "offense_rating"].mean()),
}
with open(os.path.join(OUT_DIR, "aux_subset_stats.json"), "w", encoding="utf-8") as f:
    json.dump(aux_stats, f, indent=2, ensure_ascii=False)

d.loc[no_score, ["id","text_norm","is_humor","offense_rating"]].to_csv(
    os.path.join(OUT_DIR, "aux_rows_missing_humor_rating.csv"), index=False
)

# ==========================================================
# B) sample examples by score bucket (for qualitative sanity)
# ==========================================================
examples = []
for s in sorted(dv["humor_round"].unique()):
    n = min(10, (dv["humor_round"]==s).sum())
    samp = dv[dv["humor_round"]==s].sample(n=n, random_state=0)
    examples.append(samp[["id","text_norm","is_humor","humor_rating","humor_controversy","offense_rating","humor_round"]])

examples_df = pd.concat(examples, ignore_index=True)
examples_df.to_csv(os.path.join(OUT_DIR, "examples_by_score.csv"), index=False)

# ==========================================================
# summary for report
# ==========================================================
summary = {
    "n_total": int(len(d)),
    "n_trainable": int(len(dv)),
    "missing_rate_per_col": miss_rate.to_dict(),
    "humor_min": float(dv["humor_clip"].min()),
    "humor_max": float(dv["humor_clip"].max()),
    "humor_mean": float(dv["humor_clip"].mean()),
    "humor_std": float(dv["humor_clip"].std()),
    "corr_humor_offense": corr_hum_off,
    "corr_humor_controversy": corr_hum_con,
    "highHumor_lowOffense_count": int(len(subset_cont)),
    "highHumor_lowOffense_rate": float(len(subset_cont)/len(dv)),
    "conflict_is1_lowhumor": int(len(conflict_B)),
    "conflict_is0_highhumor": int(len(conflict_A)),
}

with open(os.path.join(OUT_DIR, "summary_report.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("DONE. Outputs at:", OUT_DIR)


df = pd.read_csv("hahackathon_train.csv")
for c in ["is_humor","humor_rating","humor_controversy","offense_rating"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

no_score = df[df["humor_rating"].isna() & df["humor_controversy"].isna()].copy()
no_score["text_norm"] = no_score["text"].astype(str).str.replace(r"\s+"," ",regex=True).str.strip()

sample = no_score.sample(n=min(50, len(no_score)), random_state=0)
sample[["id","text_norm","is_humor","offense_rating"]].to_csv("outputs/hahackathon/no_score_sample50.csv", index=False)
print("saved outputs/hahackathon/no_score_sample50.csv")