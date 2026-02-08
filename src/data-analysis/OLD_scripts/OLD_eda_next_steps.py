import os, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "hahackathon_train.csv"
OUT_DIR = "eda_out_2"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# 统一列名（你这个文件就是这些）
COLS = ["id","text","is_humor","humor_rating","humor_controversy","offense_rating"]
missing = [c for c in COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing cols {missing}. Actual: {list(df.columns)}")

# 基础清洗
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

label_cols = ["is_humor","humor_rating","humor_controversy","offense_rating"]

# 1) 每列缺失率
miss_rate = d[label_cols].isna().mean().sort_values(ascending=False)
print("\n[Step1] Missing rate per label column:\n", miss_rate)

miss_rate.to_csv(os.path.join(OUT_DIR, "missing_rate_per_col.csv"), header=["missing_rate"])

# 2) 缺失模式（哪个组合最常见）
pattern = d[label_cols].isna().astype(int)
pattern["pattern"] = pattern.apply(lambda r: "".join(r.astype(str)), axis=1)  # 1=missing
pattern_counts = pattern["pattern"].value_counts()
print("\n[Step1] Missing patterns top10 (1=missing):\n", pattern_counts.head(10))

pattern_counts.head(30).to_csv(os.path.join(OUT_DIR, "missing_patterns_top30.csv"))

# 3) 缺失是否集中在某些 id 段（10等分看趋势）
tmp = d[["id"] + label_cols].copy()
tmp["missing_any"] = tmp[label_cols].isna().any(axis=1)
bins = pd.qcut(tmp["id"], 10, duplicates="drop")
by_bin = tmp.groupby(bins)["missing_any"].mean()
print("\n[Step1] missing_any rate by id decile:\n", by_bin)

by_bin.to_csv(os.path.join(OUT_DIR, "missing_any_by_id_decile.csv"), header=["missing_any_rate"])

# 可训练：文本非空 + 四个标签都不缺
dv = d[(d["text_norm"].str.len() > 0)].copy()
dv["missing_any_label"] = dv[["is_humor","humor_rating","humor_controversy","offense_rating"]].isna().any(axis=1)
dv = dv[~dv["missing_any_label"]].copy()

# 合理裁剪（不覆盖原始列）
dv["humor_clip"] = dv["humor_rating"].clip(0, 5)
dv["offense_clip"] = dv["offense_rating"].clip(0, 5)
dv["humor_round"] = dv["humor_clip"].round().astype(int).clip(0, 5)
dv["offense_round"] = dv["offense_clip"].round().astype(int).clip(0, 5)

print("\n[Step2] dv size:", len(dv), "out of", len(d))

dv.to_csv(os.path.join(OUT_DIR, "dv_clean.csv"), index=False)

# 分布统计
counts = dv["humor_round"].value_counts().sort_index()
counts.to_csv(os.path.join(OUT_DIR, "humor_round_counts.csv"), header=["count"])
print("\n[Step3] humor_round counts:\n", counts)

summary = {
    "n_total": int(len(d)),
    "n_trainable": int(len(dv)),
    "humor_min": float(dv["humor_clip"].min()),
    "humor_max": float(dv["humor_clip"].max()),
    "humor_mean": float(dv["humor_clip"].mean()),
    "humor_std": float(dv["humor_clip"].std()),
}
with open(os.path.join(OUT_DIR, "summary_basic.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# 图：直方图
plt.figure()
plt.hist(dv["humor_clip"], bins=30)
plt.title("humor_rating distribution (clipped 0-5)")
plt.xlabel("humor_rating"); plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "humor_hist.png"), dpi=160)
plt.close()

# 图：round后柱状
plt.figure()
plt.bar(counts.index.astype(str), counts.values)
plt.title("humor_rating counts (rounded)")
plt.xlabel("score"); plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "humor_round_bar.png"), dpi=160)
plt.close()

# 定义“明显冲突”的两类
# A: is_humor=0 但 humor 很高
conflict_A = dv[(dv["is_humor"] == 0) & (dv["humor_clip"] >= 3.5)].copy()
# B: is_humor=1 但 humor 很低
conflict_B = dv[(dv["is_humor"] == 1) & (dv["humor_clip"] <= 0.8)].copy()

print("\n[Step4] conflict_A:", len(conflict_A), "conflict_B:", len(conflict_B))

conflict_A[["id","text_norm","is_humor","humor_clip","offense_clip","humor_controversy"]].to_csv(
    os.path.join(OUT_DIR, "conflict_is0_highhumor.csv"), index=False
)
conflict_B[["id","text_norm","is_humor","humor_clip","offense_clip","humor_controversy"]].to_csv(
    os.path.join(OUT_DIR, "conflict_is1_lowhumor.csv"), index=False
)

# 图：散点
plt.figure()
y = dv["humor_clip"] + np.random.uniform(-0.05, 0.05, size=len(dv))
plt.scatter(dv["is_humor"], y, s=8, alpha=0.3)
plt.xticks([0,1], ["not humor","humor"])
plt.title("is_humor vs humor_rating")
plt.xlabel("is_humor"); plt.ylabel("humor_rating")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "is_humor_vs_humor.png"), dpi=160)
plt.close()


corr_hum_off = float(np.corrcoef(dv["humor_clip"], dv["offense_clip"])[0,1])
corr_hum_con = float(np.corrcoef(dv["humor_clip"], dv["humor_controversy"])[0,1])
with open(os.path.join(OUT_DIR, "corr.json"), "w", encoding="utf-8") as f:
    json.dump({"corr_humor_offense": corr_hum_off, "corr_humor_controversy": corr_hum_con}, f, indent=2)

print("\n[Step5] corr_humor_offense:", corr_hum_off)
print("[Step5] corr_humor_controversy:", corr_hum_con)

# heatmap：计数
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

# 方案1：用 round（跟 heatmap 一致）
subset_round = dv[(dv["humor_round"] >= 4) & (dv["offense_round"] <= 1)].copy()
print("\n[Step6] subset_round size:", len(subset_round), "rate:", len(subset_round)/len(dv))
subset_round[["id","text_norm","humor_clip","offense_clip","humor_controversy"]].to_csv(
    os.path.join(OUT_DIR, "highHumor_lowOffense_round.csv"), index=False
)

# 方案2：用更宽松的连续阈值
subset_cont = dv[(dv["humor_clip"] >= 3.5) & (dv["offense_clip"] <= 1.5)].copy()
print("[Step6] subset_cont size:", len(subset_cont), "rate:", len(subset_cont)/len(dv))
subset_cont[["id","text_norm","humor_clip","offense_clip","humor_controversy"]].to_csv(
    os.path.join(OUT_DIR, "highHumor_lowOffense_cont.csv"), index=False
)

# 3桶：low / mid / high（按你们数据结构设计）
# low: humor_round 0-1
# mid: humor_round 2
# high: humor_round 3-4
def to_bucket3(x):
    if x <= 1: return "low"
    if x == 2: return "mid"
    return "high"

dv["humor_bucket3"] = dv["humor_round"].map(to_bucket3)

bucket3_counts = dv["humor_bucket3"].value_counts()
bucket3_counts.to_csv(os.path.join(OUT_DIR, "humor_bucket3_counts.csv"), header=["count"])
print("\n[Step7] bucket3 counts:\n", bucket3_counts)

dv.to_csv(os.path.join(OUT_DIR, "dv_with_buckets.csv"), index=False)
