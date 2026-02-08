import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False  # 强制使用减号（ASCII）而不是Unicode负号
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# 0) 配置区：你只需要改这里
# =========================
CSV_PATH = "hahackathon_train.csv"
OUT_DIR = "eda_out"
DO_EMBEDDING = True          # 想要语义空间图就 True
USE_SBERT_IF_AVAILABLE = True  # 有 sentence-transformers 就用 SBERT，否则自动退回 TF-IDF

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 1) 读数据
# =========================
df = pd.read_csv(CSV_PATH)
print("Loaded:", df.shape)
print("Columns:", list(df.columns))

required = ["id", "text", "is_humor", "humor_rating", "humor_controversy", "offense_rating"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}. Your file columns are: {list(df.columns)}")

d = df.copy()

# =========================
# 2) 基础清洗 + 基础特征
# =========================
d["text"] = d["text"].astype(str)
d["text_norm"] = d["text"].str.replace(r"\s+", " ", regex=True).str.strip()

# numeric conversion
for c in ["is_humor", "humor_rating", "humor_controversy", "offense_rating"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

# quality flags
d["is_empty"] = d["text_norm"].str.len().eq(0)
d["missing_any_label"] = d[["is_humor", "humor_rating", "humor_controversy", "offense_rating"]].isna().any(axis=1)
d["is_dup_exact"] = d.duplicated(subset=["text_norm"], keep=False)

# rough token count (够用)
def approx_token_count(s: str) -> int:
    parts = re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)
    return len(parts)

d["char_len"] = d["text_norm"].str.len()
d["word_len"] = d["text_norm"].str.split().map(len)
d["tok_len"]  = d["text_norm"].map(approx_token_count)

d["is_too_short"] = d["tok_len"] <= 3
d["has_ctrl_char"] = d["text"].str.contains(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", regex=True)

# keep valid rows for most analysis
dv = d[~d["is_empty"] & ~d["missing_any_label"]].copy()
print("Valid rows:", dv.shape)

# clip ratings to sane ranges (不覆盖原始列，单独建 clip 列)
dv["humor_clip"] = dv["humor_rating"].clip(0, 5)
dv["offense_clip"] = dv["offense_rating"].clip(0, 5)

# rounded buckets for quick plots
dv["humor_round"] = dv["humor_clip"].round().astype(int).clip(0, 5)

# =========================
# 3) 图 1：评分分布
# =========================
plt.figure()
plt.hist(dv["humor_clip"], bins=30)
plt.title("humor_rating distribution (clipped 0-5)")
plt.xlabel("humor_rating")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "score_hist.png"), dpi=160)
plt.close()

counts = dv["humor_round"].value_counts().sort_index()
plt.figure()
plt.bar(counts.index.astype(str), counts.values)
plt.title("humor_rating counts (rounded to 0-5)")
plt.xlabel("score")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "score_counts_rounded.png"), dpi=160)
plt.close()

counts.to_csv(os.path.join(OUT_DIR, "bucket_counts.csv"), header=["count"])

# =========================
# 4) 图 2：is_humor vs humor_rating（是否一致）
# =========================
plt.figure()
# jitter for visibility
y = dv["humor_clip"] + np.random.uniform(-0.05, 0.05, size=len(dv))
plt.scatter(dv["is_humor"], y, s=8, alpha=0.35)
plt.xticks([0,1], ["not humor", "humor"])
plt.title("is_humor vs humor_rating")
plt.xlabel("is_humor")
plt.ylabel("humor_rating")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "is_humor_vs_rating.png"), dpi=160)
plt.close()

# =========================
# 5) 图 3：长度 vs 分数（是否有“长=高分”捷径）
# =========================
def scatter_len(xcol, fname, title):
    plt.figure()
    y = dv["humor_clip"] + np.random.uniform(-0.05, 0.05, size=len(dv))
    plt.scatter(dv[xcol], y, s=8, alpha=0.3)
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel("humor_rating")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=160)
    plt.close()

scatter_len("char_len", "len_char_vs_humor.png", "char_len vs humor_rating")
scatter_len("tok_len",  "len_tok_vs_humor.png",  "tok_len vs humor_rating")

# boxplot by rounded score
plt.figure()
data_by_score = [dv.loc[dv["humor_round"]==k, "tok_len"].values for k in range(0,6)]
plt.boxplot(data_by_score, labels=[str(k) for k in range(0,6)], showfliers=False)
plt.title("token length by score (rounded)")
plt.xlabel("score")
plt.ylabel("tok_len")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tok_len_by_score_box.png"), dpi=160)
plt.close()

# =========================
# 6) 图 4：humor vs offense / controversy（安全关联）
# =========================
# 2D heatmap: bin humor into 0..5, offense into 0..5 (rounded)
dv["offense_round"] = dv["offense_clip"].round().astype(int).clip(0, 5)
heat = np.zeros((6,6), dtype=float)
cnt  = np.zeros((6,6), dtype=int)

for _, r in dv.iterrows():
    h = int(r["humor_round"])
    o = int(r["offense_round"])
    heat[h,o] += r["humor_clip"]
    cnt[h,o] += 1

# show counts as color (more intuitive), and annotate count
plt.figure()
plt.imshow(cnt, aspect="auto")
plt.colorbar(label="count")
plt.xticks(range(0,6), [str(i) for i in range(0,6)])
plt.yticks(range(0,6), [str(i) for i in range(0,6)])
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

# controversy vs humor (simple scatter)
plt.figure()
y = dv["humor_clip"] + np.random.uniform(-0.05, 0.05, size=len(dv))
plt.scatter(dv["humor_controversy"], y, s=8, alpha=0.3)
plt.title("humor_controversy vs humor_rating")
plt.xlabel("humor_controversy")
plt.ylabel("humor_rating")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "controversy_vs_humor.png"), dpi=160)
plt.close()

# =========================
# 7) 关键结论数字（相关、质量、子集占比）
# =========================
def safe_corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 3:
        return None
    return float(np.corrcoef(a, b)[0,1])

summary = {}
summary["n_total"] = int(len(d))
summary["n_valid"] = int(len(dv))
summary["empty_rate"] = float(d["is_empty"].mean())
summary["missing_any_label_rate"] = float(d["missing_any_label"].mean())
summary["dup_exact_rate"] = float(d["is_dup_exact"].mean())
summary["too_short_rate"] = float(d["is_too_short"].mean())
summary["ctrl_char_rate"] = float(d["has_ctrl_char"].mean())

summary["humor_min"] = float(dv["humor_clip"].min())
summary["humor_max"] = float(dv["humor_clip"].max())
summary["humor_mean"] = float(dv["humor_clip"].mean())
summary["humor_std"] = float(dv["humor_clip"].std())

summary["corr_humor_toklen"] = safe_corr(dv["humor_clip"], dv["tok_len"])
summary["corr_humor_offense"] = safe_corr(dv["humor_clip"], dv["offense_clip"])
summary["corr_humor_controversy"] = safe_corr(dv["humor_clip"], dv["humor_controversy"])

# “高幽默低冒犯”子集：你们最想要的
high_humor_low_off = dv[(dv["humor_clip"] >= 4.0) & (dv["offense_clip"] <= 1.0)].copy()
summary["high_humor_low_offense_count"] = int(len(high_humor_low_off))
summary["high_humor_low_offense_rate"] = float(len(high_humor_low_off) / len(dv)) if len(dv) else None

with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

high_humor_low_off[["id","text_norm","humor_clip","offense_clip","humor_controversy","is_humor","tok_len","char_len"]] \
    .sort_values(["humor_clip","offense_clip"], ascending=[False, True]) \
    .to_csv(os.path.join(OUT_DIR, "high_humor_low_offense.csv"), index=False)

print("Saved summary.json and high_humor_low_offense.csv")

# =========================
# 8) 可疑样本导出（方便你人工 spot-check）
# =========================
sus = dv[
    dv["is_dup_exact"] |
    dv["is_too_short"] |
    dv["has_ctrl_char"] |
    ((dv["is_humor"] == 0) & (dv["humor_clip"] >= 3.5)) |
    ((dv["is_humor"] == 1) & (dv["humor_clip"] <= 0.8))
].copy()

sus = sus.sort_values(
    ["is_dup_exact","is_too_short","tok_len","humor_clip"],
    ascending=[False, False, True, False]
)

sus_out = sus[["id","text_norm","is_humor","humor_clip","humor_controversy","offense_clip","tok_len","char_len",
               "is_dup_exact","is_too_short","has_ctrl_char"]]
sus_out.to_csv(os.path.join(OUT_DIR, "suspicious_samples.csv"), index=False)
print("Saved suspicious_samples.csv:", len(sus_out))

# =========================
# 9) 可选：语义空间可分性（PCA 可视化）
# =========================
if DO_EMBEDDING:
    texts = dv["text_norm"].tolist()
    y = dv["humor_round"].values

    used = None
    Z = None

    if USE_SBERT_IF_AVAILABLE:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
            Z = PCA(n_components=2, random_state=0).fit_transform(emb)
            used = "SBERT(all-MiniLM-L6-v2)"
        except Exception as e:
            print("SBERT not available / failed, fallback to TF-IDF. Reason:", repr(e))

    if Z is None:
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        X = vec.fit_transform(texts)
        Z = PCA(n_components=2, random_state=0).fit_transform(X.toarray())
        used = "TF-IDF(1-2gram)+PCA"

    plt.figure()
    plt.scatter(Z[:,0], Z[:,1], c=y, s=8, alpha=0.5)
    plt.title(f"Embedding PCA colored by humor score (rounded) [{used}]")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "embedding_pca.png"), dpi=160)
    plt.close()
    print("Saved embedding_pca.png using:", used)

print("DONE. Outputs in:", OUT_DIR)
