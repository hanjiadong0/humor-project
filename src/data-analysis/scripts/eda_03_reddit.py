import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "reddit_full_data.csv"
OUT_DIR = "outputs/reddit"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# 选文本字段：fulltext 优先；没有就 selftext+title
if "fulltext" in df.columns:
    text = df["fulltext"].fillna("")
else:
    text = df.get("selftext", "").fillna("") + " " + df.get("title", "").fillna("")

df["text"] = text.astype(str).str.replace(r"\s+"," ",regex=True).str.strip()

def tok(s):
    return len(re.findall(r"\w+|[^\w\s]", s))

df["tok_len"] = df["text"].map(tok)

# over_18 比例（如果有）
if "over_18" in df.columns:
    over18_rate = float(df["over_18"].mean())
    with open(os.path.join(OUT_DIR, "over18_rate.txt"), "w", encoding="utf-8") as f:
        f.write(str(over18_rate))

# 长度分布
plt.figure()
plt.hist(df["tok_len"], bins=50)
plt.title("Reddit token length distribution")
plt.xlabel("tok_len"); plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toklen_hist.png"), dpi=160)
plt.close()

# score 长尾（如果有）
if "score" in df.columns:
    s = df["score"].fillna(0)
    plt.figure()
    plt.hist(np.log1p(s), bins=50)
    plt.title("Reddit score distribution (log1p)")
    plt.xlabel("log1p(score)"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "score_log_hist.png"), dpi=160)
    plt.close()

print("DONE. Outputs at:", OUT_DIR)
