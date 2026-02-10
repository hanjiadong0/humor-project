import os, re
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "detection_dataset.csv"
OUT_DIR = "outputs/detection"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df["text"] = df["text"].astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
df["humor"] = df["humor"].astype(bool)

def tok(s):
    return len(re.findall(r"\w+|[^\w\s]", s))

df["tok_len"] = df["text"].map(tok)

# 类比例
dist = df["humor"].value_counts()
dist.to_csv(os.path.join(OUT_DIR, "class_counts.csv"), header=["count"])
(df["humor"].value_counts(normalize=True)).to_csv(os.path.join(OUT_DIR, "class_ratio.csv"), header=["ratio"])

# 长度分布对比
plt.figure()
plt.hist(df.loc[df["humor"], "tok_len"], bins=30, alpha=0.6, label="humor=True")
plt.hist(df.loc[~df["humor"], "tok_len"], bins=30, alpha=0.6, label="humor=False")
plt.legend()
plt.title("Detection dataset token length")
plt.xlabel("tok_len"); plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "toklen_hist.png"), dpi=160)
plt.close()

print("DONE. Outputs at:", OUT_DIR)
