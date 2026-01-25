import pandas as pd
from common.utils import load_config
import matplotlib.pyplot as plt

def main(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    df = pd.read_parquet(cfg["data"]["processed_path"])

    df["len"] = df["text"].str.len()
    print(df.describe(include="all"))
    print("\nMissing cls:", (df["y_cls"] == -1).mean())
    print("Missing reg:", (df["y_reg"] == -1).mean())
    print("\nSources:", df["source"].value_counts().head(10))

    # label distributions (only where available)
    print("\nCls distribution:\n", df[df.y_cls!=-1]["y_cls"].value_counts(normalize=True))
    print("\nReg distribution:\n", df[df.y_reg!=-1]["y_reg"].describe())

    df["len"].hist(bins=50)
    plt.title("Text length distribution")
    plt.show()

if __name__ == "__main__":
    main()

