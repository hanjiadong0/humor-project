"""
Headline Evaluator
Part 1: EDA statistics on task-a-en.tsv headlines
Part 2: Semantic similarity between headlines and generated jokes
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter

# ============================================================================
# PART 1: DATASET EDA STATISTICS
# ============================================================================

def run_eda(tsv_path: str):
    """Run exploratory data analysis on the headline dataset."""
    df = pd.read_csv(tsv_path, sep="\t")
    TEXT_COL = df.columns[-1]  # "headline"
    texts = df[TEXT_COL].astype(str)

    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    df["tokens"] = texts.apply(tokenize)
    df["num_words"] = df["tokens"].apply(len)
    df["num_chars"] = texts.apply(len)

    stats = {
        "Total documents": len(df),
        "Average words per document": df["num_words"].mean(),
        "Median words per document": df["num_words"].median(),
        "Min words": df["num_words"].min(),
        "Max words": df["num_words"].max(),
        "Average characters per document": df["num_chars"].mean(),
        "Median characters per document": df["num_chars"].median(),
    }

    short_docs = df[df["num_words"] <= 3]
    headline_docs = df[df["num_words"] > 3]

    stats.update({
        "Short inputs (<=3 words)": len(short_docs),
        "Headline-like inputs (>3 words)": len(headline_docs),
        "Avg headline length (words)": headline_docs["num_words"].mean()
    })

    all_tokens = [t for tokens in df["tokens"] for t in tokens]
    vocab = set(all_tokens)

    stats.update({
        "Vocabulary size": len(vocab),
        "Type-token ratio": len(vocab) / len(all_tokens)
    })

    top_words = Counter(all_tokens).most_common(15)

    print("\n=== DATASET STATISTICS ===\n")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nTop 15 most common words:")
    for word, count in top_words:
        print(f"  {word}: {count}")

    return df, stats


# ============================================================================
# PART 2: HEADLINE-JOKE SEMANTIC SIMILARITY
# ============================================================================

def compute_headline_similarity(
    jokes_csv_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = None,
):
    """
    Compute cosine similarity between each headline and its generated jokes.

    Measures topical relevance: jokes should be semantically related to
    the headline (moderate similarity) without being direct reformulations
    (very high similarity).

    Args:
        jokes_csv_path: Path to all_headline_jokes CSV
        model_name: Sentence transformer model for embeddings
        device: Device for encoding (None = auto)

    Returns:
        DataFrame with per-joke similarity and summary stats dict
    """
    from sentence_transformers import SentenceTransformer

    print(f"\n=== HEADLINE-JOKE SIMILARITY ANALYSIS ===\n")
    print(f"Loading jokes from: {jokes_csv_path}")

    df = pd.read_csv(jokes_csv_path)
    print(f"  {len(df)} jokes across {df['HeadlineIdx'].nunique()} headlines")

    # Load embedding model
    print(f"  Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    if device:
        model = model.to(device)

    # Get unique headlines and their jokes
    headlines = df["Headline"].values
    jokes = df["Joke"].values

    print(f"  Encoding {len(headlines)} headline-joke pairs...")
    h_embeddings = model.encode(headlines.tolist(), batch_size=64, show_progress_bar=False)
    j_embeddings = model.encode(jokes.tolist(), batch_size=64, show_progress_bar=False)

    # Normalize for cosine similarity
    h_norm = h_embeddings / np.linalg.norm(h_embeddings, axis=1, keepdims=True)
    j_norm = j_embeddings / np.linalg.norm(j_embeddings, axis=1, keepdims=True)

    # Per-joke cosine similarity with its headline
    similarities = np.sum(h_norm * j_norm, axis=1)
    df["Headline_Similarity"] = similarities

    # --- Per-headline statistics ---
    per_headline = df.groupby(["HeadlineIdx", "ID", "Headline"]).agg(
        avg_sim=("Headline_Similarity", "mean"),
        min_sim=("Headline_Similarity", "min"),
        max_sim=("Headline_Similarity", "max"),
        num_jokes=("Joke", "count"),
        avg_score=("Final_0to10", lambda x: pd.to_numeric(x, errors="coerce").mean()),
    ).reset_index()

    # --- Overall statistics ---
    overall = {
        "Mean similarity (all jokes)": similarities.mean(),
        "Std similarity": similarities.std(),
        "Min similarity": similarities.min(),
        "Max similarity": similarities.max(),
        "Median similarity": np.median(similarities),
        "Low similarity (<0.2)": (similarities < 0.2).sum(),
        "Moderate similarity (0.2-0.5)": ((similarities >= 0.2) & (similarities < 0.5)).sum(),
        "High similarity (0.5-0.8)": ((similarities >= 0.5) & (similarities < 0.8)).sum(),
        "Very high similarity (>0.8)": (similarities >= 0.8).sum(),
    }

    print("\n--- Overall Similarity Statistics ---")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Headlines with lowest average similarity (potential off-topic jokes)
    print("\n--- 5 Headlines with LOWEST avg similarity (potential off-topic) ---")
    bottom5 = per_headline.nsmallest(5, "avg_sim")
    for _, row in bottom5.iterrows():
        print(f"  [{row['ID']}] sim={row['avg_sim']:.3f}  {row['Headline'][:60]}...")

    # Headlines with highest average similarity (very on-topic)
    print("\n--- 5 Headlines with HIGHEST avg similarity ---")
    top5 = per_headline.nlargest(5, "avg_sim")
    for _, row in top5.iterrows():
        print(f"  [{row['ID']}] sim={row['avg_sim']:.3f}  {row['Headline'][:60]}...")

    return df, per_headline, overall


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Resolve project root
    PROJECT_ROOT = Path(__file__).resolve().parent
    while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
        PROJECT_ROOT = PROJECT_ROOT.parent

    tsv_path = PROJECT_ROOT / "data" / "task-a-en.tsv"

    # Part 1: EDA
    df_eda, stats = run_eda(str(tsv_path))

    # Part 2: Similarity (if headline jokes CSV exists)
    headline_output = PROJECT_ROOT / "data" / "headline_output"
    # Find most recent all_headline_jokes CSV
    csv_files = sorted(headline_output.glob("all_headline_jokes_*.csv"), reverse=True)

    if csv_files:
        latest_csv = csv_files[0]
        print(f"\nUsing latest jokes CSV: {latest_csv.name}")
        df_sim, per_headline, overall_stats = compute_headline_similarity(str(latest_csv))

        # Save similarity results
        sim_csv = headline_output / "headline_similarity.csv"
        per_headline.to_csv(sim_csv, index=False)
        print(f"\nPer-headline similarity saved to: {sim_csv}")
    else:
        print(f"\nNo headline jokes CSV found in {headline_output}")
        print("Run main_v4_headline.ipynb first to generate headline jokes.")
