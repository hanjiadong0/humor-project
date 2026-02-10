

## 1) `hahackathon_train.csv` (Humor Scoring / Evaluator)

### 1. Goal

We aim to train a **humor evaluator / scorer** that predicts a humor score (originally intended as 0–5). Before model training, we performed exploratory data analysis (EDA) on `hahackathon_train.csv` to validate label availability, distribution, and potential biases, and to produce clean training-ready datasets.

------

### 2. Dataset Overview

`hahackathon_train.csv` contains **8,000** rows with the following relevant columns:

- `text` (input)
- `is_humor` (binary label)
- `humor_rating` (continuous humor score)
- `humor_controversy` (controversy label)
- `offense_rating` (offense label)

A key finding is that label availability is **not missing at random**:

- **Trainable for humor scoring (complete labels):** **4,932** rows
- **Missing `humor_rating` and `humor_controversy`:** **3,068** rows (**38.35%**)
- Missingness is systematic: the missing subset has `is_humor=0` for all rows, while the rated subset has `is_humor=1` for all rows.

This strongly suggests that the file is a **composition of two subsets**:

1. a **rated humor subset** (`is_humor=1`) with `humor_rating` and `humor_controversy`, and
2. a **non-humor negative subset** (`is_humor=0`) without humor ratings.

------

### 3. Humor Rating Distribution (Rated Subset)

We analyzed humor ratings on the **trainable subset (n=4,932)**.

- `humor_rating` range: **0.1 – 4.0**
- mean: **2.26**, std: **0.57**
- Rounded distribution (0–4):
  - 0: 16
  - 1: 410
  - 2: 2,835
  - 3: 1,624
  - 4: 47
  - (no 5s)

This distribution is highly concentrated around **2–3**, with very few high-score examples (4). As a result:

- pure regression is likely to collapse toward the mean,
- and multi-class classification would be heavily imbalanced.

We therefore recommend using:

- a **bucketed classification baseline** (e.g., low/mid/high), and/or
- regression with **re-weighting / re-sampling**.

------

### 4. Relationships with Offense and Controversy

On the trainable subset (n=4,932), we measured correlations:

- Pearson correlation between humor and offense: **-0.309**
   → higher humor scores tend to coincide with **lower offense** in this dataset.
- Pearson correlation between humor and controversy: **+0.174**
   → controversy has a weak positive association with humor score.

We also verified that the “high humor & low offense” region is extremely small:

- (`humor ≥ 3.5` and `offense ≤ 1.5`) → **45 samples**, i.e. **0.91%** of the rated subset.

This indicates that supervised signal for “safe but highly humorous” outputs is scarce; achieving this reliably may require additional data or targeted strategies.

------

### 5. Label Consistency Checks

We performed simple consistency checks between `is_humor` and `humor_rating` (on the rated subset):

- No cases of `is_humor=0` with high humor score (`humor ≥ 3.5`): **0**
- Cases of `is_humor=1` with very low humor score (`humor ≤ 0.8`): **51**

The second case likely reflects “attempted jokes that are not funny” and can be retained as informative negative signal within the humor subset.

------

### 6. Training-Ready Data Exports

Based on the dataset structure, we export three datasets:

1. **`train_detector.csv`** (**8,000 rows**)
   - Purpose: humor detection (binary)
   - Task: predict `is_humor` from `text_norm`
2. **`train_scorer.csv`** (**4,932 rows**)
   - Purpose: humor scoring on humor-only subset
   - Task: predict `humor_rating` (regression) or `humor_bucket3` (classification)
3. **`train_multitask.csv`** (**8,000 rows**)
   - Purpose: multi-head training
   - Labels: `is_humor`, `offense_rating` always available; `humor_rating` and `humor_controversy` only when `has_humor_score=True`
   - Important: **mask the loss** for humor rating/controversy when `has_humor_score=False`


## 2) JSONL Datasets for RAG and SFT

### JSONL format

Both JSONL datasets follow a ChatML-style schema:

```json
{
  "messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
  "word1": "...",
  "word2": "..."
}
```

Validation:

- Parsing errors: **0**
- Missing required fields: **0**
- Word constraint compliance: **100%** (assistant output contains both `word1` and `word2`)

------

### Dataset A: `humor_RAG_data_20000_RAG.jsonl` (RAG corpus)

**Before dedup**

- Rows: **20,000**
- Answer duplication rate: **0.172**
- Pair duplication rate (prompt+answer): **0.0**

**After dedup by assistant answer**

- Output: `humor_RAG_data_20000_RAG_dedupByAnswer.jsonl`
- Rows: **16,557** (kept **82.78%**, dropped 3,443)
- Answer duplication rate: **0.0**
- Word constraint pass rate: **1.0**
- Median / P90 answer length (chars): **114 / 142**

**Recommended usage**

- Use the **deduplicated** file as the primary **RAG retrieval corpus** to maximize diversity.

------

### Dataset B: `humor_training_data_5000_Train.jsonl` (SFT training)

**Before dedup**

- Rows: **5,000**
- Answer duplication rate: **0.375**
- Pair duplication rate (prompt+answer): **0.0**

**After dedup by assistant answer**

- Output: `humor_training_data_5000_Train_dedupByAnswer.jsonl`
- Rows: **3,121** (kept **62.42%**, dropped 1,879)
- Answer duplication rate: **0.0**
- Word constraint pass rate: **1.0**
- Median / P90 answer length (chars): **110 / 148**

**Recommended usage**

- Use the **deduplicated** file for **SFT fine-tuning** to reduce repetition and improve generalization.

------

### Scripts

- `scripts/analyze_jsonl_humor.py`: format validation + constraint pass rate + duplication and length stats
- `scripts/dedup_jsonl_by_answer.py`: deduplicate by normalized assistant answer text and export clean JSONL files

------

## What’s next (baseline training)

- Humor detection (binary): train on `train_detector.csv`
- Humor scoring: train on `train_scorer.csv` using:
  - bucketed classification (low/mid/high) **and** regression baseline (MAE + Spearman)
- Optional: multi-head model with masked loss using `train_multitask.csv