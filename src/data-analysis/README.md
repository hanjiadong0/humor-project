## EDA Summary for `hahackathon_train.csv` and Training Data Exports

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