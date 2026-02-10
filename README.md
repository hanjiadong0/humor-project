# Humor Generation and Evaluation Pipeline

A multi-model humor generation system that combines fine-tuned language models, retrieval-augmented generation (RAG), and a dual-evaluator ensemble to automatically generate and score jokes from word pairs or news headlines.

## Abstract

This project implements an end-to-end pipeline for computational humor generation and evaluation. The system generates jokes using two parallel generators -- a fine-tuned **Llama-3-8B** model with RAG retrieval and a **GPT-OSS-120B** model -- then scores them using a fine-tuned **Ministral-3-8B** evaluator as the primary ranking signal. The Mistral evaluator is trained as a multi-task model with both classification (humor/not-humor) and regression (humor rating 0--4) heads on the HaHackathon dataset. An LLM-based multi-criteria scorer (currently GPT-OSS) was tested experimentally but exhibits self-bias when the same model family generates and evaluates jokes. For the final dataset, only the fine-tuned Mistral scorer is used for joke selection. The LLM evaluator is designed with a swappable backend so different APIs can be tested in the future. A LangGraph-based orchestrator manages the pipeline with iterative refinement and source attribution. The system supports two input modes: word-pair jokes (connect two unrelated words) and headline commentary (funny response to a news headline).

---

## Architecture Overview

```
                              INPUT
                     (word1 + word2) or (headline)
                               |
                    +----------+----------+
                    |                     |
             [Llama-3-8B]          [GPT-OSS-120B]
             + LoRA SFT            Temperature 0.9
             + RAG (FAISS)         2 jokes
             + 3-Stage Pipeline
             6 jokes (3 RAG + 3 no-RAG)
                    |                     |
                    +----------+----------+
                               |
                        [Combine Jokes]
                     8 candidates + source tags
                               |
                    +----------+----------+
                    |                     |
             [LLM Scorer]          [Mistral Scorer]        <-- PRIMARY
             Multi-criteria 1-10   cls_head gate +
             (experimental,        reg_head 0-4
              reference only)
                                   normalized to 0-10
                    |                     |
                    +----------+----------+
                               |
                    [Rank by Mistral Normalized Score]
                    Ensemble (LLM 60% + Mistral 40%) as reference
                    LLM self-bias discount 0.80
                               |
                    [Mistral Score >= 7.0?]
                      /              \
                   YES                NO (& iterations < max)
                    |                  |
               [Output]          [Refine & Retry]
          Best + 2nd best
          CSV + JSON export
```

**Note:** In headline mode (v4), LLM evaluation is skipped entirely. Only the fine-tuned Mistral scorer runs, which is sufficient for the final dataset.

---

## Project Structure

```
humor-project/
|-- configs/
|   +-- default.yaml                  # Training hyperparameters (Mistral fine-tuning)
|-- data/
|   |-- labeled_data/
|   |   +-- hahackathon_train.csv     # 8,000 humor-scored samples (training)
|   |-- humor_RAG_data_20000_RAG.jsonl         # RAG corpus (16,557 after dedup)
|   |-- humor_training_data_5000_Train.jsonl   # SFT training data (3,121 after dedup)
|   |-- task-a-en.tsv                          # 299 news headlines (headline mode input)
|   |-- 2word_output/                          # Word-pair generation output CSVs
|   +-- headline_output/                       # Headline generation output CSVs
|-- src/
|   |-- common/
|   |   |-- path.py                   # Project root resolution
|   |   +-- utils.py                  # Config & env loading
|   |-- data-analysis/
|   |   |-- README.md                 # Dataset documentation & EDA findings
|   |   |-- scripts/
|   |   |   |-- eda_01_hahackathon.py # HaHackathon EDA
|   |   |   |-- eda_02_detection.py   # Detection dataset EDA
|   |   |   +-- eda_03_reddit.py      # Reddit dataset EDA
|   |   +-- eda_detection.ipynb       # Interactive EDA notebook
|   |-- evaluation/
|   |   |-- mistral_evaluator.py      # MistralHumorScorer (fine-tuned, cls+reg) -- PRIMARY
|   |   |-- llm_evaluator.py          # LLMHumorScorer (API-based, swappable backend)
|   |   |-- headline_evaluator.py     # Headline EDA & semantic similarity
|   |   |-- GPT_joke_evaluator.ipynb  # Interactive LLM evaluation notebook
|   |   +-- mistral_model/
|   |       |-- finetune_mistral3.ipynb                    # Mistral fine-tuning notebook
|   |       |-- plot_mistral_learning_curve.py             # Training visualization
|   |       |-- MISTRAL_USAGE_GUIDE.md                     # API reference
|   |       |-- TRAINING_ANALYSIS_REPORT.md                # Training report
|   |       +-- checkpoints_ministral3_multitask/
|   |           +-- checkpoint-600/                        # Best checkpoint (selected)
|   |-- generation/
|   |   |-- patterns.py               # 15 humor pattern templates
|   |   |-- prompt_pipeline.py        # 3-stage prompt pipeline
|   |   |-- gpt_joke_generation.py    # GPT-OSS-120B generator
|   |   |-- using_llama_with_rag.py   # Llama-3 + RAG generator (primary)
|   |   |-- main_v3_2words.ipynb      # Word-pair pipeline (LangGraph)
|   |   |-- main_v4_headline.ipynb    # Headline pipeline (LangGraph)
|   |   +-- llama3_humor-neu/
|   |       |-- training_v2.ipynb              # Llama fine-tuning notebook
|   |       |-- plot_llama_learning_curve.py   # Training visualization
|   |       |-- llama3_learning_curves.png     # Learning curve plot
|   |       +-- llama3-humor-lora/
|   |           +-- checkpoint-2300/           # Llama LoRA weights (best)
|   +-- humor_classifier/
|       +-- finetune_MIstral3.ipynb   # Classifier training (experimental)
|-- .env                              # API keys (not committed)
|-- requirements.txt                  # Python dependencies
+-- README.md                         # This file
```

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU with >= 16 GB VRAM (tested on NVIDIA RTX 5090 Laptop GPU, 24 GB)
- Hugging Face account (for model downloads)
- NVIDIA API key (for GPT-OSS generation and evaluation)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd humor-project

# Install dependencies
pip install torch transformers>=4.45.0 datasets accelerate peft bitsandbytes trl
pip install scikit-learn pandas numpy pyyaml matplotlib
pip install sentence-transformers faiss-cpu langgraph
pip install python-dotenv
```

### Environment Variables

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
HF_TOKEN=your_huggingface_token_here
WANDB_DISABLED=true
```

- **NVIDIA_API_KEY**: Required for GPT-OSS generation and LLM evaluation (via NVIDIA NIM API)
- **HF_TOKEN**: Required for downloading Llama-3 and Ministral-3 base models from Hugging Face
- **WANDB_DISABLED**: Set to `true` to disable Weights & Biases logging

---

## Path Configuration

The notebooks find `PROJECT_ROOT` by walking up from the current directory until they locate the `src/` folder:

```python
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
```

### Paths That Must Be Updated

If you clone this repository to a different machine, update the following **hardcoded paths**:

| File | Variable | Default Value |
|------|----------|---------------|
| `src/generation/using_llama_with_rag.py` | `RAGConfig.LORA_PATH` | `C:\Users\Anwender\humor-project\src\generation\llama3_humor-neu\llama3-humor-lora\checkpoint-2300` |
| `src/generation/using_llama_with_rag.py` | `RAGConfig.JOKES_DATASET` | `C:\Users\Anwender\humor-project\data\humor_RAG_data_20000_RAG.jsonl` |

The Mistral checkpoint path is auto-detected relative to the project structure (`src/evaluation/mistral_model/checkpoints_ministral3_multitask/checkpoint-600`).

All other paths (data directories, output directories, CSV paths) are resolved dynamically relative to `PROJECT_ROOT`.

### Checkpoint Path Note

When specifying the Mistral checkpoint, you must include the **specific checkpoint folder**, not just the parent directory:

```python
# Correct:
checkpoint_path = ".../checkpoints_ministral3_multitask/checkpoint-600"

# Wrong (will fail to load head weights):
checkpoint_path = ".../checkpoints_ministral3_multitask"
```

---

## API Configuration

### NVIDIA NIM API

The GPT-OSS generator and LLM scorer use the **NVIDIA NIM API** (OpenAI-compatible endpoint). The LLM scorer backend is swappable -- any OpenAI-compatible API can be used:

| Component | Model ID | Base URL | Temperature |
|-----------|----------|----------|-------------|
| GPT Generator | `openai/gpt-oss-120b` | `https://integrate.api.nvidia.com/v1` | 0.9 |
| LLM Scorer | `openai/gpt-oss-120b` | `https://integrate.api.nvidia.com/v1` | 0.3 |

Both components use the `openai` Python client with a custom `base_url`:

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)
```

### Hugging Face

Model downloads (Llama-3, Ministral-3) require a Hugging Face token with access to gated models. Set `HF_TOKEN` in `.env` or run:

```bash
huggingface-cli login
```

---

## Datasets

### Training Datasets

| Dataset | Rows | Purpose | Key Statistics |
|---------|------|---------|----------------|
| `hahackathon_train.csv` | 8,000 | Mistral evaluator training | `is_humor` (binary), `humor_rating` (0.1--4.0, mean 2.26, std 0.57). 4,932 rows with ratings (`is_humor=1`), 3,068 negative samples (`is_humor=0`) |
| `humor_RAG_data_20000_RAG.jsonl` | 16,557 (deduped) | RAG retrieval corpus | ChatML format, median answer length 114 chars |
| `humor_training_data_5000_Train.jsonl` | 3,121 (deduped) | Llama-3 SFT fine-tuning | ChatML format, 100% word constraint compliance |

### Input Datasets

| Dataset | Format | Content |
|---------|--------|---------|
| Word pairs | Hardcoded in notebook | 19 word pairs (e.g., `("banana", "satellite")`) |
| `task-a-en.tsv` | TSV | 299 news headlines for headline-mode joke generation |

For detailed EDA findings, see [`src/data-analysis/README.md`](src/data-analysis/README.md).

---

## Components

### 1. Joke Generation

#### Llama-3-8B + LoRA + RAG (Primary Generator)

**File:** [`src/generation/using_llama_with_rag.py`](src/generation/using_llama_with_rag.py)

- **Base model:** `meta-llama/Meta-Llama-3-8B-Instruct`
- **Fine-tuning:** LoRA adapter trained on 3,121 deduplicated joke samples (SFT)
- **RAG retrieval:** FAISS index over 16,557 jokes using `all-MiniLM-L6-v2` embeddings
- **Pipeline:** 3-stage prompt pipeline (Associations -> Imagery -> Final Joke)
- **Output:** 6 jokes per input (3 with RAG, 3 without RAG)
- **Retrieval:** MMR diversity sampling (top_k=3, candidate_k=12, lambda=0.4)

#### GPT-OSS-120B (Secondary Generator)

**File:** [`src/generation/gpt_joke_generation.py`](src/generation/gpt_joke_generation.py)

- **Model:** `nvidia/llama-3.3-nemotron-super-49b-v1` via NVIDIA NIM API
- **Temperature:** 0.9 (high creativity)
- **Output:** 2 jokes per input (reduced from 5 to limit GPT self-bias in evaluation)

#### 3-Stage Prompt Pipeline

**File:** [`src/generation/prompt_pipeline.py`](src/generation/prompt_pipeline.py)

1. **Step 1 -- Associations:** Generate 10 diverse associations from input (objects, emotions, places, actions)
2. **Step 2 -- Imagery:** Convert associations into vivid mental images
3. **Step 3 -- Final Joke:** Generate joke using a randomly selected humor pattern with PRINCIPLE -> SURPRISE framework

#### 15 Humor Patterns

**File:** [`src/generation/patterns.py`](src/generation/patterns.py)

Each pattern defines a principle (setup) and surprise (twist) mechanism:

| # | Pattern | Description |
|---|---------|-------------|
| 1 | Pun (re-interpretation) | Double meaning / wordplay |
| 2 | Absurd Logic | Serious reasoning in a weird world |
| 3 | Mini Story | Setup -> turn -> punchline |
| 4 | Sarcasm | Meaning inversion |
| 5 | Mock Breaking News | Formal frame, silly content |
| 6 | Deadpan | Flat delivery of absurdity |
| 7 | Irony | Expectation vs. outcome |
| 8 | Exaggeration / Heightening | Bigger, bigger, biggest |
| 9 | Observational | Relatable truth -> twist |
| 10 | Fake Expert | Confident nonsense logic |
| 11 | Self-aware | Meta rule-break |
| 12 | Understatement | Tiny reaction to huge thing |
| 13 | Rule of Three | Pattern -> pattern -> twist |
| 14 | Literalism | Take figurative language literally |
| 15 | Status Reversal | Who's "above" flips |

### 2. Joke Evaluation

#### Mistral Humor Scorer (Primary -- Drives Ranking)

**File:** [`src/evaluation/mistral_evaluator.py`](src/evaluation/mistral_evaluator.py)

- **Base model:** `mistralai/Ministral-3-8B-Base-2512`
- **Architecture:** Multi-task model with two heads:
  - `cls_head`: Binary classification (humor / not-humor) -- used as a **gate filter**
  - `reg_head`: Regression (humor rating 0--4) -- used for scoring
- **Classification gate:** If `cls_head` predicts class 0 (not humor), the joke receives score 0 regardless of regression output
- **Normalization:** Piecewise linear mapping from 0--4 to 0--10 (calibrated to training distribution, see Scoring Methodology below)
- **Best checkpoint:** `checkpoint-600` (see Training Details below)
- **Role:** All joke selection, ranking, sorting, and threshold checks use the Mistral normalized score

#### LLM Multi-Criteria Scorer (Experimental -- Reference Only)

**File:** [`src/evaluation/llm_evaluator.py`](src/evaluation/llm_evaluator.py)

- **Current backend:** `nvidia/llama-3.3-nemotron-super-49b-v1` (GPT-OSS) via NVIDIA NIM API
- **Temperature:** 0.3 (low for consistency)
- **Scale:** 1--10 (multi-criteria assessment)
- **Evaluation criteria:** Creativity, word integration / headline relevance, humor impact, execution quality, coherence
- **Role:** Experimental. Ensemble score (LLM 60% + Mistral 40%) is computed and logged as reference but does **not** drive joke selection
- **Limitation:** When the same LLM family is used for both generation and evaluation, it tends to prefer its own output (self-bias). In our experiments, the GPT-OSS evaluator consistently rated GPT-generated jokes higher than Llama-generated jokes, regardless of actual quality. This is why the fine-tuned Mistral scorer is used as the primary ranking signal for the final dataset.
- **Headline mode:** LLM evaluation is skipped entirely for faster inference
- **Future work:** The evaluator interface is designed to be swappable -- different LLM APIs can be tested to find a less biased external scorer

### 3. Pipeline Orchestration

**Files:** [`src/generation/main_v3_2words.ipynb`](src/generation/main_v3_2words.ipynb) (word-pair mode), [`src/generation/main_v4_headline.ipynb`](src/generation/main_v4_headline.ipynb) (headline mode)

- **Framework:** LangGraph (stateful workflow orchestration)
- **Parallel generation:** Llama and GPT run concurrently
- **Iterative refinement:** If best Mistral score < 7.0 and iterations remain, regenerate and re-score
- **Source tracking:** Each joke tagged with generator source (`Llama` or `GPT`)
- **Incremental CSV output:** Results flushed after each input to preserve progress on interruption

---

## Model Training Details

### Llama-3-8B SFT Fine-Tuning

**Notebook:** [`src/generation/llama3_humor-neu/training_v2.ipynb`](src/generation/llama3_humor-neu/training_v2.ipynb)
**Learning curves:** [`src/generation/llama3_humor-neu/llama3_learning_curves.png`](src/generation/llama3_humor-neu/llama3_learning_curves.png)

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Training method | SFT (Supervised Fine-Tuning) with LoRA |
| Training data | 25,000 samples (3,121 deduplicated, ChatML format) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (7 modules) |
| Quantization | 4-bit QLoRA (BitsAndBytes) |
| Max sequence length | 256 tokens |
| Learning rate | 2e-4 |
| LR scheduler | Cosine (decays to ~0) |
| Epochs | 3 |
| Batch size | 20 |
| Precision | BFloat16 |
| Gradient checkpointing | Enabled |
| Optimizer | `paged_adamw_8bit` |
| Save steps | 100 |
| Eval steps | 50 |
| Trainable parameters | 41,943,040 (0.92% of total) |
| Total training steps | 3,564 (1,188 steps/epoch) |

#### Checkpoint Selection: Why `checkpoint-2300`?

Training progresses through three distinct phases:

1. **Epoch 1 (steps 0--1188) -- Rapid learning:** Eval loss drops steadily from 1.71 to 0.89. The model learns joke structure, humor patterns, and word-pair integration.

2. **Epoch 2 (steps 1188--2376) -- Refinement:** Eval loss continues decreasing but slower, from 0.89 to 0.69. The model refines punchline delivery and creative associations. **Best eval loss reached at step 2350 (0.6936).**

3. **Epoch 3 (steps 2376--3564) -- Overfitting:** Eval loss rises back to ~0.72 and plateaus. Train loss continues decreasing (memorization). The widening gap between train and eval loss is a classic overfitting indicator.

**`checkpoint-2300`** is the closest saved checkpoint (at `save_steps=100`) to the best evaluation point. The trainer's `load_best_model_at_end=True` with `metric_for_best_model="eval_loss"` automatically identified this region as optimal.

| Metric | Value |
|--------|-------|
| Best eval loss | **0.6936** (step 2350) |
| Checkpoint-2300 eval loss | **0.6989** |
| Epoch 3 avg eval loss | 0.7195 (+3.7% worse than best) |
| Overfitting gap (epoch 3) | Train 0.40 vs Eval 0.72 (delta 0.32) |

### Mistral-3-8B Multi-Task Fine-Tuning

**Notebook:** [`src/evaluation/mistral_model/finetune_mistral3.ipynb`](src/evaluation/mistral_model/finetune_mistral3.ipynb)
**Config:** [`configs/default.yaml`](configs/default.yaml)

#### Architecture

The Mistral evaluator uses a `MultiTaskRegCls` architecture that extends the base Ministral-3-8B model with two task-specific heads:

```
Ministral-3-8B-Base-2512 (backbone, frozen except LoRA)
    |
    +-- LoRA adapters (q_proj, k_proj, v_proj, o_proj)
    |
    [Last-token pooling]
    |
    +-- cls_head: Linear(hidden_size, 2)   --> humor / not-humor
    +-- reg_head: Linear(hidden_size, 1)   --> humor rating (0-4)
```

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `mistralai/Ministral-3-8B-Base-2512` |
| Training task | Multi-task (classification + regression) |
| Training data | 8,000 samples from HaHackathon (`hahackathon_train.csv`) |
| -- Classification | 8,000 samples (binary `is_humor`) |
| -- Regression | 4,932 samples (rated subset only, loss masked for non-humor) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Max sequence length | 256 tokens |
| Learning rate | 1e-4 |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 (effective batch size = 16) |
| Warmup ratio | 0.03 |
| Weight decay | 0.0 |
| Precision | FP16 |
| Loss weights | `cls_weight=1.0`, `reg_weight=1.0` |
| Loss masking | Regression loss masked when `has_humor_score=False` |

#### Checkpoint Selection

27 checkpoints were saved (every 200 steps, from step 200 to step 5400). **Checkpoint-600** was selected as the best based on a composite score:

| Metric | Checkpoint-600 | Notes |
|--------|---------------|-------|
| Eval Loss | **1.1490** | Best among all checkpoints |
| Regression R^2 | **0.7084** | Best among all checkpoints |
| Classification F1 | **0.8719** | Near-best |
| Composite Score | **0.9748** | Best overall (weighted combination) |

Performance degrades after step 1200+ due to overfitting on the small rated subset.

For detailed training analysis, see [`TRAINING_ANALYSIS_REPORT.md`](src/evaluation/mistral_model/TRAINING_ANALYSIS_REPORT.md).
---

## Scoring Methodology

### Mistral-Primary Ranking

Joke selection, sorting, and threshold checks are driven by the **Mistral normalized score** (0--10). The fine-tuned Mistral model is trained on human humor ratings, making it an independent judge that does not favor any particular generator.

During experimentation, we observed that using an LLM (GPT-OSS) for both generation and evaluation creates a **self-bias problem**: the LLM evaluator consistently rates its own generated jokes higher than Llama-generated jokes, regardless of actual quality. For the final dataset, only the Mistral scorer is used for joke selection.

A weighted ensemble with the LLM scorer is computed for reference in word-pair mode:

```
Ensemble_Score = 0.60 * LLM_Score + 0.40 * Mistral_Score_Normalized
```

- **Mistral (primary):** Pure humor funniness rating. Trained on human humor ratings (HaHackathon dataset). Original scale 0--4, normalized to 0--10. Drives all ranking decisions.
- **LLM scorer (experimental, reference only):** Multi-criteria quality assessment on a 1--10 scale. Currently uses GPT-OSS via NVIDIA NIM API. Logged as ensemble reference but does not influence joke selection. Subject to self-bias when the same model family generates and evaluates.
- **Headline mode:** LLM evaluation is skipped entirely; only Mistral scores are used.

### Mistral Classification Gate

Before computing the regression score, the Mistral `cls_head` acts as a binary filter:
- If `cls_head` predicts class 0 (not humor) -> score is set to **0.0**
- If `cls_head` predicts class 1 (humor) -> regression score is used

This prevents clearly non-humorous outputs from receiving inflated regression scores.

### Mistral Score Normalization (0--4 to 0--10)

A piecewise linear mapping calibrated to the training data distribution (mean 2.26, std 0.57):

| Mistral (0--4) | Normalized (0--10) | Interpretation |
|----------------|-------------------|----------------|
| 0.0 | 0.0 | No humor |
| 1.0 | 2.0 | Weak humor |
| 2.0 | 4.5 | Below average |
| 2.26 | 5.0 | Average (training mean) |
| 3.0 | 7.0 | Good humor |
| 3.5 | 8.5 | Very good |
| 4.0 | 10.0 | Exceptional |

### LLM Self-Bias Discount (Experimental)

When using an LLM for both generation and evaluation, the evaluator tends to rate its own generated jokes higher. To partially correct this in word-pair mode:

```
If joke source == "GPT":
    LLM_Score_Adjusted = LLM_Score_Raw * 0.80
Else:
    LLM_Score_Adjusted = LLM_Score_Raw
```

The discount factor of **0.80** reduces LLM evaluation scores by 20% for LLM-sourced jokes. However, this correction is only approximate -- the fundamental self-bias problem is why Mistral is used as the primary scorer for the final dataset, and why the LLM evaluator is considered experimental.

---

## How to Run

### Prerequisites

1. Ensure all dependencies are installed (see Setup)
2. Set up `.env` with API keys
3. Download base models from Hugging Face (automatic on first run with valid `HF_TOKEN`)

### Word-Pair Mode

Open and run [`src/generation/main_v3_2words.ipynb`](src/generation/main_v3_2words.ipynb) in Jupyter / VS Code:

| Cell | Action |
|------|--------|
| 1 | Imports and environment check |
| 2 | Setup `PROJECT_ROOT`, import all modules |
| 3 | Define `JokeGenerationState` (LangGraph state) |
| 4 | Define node functions (generate, score, output) |
| 5 | Build LangGraph workflow |
| 6 | **Load all models** (Llama, GPT, Mistral -- requires GPU) |
| 7 | **Build RAG database** and compile graph |
| 8 | **Run pipeline** -- generates jokes for all word pairs, saves CSVs incrementally |
| 9 | **Output summary** -- formatted results + JSON export |

Edit the `test_pairs` list in Cell 8 to use your own word pairs:
```python
test_pairs = [
    ("banana", "satellite"),   # word-pair mode
    ("angry", "teacup"),
    # ...
]
```

### Headline Mode

Open and run [`src/generation/main_v4_headline.ipynb`](src/generation/main_v4_headline.ipynb):

- Reads headlines from `data/task-a-en.tsv` automatically
- Same cell structure as word-pair mode
- Output saved to `data/headline_output/`

### Interrupting Long Runs

The pipeline saves results incrementally after each input. You can interrupt cell 8 at any time and still have partial results in the CSV files. Re-run cell 9 (output) to see a formatted summary of completed entries.

---

## Output Format

### CSV Files

Two CSV files are generated per run. Columns differ slightly between word-pair and headline modes.

**All Jokes (`all_*_jokes_<timestamp>.csv`):**

*Word-pair mode:*

| Column | Description |
|--------|-------------|
| # | Row number |
| PairIdx | Input index |
| Rank | Score rank within this input (by Mistral normalized) |
| Word1, Word2 | Input word pair |
| Joke | Generated joke text |
| Source | Generator: `Llama` or `GPT` |
| Mistral_0to4 | Raw Mistral regression score (0--4) |
| Mistral_Norm | Normalized Mistral score (0--10) -- **primary ranking** |
| LLM_Score | LLM evaluation score (1--10, after bias discount) -- experimental |
| Ensemble | Weighted ensemble (LLM 60% + Mistral 40%) -- reference |

*Headline mode (GPT evaluation skipped):*

| Column | Description |
|--------|-------------|
| # | Row number |
| HeadlineIdx | Input index |
| Rank | Score rank (by Mistral normalized) |
| Headline | Input headline text |
| Joke | Generated joke text |
| Source | Generator: `Llama` or `GPT` |
| Mistral_0to4 | Raw Mistral regression score (0--4) |
| Mistral_0to10 | Normalized Mistral score (0--10) |

**Best Jokes (`best_*_jokes_<timestamp>.csv`):**

*Word-pair mode:*

| Column | Description |
|--------|-------------|
| # | Input index |
| Word1, Word2 | Input word pair |
| Best_Joke / 2nd_Best_Joke | Top-2 jokes by Mistral score |
| Best_Source / 2nd_Source | Generator source |
| Best_Mistral_0to4 / Best_Mistral_Norm | Mistral scores for best joke |
| Best_LLM_Score / Best_Ensemble | LLM and ensemble scores (reference) |
| 2nd_Mistral_0to4 / 2nd_Mistral_Norm / 2nd_LLM_Score / 2nd_Ensemble | Scores for 2nd best |

*Headline mode:*

| Column | Description |
|--------|-------------|
| # | Input index |
| ID | Headline ID from TSV |
| Headline | Input headline text |
| Best_Joke / 2nd_Best_Joke | Top-2 jokes by Mistral score |
| Best_Source / 2nd_Source | Generator source |
| Best_Mistral_0to4 / Best_Mistral_Norm | Mistral scores for best joke |
| 2nd_Mistral_0to4 / 2nd_Mistral_Norm | Mistral scores for 2nd best |

### JSON Output

A full `jokes_for_rating_<timestamp>.json` is also exported with all jokes, scores, sources, and metadata for programmatic access.

---

## Design Decisions & Experimental Findings

### Why Mistral-Only for the Final Dataset?

During development, we experimented with using GPT-OSS as both a joke generator and a multi-criteria evaluator. We observed a consistent **LLM self-bias**: when the same model (or model family) generates and evaluates jokes, it systematically rates its own output higher. In our experiments, GPT-OSS-generated jokes received inflated scores from the GPT-OSS evaluator compared to Llama-generated jokes of similar or better quality.

To produce a fair, unbiased final dataset, we use only the fine-tuned **Mistral evaluator** for joke selection and ranking. The Mistral model is trained on human humor ratings from the HaHackathon dataset and acts as an independent judge that does not favor either generator.

The LLM evaluator (`llm_evaluator.py`) remains in the codebase as a reference scorer and for future experimentation with different LLM backends.

### Future Work

- **Swap LLM evaluator backend:** The `LLMHumorScorer` class uses the OpenAI-compatible API format, making it straightforward to test different LLM APIs (e.g., other NVIDIA NIM models, local models, or third-party APIs) for less biased multi-criteria evaluation.
- **Human evaluation comparison:** Compare Mistral scores against human ratings on the generated dataset to validate the scorer's alignment with human humor perception.
- **Cross-model evaluation:** Use a completely different model family for evaluation to minimize self-bias effects.

---

## References

- **HaHackathon Dataset:** Meaney, J. A., et al. (2021). "SemEval-2021 Task 7: HaHackathon, Detecting and Rating Humor and Offense." SemEval 2021.
- **Llama 3:** Meta AI (2024). "Llama 3: Open Foundation and Fine-Tuned Chat Models."
- **Ministral 3:** Mistral AI (2025). "Ministral-3-8B-Base-2512."
- **LoRA:** Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- **LangGraph:** LangChain (2024). LangGraph: Stateful multi-actor applications with LLMs.
- **FAISS:** Johnson, J., Douze, M., & Jegou, H. (2019). "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data.
- **Sentence-Transformers:** Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.
