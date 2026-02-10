# Mistral Checkpoint-600 Usage Guide

## Quick Start

### 1. Basic Usage in Python

```python
from mistral_evaluator import MistralHumorScorer

# Initialize (uses checkpoint-600 by default)
scorer = MistralHumorScorer()

# Load the model
scorer.load_model()

# Score jokes
jokes = [
    "Why did the chicken cross the road? To get to the other side!",
    "This is a serious statement about politics."
]

scores = scorer.score_jokes(jokes)
print(scores)  # [2.5, 0.3] (example output, 0-4 scale)
```

### 2. Usage in main_v3.ipynb (Dual Evaluator Pipeline)

The checkpoint is already configured as the default in `mistral_evaluator.py`:

```python
# In main_v3.ipynb, the initialization is already correct:
mistral_scorer = MistralHumorScorer(
    base_model_id="mistralai/Ministral-3-8B-Base-2512",
    checkpoint_path="./checkpoints_ministral3_multitask/checkpoint-600",  # Already set as default
    num_labels=2,
    max_length=512
)
mistral_scorer.load_model()
```

### 3. Custom Checkpoint Path

If you need to use a different checkpoint:

```python
scorer = MistralHumorScorer(
    checkpoint_path="./checkpoints_ministral3_multitask/checkpoint-1000"  # Different checkpoint
)
```

## Checkpoint-600 Specifications

| Property | Value |
|----------|-------|
| **Training Step** | 600 |
| **Eval Loss** | 1.1490 (best) |
| **F1 Score** | 0.8719 |
| **R² Score** | 0.7084 (best) |
| **Composite Score** | 0.9748 (best) |
| **Output Range** | 0-4 (humor rating scale) |
| **Expected MAE** | ~0.62 |

## API Reference

### Initialization

```python
MistralHumorScorer(
    base_model_id="mistralai/Ministral-3-8B-Base-2512",  # Base model
    checkpoint_path="./checkpoints_ministral3_multitask/checkpoint-600",  # LoRA weights
    num_labels=2,           # Binary classification (humor/not humor)
    max_length=512,         # Max token length
    device=None,            # Auto-detect GPU/CPU
    batch_size=8            # Batch size for scoring
)
```

### Methods

#### `load_model()`
Loads the base model + LoRA weights. Must be called before scoring.

```python
scorer.load_model()
```

#### `score_jokes(jokes, word1="", word2="", verbose=False)`
Score a list of jokes.

**Args:**
- `jokes`: List of joke strings
- `word1`, `word2`: Optional word pair (for API compatibility, not used)
- `verbose`: Print detailed scoring info

**Returns:**
- List of scores in range [0, 4]

```python
scores = scorer.score_jokes(
    ["Why did the banana go to the doctor? It wasn't peeling well!"],
    verbose=True
)
# Output: [2.8]
```

#### `score_single_joke(joke, word1="", word2="")`
Score a single joke (convenience method).

```python
score = scorer.score_single_joke("What do you call a bear with no teeth? A gummy bear!")
# Output: 3.2
```

#### `batch_score_with_metadata(jokes, word_pairs=None, return_classifications=False)`
Score with additional metadata.

**Args:**
- `jokes`: List of jokes
- `word_pairs`: Optional list of (word1, word2) tuples
- `return_classifications`: If True, return binary humor classification

**Returns:**
- List of dicts with keys: `joke`, `score`, (`is_humor`, `word_pair`)

```python
results = scorer.batch_score_with_metadata(
    ["Joke 1", "Joke 2"],
    word_pairs=[("banana", "doctor"), ("bear", "teeth")],
    return_classifications=True
)

# Output: [
#   {"joke": "Joke 1", "score": 2.8, "is_humor": 1, "word_pair": ("banana", "doctor")},
#   {"joke": "Joke 2", "score": 3.2, "is_humor": 1, "word_pair": ("bear", "teeth")}
# ]
```

## Score Interpretation

| Score Range | Interpretation | Example |
|-------------|----------------|---------|
| 3.5 - 4.0 | Very Funny | Excellent joke, high-quality humor |
| 2.5 - 3.5 | Funny | Good joke, likely to get laughs |
| 1.5 - 2.5 | Moderately Funny | Mildly amusing, hit or miss |
| 0.5 - 1.5 | Slightly Funny | Weak joke, few laughs expected |
| 0.0 - 0.5 | Not Funny | Not humorous or serious content |

Based on training data (mean: 2.26, std: 0.57), most jokes score between 1.7-2.8.

## Performance Expectations

### Classification (Humor Detection)
- **Accuracy:** ~88.5%
- **F1 Score:** 0.8719
- **Use case:** Filtering humor from non-humor content

### Regression (Humor Rating)
- **R²:** 0.7084 (explains 71% of variance)
- **MAE:** ~0.62 (average error ±0.62 points)
- **RMSE:** ~0.89
- **Use case:** Scoring joke quality on 0-4 scale

## Integration with main_v3.ipynb

The dual evaluator pipeline combines GPT-OSS and Mistral scores:

```python
# In score_jokes_dual() function:

# 1. Score with GPT-OSS (0-10 scale)
gpt_scores = gpt_scorer.score_jokes(jokes, word1, word2)

# 2. Score with Mistral (0-4 scale)
mistral_scores = mistral_scorer.score_jokes(jokes, word1, word2)

# 3. Normalize Mistral to 0-10 scale
mistral_scores_norm = [score * 2.5 for score in mistral_scores]

# 4. Average the two scores
final_scores = [(gpt + mistral) / 2.0 for gpt, mistral in zip(gpt_scores, mistral_scores_norm)]
```

## Troubleshooting

### Issue: "Checkpoint not found"

**Solution:** Verify checkpoint path is correct relative to script location:

```python
from pathlib import Path

checkpoint = Path("checkpoints_ministral3_multitask/checkpoint-600")
print(f"Exists: {checkpoint.exists()}")
print(f"Absolute path: {checkpoint.absolute()}")
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size or use CPU:

```python
scorer = MistralHumorScorer(
    batch_size=2,     # Smaller batch size
    device="cpu"      # Force CPU usage
)
```

### Issue: Model loading is slow

**Expected behavior:** First load takes 2-5 minutes to download/load 8B parameter model.

**Tips:**
- Model is cached after first load (faster subsequent loads)
- Use GPU for faster inference if available
- Consider using bfloat16 (already configured in code)

## File Locations

```
humor-project/
├── src/
│   └── evaluation/
│       ├── mistral_evaluator.py              # Main API
│       ├── MISTRAL_USAGE_GUIDE.md            # This file
│       ├── test_mistral_checkpoint600.py     # Test script
│       └── mistral_evaluator/
│           ├── TRAINING_ANALYSIS_REPORT.md   # Training analysis
│           ├── checkpoints_ministral3_multitask/
│           │   └── checkpoint-600/           # Production checkpoint
│           │       ├── adapter_config.json
│           │       ├── adapter_model.safetensors  # LoRA weights
│           │       └── ...
│           ├── mistral_learning_curves.png    # Training visualizations
│           └── mistral_learning_curves_early.png
```

## Testing

Run the test script to verify checkpoint loads correctly:

```bash
cd src/evaluation
python test_mistral_checkpoint600.py
```

Expected output:
```
OK: Successfully imported MistralHumorScorer
OK: MistralHumorScorer initialized successfully
OK: Model loaded successfully!

RESULTS:
1. Score: 2.85/4.0 [Funny]
   Why did the scarecrow win an award? ...
...

OK: ALL TESTS PASSED!
```

## Further Reading

- **Training Analysis:** [TRAINING_ANALYSIS_REPORT.md](mistral_evaluator/TRAINING_ANALYSIS_REPORT.md)
- **Dataset Analysis:** See Section 2 of training report
- **Learning Curves:** View PNG files in `mistral_evaluator/` directory
- **main_v3.ipynb:** See cells for dual evaluator integration

---

**Last Updated:** 2026-02-09
**Checkpoint Version:** checkpoint-600 (step 600, optimal)
**Model:** Ministral-3-8B-Base-2512 + LoRA fine-tuning
