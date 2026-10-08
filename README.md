# Humor project learning snapshot

The ordinary Git package is `humor-learning-core.zip`. The two checkpoint ZIPs are separate GitHub Release assets.
`SHA256SUMS.txt` verifies all three files. The snapshot includes source, notebooks, learning reports, plots, selected generated results, and all saved scalar trainer states.
It omits environments, Git history, raw datasets, retrieval indexes, duplicate archive files, optimizer state, and unverified final exports.
One notebook credential literal was replaced with `REDACTED_CREDENTIAL`; private environment files were omitted. Original private data remains in the local safety backup.

## Checkpoint selection and limits

- Mistral checkpoint 600 has the lowest recorded validation loss (1.148967), lowest RMSE (0.891958), and highest R2 (0.708412). Its adapter includes classification and regression heads. This is a metadata-complete candidate; no inference or new evaluation was performed.
- Mistral checkpoints 1000 and 5400 win classification F1 and regression MAE respectively. Their adapters omit the trained multitask heads, so only their scalar metrics are in this public package. Their weights remain in the local safety backup.
- Llama checkpoint 2300 is the saved checkpoint named by the latest trainer state. Its recorded evaluation loss is 0.698892. The state reports a lower best metric at step 2350, but that checkpoint is absent. This package does not equate those measurements.
- The source references `mistralai/Ministral-3-8B-Base-2512`; the Mistral adapter leaves its base identifier null. Exact training revision is unresolved. Llama metadata specifies `meta-llama/Meta-Llama-3-8B-Instruct`.
- These are LoRA adapters. Base model weights, required base-model access, and the original custom Mistral multitask architecture are required. They are not standalone models. See `MODEL_SELECTION.json` for evidence and limitations.
- The archived `requirements.txt` contains its original malformed wrapper. It is a historical source file, not a verified environment installation recipe.

## Restore

1. Download the core ZIP and the two matching Release assets.
2. Compare their SHA-256 hashes with `SHA256SUMS.txt` before extraction.
3. Extract `humor-learning-core.zip` into a new empty project directory.
4. Extract `humor-mistral-checkpoint-600.zip` into `src/evaluation/mistral_model/checkpoints_ministral3_multitask/checkpoint-600`.
5. Extract `humor-llama-checkpoint-2300.zip` into `src/generation/llama3_humor-neu/llama3-humor-lora/checkpoint-2300`.

Each ZIP contains a `RETENTION_MANIFEST.json` with the retained file hashes. Each checkpoint asset includes its adapter, tokenizer/configuration files, model-card README, and scalar trainer state. Training restart optimizer/RNG/scheduler state is intentionally absent.

Full local history and private backups are not part of the public upload.
