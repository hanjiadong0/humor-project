"""
Mistral-based Humor Evaluator for main_v3.ipynb
Fine-tuned Mistral model for scoring jokes (0-4 scale)

This module provides the MistralHumorScorer class that loads a fine-tuned
Ministral-3-8B model with LoRA weights and scores jokes for funniness.
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftConfig, LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors


# ============================================================================
# UTILITY FUNCTIONS (matching training notebook)
# ============================================================================

def get_text_backbone(model):
    """
    Extract text backbone from potentially multimodal wrapper.
    Handles Mistral3Config wrapper structure.
    """
    cfg = getattr(model, "config", None)
    is_wrapper = (cfg is not None) and hasattr(cfg, "text_config") and (cfg.text_config is not None)

    if not is_wrapper:
        return model

    # Try known text module names
    for name in ["text_model", "language_model"]:
        if hasattr(model, name):
            m = getattr(model, name)
            if m is not None:
                return m

    return model


def get_hidden_size(cfg) -> int:
    """
    Extract hidden size from config (handles Mistral3Config wrapper).

    Args:
        cfg: Model config object

    Returns:
        Hidden size as integer
    """
    # Check if text_config exists (multimodal wrapper)
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        tc = cfg.text_config
    else:
        tc = cfg

    # Try as dict
    if isinstance(tc, dict):
        for k in ["hidden_size", "dim", "d_model", "model_dim"]:
            if k in tc and tc[k] is not None:
                return int(tc[k])
        raise ValueError(f"Hidden size not found in text_config dict. Keys: {list(tc.keys())}")

    # Try as object attributes
    for attr in ["hidden_size", "dim", "d_model", "model_dim"]:
        if hasattr(tc, attr) and getattr(tc, attr) is not None:
            return int(getattr(tc, attr))

    # Last resort: convert to dict
    if hasattr(tc, "to_dict"):
        d = tc.to_dict()
        for k in ["hidden_size", "dim", "d_model", "model_dim"]:
            if k in d and d[k] is not None:
                return int(d[k])

    raise ValueError("Could not find hidden size in config")


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool the last real token (not padding).

    Args:
        last_hidden_state: [B, T, H] tensor
        attention_mask: [B, T] tensor with 1 for tokens, 0 for padding

    Returns:
        [B, H] tensor
    """
    if attention_mask is None:
        return last_hidden_state[:, -1, :]

    lengths = attention_mask.long().sum(dim=1)  # [B]
    idx = torch.clamp(lengths - 1, min=0)       # [B]
    batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, idx, :]  # [B, H]


# ============================================================================
# MULTITASK MODEL (matching training architecture)
# ============================================================================

class MultiTaskRegCls(nn.Module):
    """
    Multi-task classification + regression model.
    MUST match the architecture used in finetune_mistral3.ipynb.
    """
    def __init__(self, backbone, num_labels: int, loss_w_reg: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        h = get_hidden_size(backbone.config)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(h, num_labels)
        self.reg_head = nn.Linear(h, 1)
        self.loss_w_reg = loss_w_reg
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, y_cls=None, y_reg=None, **kwargs):
        """
        Forward pass.

        Returns:
            dict with keys: logits_cls, pred_reg, (optionally loss)
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled = last_token_pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        pooled = pooled.float()  # cast to float32 for head compatibility

        logits_cls = self.cls_head(pooled)               # [B, C]
        pred_reg = self.reg_head(pooled).squeeze(-1)     # [B]

        loss = None
        if (y_cls is not None) or (y_reg is not None):
            loss = 0.0
            if y_cls is not None:
                loss = loss + F.cross_entropy(logits_cls, y_cls.long())
            if y_reg is not None:
                loss = loss + self.loss_w_reg * F.mse_loss(pred_reg.float(), y_reg.float())

        return {
            "logits_cls": logits_cls,
            "pred_reg": pred_reg,
            "loss": loss
        }


# ============================================================================
# MISTRAL HUMOR SCORER (main API)
# ============================================================================

class MistralHumorScorer:
    """
    Humor scorer using fine-tuned Mistral model.

    Compatible with main_v3.ipynb LangGraph pipeline.
    Outputs regression scores in range [0, 4] based on training data.
    """

    def __init__(
        self,
        base_model_id: str = "mistralai/Ministral-3-8B-Base-2512",
        checkpoint_path: str = None,  # Auto-detected; see TRAINING_ANALYSIS_REPORT.md
        num_labels: int = 2,
        max_length: int = 512,
        device: str = None,
        batch_size: int = 8
    ):
        """
        Initialize Mistral scorer.

        Args:
            base_model_id: HuggingFace model ID for base Mistral model
            checkpoint_path: Path to fine-tuned LoRA checkpoint
            num_labels: Number of classification labels (default 2: humor/not humor)
            max_length: Max token length for inputs
            device: Device to run on (auto-detected if None)
            batch_size: Batch size for scoring
        """
        self.base_model_id = base_model_id
        if checkpoint_path is None:
            self.checkpoint_path = (
                Path(__file__).resolve().parent
                / "mistral_model"
                / "checkpoints_ministral3_multitask"
                / "checkpoint-600"
            )
        else:
            self.checkpoint_path = Path(checkpoint_path)
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.tokenizer = None
        self.model = None

    def load_model(self):
        """
        Load the fine-tuned Mistral model with LoRA weights.

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist
            RuntimeError: If model loading fails
        """
        print(f"Loading Mistral evaluator from {self.checkpoint_path}...")

        # Verify checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please ensure you have trained the model using finetune_mistral3.ipynb\n"
                f"Expected files: adapter_config.json, adapter_model.safetensors"
            )

        # Verify required files exist
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing_files = [f for f in required_files if not (self.checkpoint_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Checkpoint incomplete. Missing files: {missing_files}\n"
                f"Checkpoint path: {self.checkpoint_path}"
            )

        try:
            # Load tokenizer
            print("   Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                use_fast=True,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            # Load base model backbone (using bfloat16 for inference)
            # IMPORTANT: use the FULL backbone (NOT get_text_backbone) to match
            # training architecture. Training used MultiTaskRegCls(backbone, ...)
            # where backbone is the full Mistral3Model (including vision_tower).
            # The LoRA checkpoint keys include "backbone.language_model.layers..."
            # which only matches when backbone IS the full wrapper model.
            print("   Loading base model backbone (4-bit NF4)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            backbone = AutoModel.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            # Wrap in MultiTaskRegCls architecture (MUST match training)
            # Use FULL backbone, not text_backbone - matches training cell 7efe0264
            print("   Wrapping in MultiTaskRegCls...")
            model = MultiTaskRegCls(
                backbone,
                num_labels=self.num_labels,
                loss_w_reg=1.0,
                dropout=0.1
            )
            # Cast heads to backbone's device; use float32 for numerical stability
            # (bfloat16 heads can produce NaN in regression output)
            head_device = next(backbone.parameters()).device
            model.cls_head = model.cls_head.to(device=head_device, dtype=torch.float32)
            model.reg_head = model.reg_head.to(device=head_device, dtype=torch.float32)

            # Load LoRA config and apply PEFT wrapper.
            # v2 checkpoint: modules_to_save=["cls_head","reg_head"] (head weights included)
            # v1 checkpoint: modules_to_save=["classifier","score"] (head weights missing)
            print("   Loading LoRA weights...")
            saved_cfg = PeftConfig.from_pretrained(str(self.checkpoint_path))

            # Check if checkpoint has head weights (v2) or not (v1)
            adapter_weights = load_safetensors(
                str(self.checkpoint_path / "adapter_model.safetensors")
            )
            has_head_weights = any("cls_head" in k or "reg_head" in k for k in adapter_weights)

            if has_head_weights:
                modules_to_save = ["cls_head", "reg_head"]
            else:
                modules_to_save = None  # v1: heads not in checkpoint

            lora_cfg = LoraConfig(
                r=saved_cfg.r,
                lora_alpha=saved_cfg.lora_alpha,
                lora_dropout=0.0,  # no dropout at inference
                bias=saved_cfg.bias,
                target_modules=list(saved_cfg.target_modules),
                modules_to_save=modules_to_save,
                inference_mode=True,
            )
            model = get_peft_model(model, lora_cfg)

            # Load weights into model (adapter_weights already loaded above)
            model_state = model.state_dict()

            # Load weights with key remapping.
            # LoRA keys need ".default" inserted (PEFT adapter name).
            # Head keys (v2 checkpoint) match directly.
            model_key_set = set(model_state.keys())
            matched, skipped = 0, 0
            for k, v in adapter_weights.items():
                # Try direct match first
                if k in model_key_set:
                    model_state[k] = v
                    matched += 1
                    continue
                # Try inserting ".default" before ".weight" for LoRA keys
                if ".lora_A.weight" in k:
                    remapped = k.replace(".lora_A.weight", ".lora_A.default.weight")
                elif ".lora_B.weight" in k:
                    remapped = k.replace(".lora_B.weight", ".lora_B.default.weight")
                else:
                    remapped = None

                if remapped and remapped in model_key_set:
                    model_state[remapped] = v
                    matched += 1
                else:
                    skipped += 1

            model.load_state_dict(model_state, strict=False)
            head_matched = sum(1 for k in adapter_weights if "cls_head" in k or "reg_head" in k)
            lora_matched = matched - head_matched
            print(f"   Loaded {lora_matched} LoRA + {head_matched} head weight tensors ({skipped} skipped)")
            if not has_head_weights:
                print("   WARNING: No head weights in checkpoint. Call calibrate_heads() for usable scores.")

            model.eval()
            self.model = model

            # Sanity check: score a dummy joke and check for NaN
            self._sanity_check()

            print(f"OK: Mistral evaluator loaded successfully on {self.device}")
            print(f"   Model: {self.base_model_id}")
            print(f"   Checkpoint: {self.checkpoint_path}")
            print(f"   Max length: {self.max_length}")
            print(f"   Output range: [0, 4]\n")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model files not found. Ensure checkpoint exists at: {self.checkpoint_path}\n"
                f"Original error: {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Mistral model. This could be due to:\n"
                f"  - Missing dependencies (transformers, peft, torch)\n"
                f"  - Incompatible checkpoint format\n"
                f"  - Insufficient memory (model requires ~8GB)\n"
                f"  - CUDA/device errors\n"
                f"Original error: {str(e)}"
            ) from e

    def calibrate_heads(
        self,
        train_csv: str = None,
        n_samples: int = 500,
        epochs: int = 30,
        lr: float = 1e-3,
    ):
        """
        Quick calibration of cls_head and reg_head on training data.

        The LoRA checkpoint only saved backbone adapter weights; the heads
        (cls_head, reg_head) were never saved due to a modules_to_save naming
        mismatch during training.  This method freezes the backbone, extracts
        pooled embeddings once, and trains the two tiny heads (~12K params)
        on a random subset of the original training data so the model can
        produce meaningful scores.

        Args:
            train_csv: Path to hahackathon_train.csv (auto-detected if None)
            n_samples: Number of samples to use (more = better but slower)
            epochs: Training epochs for the heads
            lr: Learning rate for head optimiser
        """
        import pandas as pd

        if self.model is None:
            raise RuntimeError("Call load_model() before calibrate_heads()")

        # Auto-detect training CSV
        if train_csv is None:
            for candidate in [
                Path(__file__).resolve().parent.parent.parent / "data" / "labels" / "hahackathon_train.csv",
                self.checkpoint_path.parent.parent.parent.parent / "data" / "labels" / "hahackathon_train.csv",
            ]:
                if candidate.exists():
                    train_csv = str(candidate)
                    break
            if train_csv is None:
                raise FileNotFoundError(
                    "Cannot find hahackathon_train.csv. Pass train_csv= explicitly."
                )

        print(f"   Calibrating heads on {n_samples} samples from {Path(train_csv).name}...")

        # Load data -- keep only rows with humor_rating
        df = pd.read_csv(train_csv)
        df = df.dropna(subset=["humor_rating"])
        df = df.sample(n=min(n_samples, len(df)), random_state=42)

        texts = df["text"].tolist()
        y_reg = torch.tensor(df["humor_rating"].values, dtype=torch.float32)
        y_cls = torch.tensor(df["is_humor"].values, dtype=torch.long)

        # ---- Phase 1: extract pooled embeddings (backbone frozen) ----
        all_pooled = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                inputs = self.tokenizer(
                    batch, truncation=True, max_length=self.max_length,
                    padding="max_length", return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                backbone = self.model.base_model.model.backbone
                out = backbone(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True
                )
                pooled = last_token_pool(out.last_hidden_state, inputs["attention_mask"])
                all_pooled.append(pooled.float().cpu())

        X = torch.cat(all_pooled, dim=0)  # [N, H]
        print(f"   Extracted {X.shape[0]} embeddings (dim={X.shape[1]})")

        # ---- Phase 2: train heads on cached embeddings ----
        # Get the actual head modules (inside PeftModel wrapper)
        inner = self.model.base_model.model
        cls_head = inner.cls_head
        reg_head = inner.reg_head

        # PEFT freezes all non-LoRA params; re-enable grads on heads
        for p in cls_head.parameters():
            p.requires_grad_(True)
        for p in reg_head.parameters():
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(
            list(cls_head.parameters()) + list(reg_head.parameters()), lr=lr
        )

        X_dev = X.to(self.device)
        y_reg_dev = y_reg.to(self.device)
        y_cls_dev = y_cls.to(self.device)

        best_loss = float("inf")
        for epoch in range(epochs):
            cls_head.train()
            reg_head.train()

            logits = cls_head(X_dev)
            preds = reg_head(X_dev).squeeze(-1)

            loss_cls = F.cross_entropy(logits, y_cls_dev)
            loss_reg = F.mse_loss(preds, y_reg_dev)
            loss = loss_cls + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

        cls_head.eval()
        reg_head.eval()

        # Quick validation
        with torch.no_grad():
            final_preds = reg_head(X_dev).squeeze(-1)
            r2 = 1.0 - F.mse_loss(final_preds, y_reg_dev) / y_reg_dev.var()
            print(f"   Head calibration done: loss={best_loss:.4f}, R2={r2.item():.3f}")
            print(f"   Pred range: [{final_preds.min().item():.2f}, {final_preds.max().item():.2f}]")

    def _sanity_check(self):
        """Run a quick forward pass to verify the model produces valid (non-NaN) output."""
        test_text = "Why did the chicken cross the road? To get to the other side!"
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    [test_text], truncation=True, max_length=64,
                    padding="max_length", return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                pred = outputs["pred_reg"].float().cpu().item()
                if np.isnan(pred):
                    print(f"WARNING: Sanity check produced NaN! Model may not be working correctly.")
                    # Debug: check hidden states
                    backbone = self.model.base_model.model.backbone
                    out = backbone(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], return_dict=True)
                    hs = out.last_hidden_state
                    print(f"   Hidden states: shape={list(hs.shape)}, has_nan={torch.isnan(hs).any().item()}, "
                          f"min={hs.float().min().item():.4f}, max={hs.float().max().item():.4f}")
                else:
                    print(f"   Sanity check passed: test score = {pred:.3f}")
        except Exception as e:
            print(f"WARNING: Sanity check failed: {e}")

    def score_jokes(
        self,
        jokes: List[str],
        word1: str = "",
        word2: str = "",
        verbose: bool = False
    ) -> List[float]:
        """
        Score a list of jokes (main API for main_v3.ipynb).

        Args:
            jokes: List of joke strings
            word1: First word (kept for API consistency, not used directly)
            word2: Second word (kept for API consistency, not used directly)
            verbose: Print detailed scoring info

        Returns:
            List of scores in range [0, 4]

        Raises:
            RuntimeError: If model not loaded
            TypeError: If jokes is not a list
            ValueError: If jokes contains non-string elements
        """
        # Input validation
        if jokes is None:
            raise TypeError("jokes parameter cannot be None. Provide an empty list instead.")

        if not isinstance(jokes, list):
            raise TypeError(f"jokes must be a list, got {type(jokes).__name__}")

        if not jokes:
            return []

        # Validate all elements are strings
        for i, joke in enumerate(jokes):
            if not isinstance(joke, str):
                raise ValueError(f"All jokes must be strings. Element {i} is {type(joke).__name__}")

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        scores = []

        # Process in batches for efficiency
        try:
            with torch.no_grad():
                for i in range(0, len(jokes), self.batch_size):
                    batch_jokes = jokes[i:i + self.batch_size]

                    try:
                        # Tokenize batch
                        inputs = self.tokenizer(
                            batch_jokes,
                            truncation=True,
                            max_length=self.max_length,
                            padding="max_length",
                            return_tensors="pt"
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        # Forward pass
                        outputs = self.model(**inputs)

                        # Classification gate: if cls_head says "not humor" (class 0), score = 0
                        cls_logits = outputs["logits_cls"].float().cpu().numpy()
                        cls_preds = np.argmax(cls_logits, axis=1)  # 0=not humor, 1=humor

                        # Extract regression predictions (convert bfloat16 to float32 for numpy)
                        batch_scores = outputs["pred_reg"].float().cpu().numpy()

                        # Replace NaN with default score, then clip to [0, 4]
                        nan_mask = np.isnan(batch_scores)
                        if nan_mask.any():
                            print(f"WARNING: {nan_mask.sum()} NaN scores in batch, replacing with 2.0")
                            batch_scores = np.where(nan_mask, 2.0, batch_scores)
                        batch_scores = np.clip(batch_scores, 0.0, 4.0)

                        # Zero out scores for jokes classified as "not humor"
                        not_humor_mask = cls_preds == 0
                        if not_humor_mask.any() and verbose:
                            print(f"  cls_head filtered {not_humor_mask.sum()}/{len(cls_preds)} as not-humor -> score 0")
                        batch_scores = np.where(not_humor_mask, 0.0, batch_scores).tolist()
                        scores.extend(batch_scores)

                        if verbose:
                            for joke, score, is_humor in zip(batch_jokes, batch_scores, cls_preds):
                                label = "humor" if is_humor else "NOT humor"
                                print(f"  Mistral score: {score:.2f}/4 [{label}] - {joke[:80]}...")

                    except Exception as batch_error:
                        # Log batch error and append default scores
                        print(f"WARNING: Batch {i//self.batch_size + 1} failed: {batch_error}")
                        print(f"         Returning default scores (2.0) for batch")
                        # Append default mid-range scores for failed batch
                        scores.extend([2.0] * len(batch_jokes))

        except Exception as e:
            raise RuntimeError(f"Scoring failed: {str(e)}") from e

        # Final validation
        assert len(scores) == len(jokes), f"Score count mismatch: {len(scores)} vs {len(jokes)}"

        return scores

    def score_single_joke(self, joke: str, word1: str = "", word2: str = "") -> float:
        """
        Score a single joke - convenience method.

        Args:
            joke: Joke text
            word1: First word (optional)
            word2: Second word (optional)

        Returns:
            Score in range [0, 4]
        """
        scores = self.score_jokes([joke], word1, word2)
        return scores[0] if scores else 0.0

    def health_check(self) -> Dict:
        """
        Perform health check for production monitoring.

        Returns:
            Dict with health status information
        """
        status = {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device": str(self.device),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists() if hasattr(self, 'checkpoint_path') else False,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "ready": False
        }

        # Overall readiness
        status["ready"] = (
            status["model_loaded"] and
            status["tokenizer_loaded"] and
            status["checkpoint_exists"]
        )

        return status

    def batch_score_with_metadata(
        self,
        jokes: List[str],
        word_pairs: List[tuple] = None,
        return_classifications: bool = False
    ) -> List[Dict]:
        """
        Score jokes with additional metadata.

        Args:
            jokes: List of jokes
            word_pairs: Optional list of (word1, word2) tuples
            return_classifications: If True, also return binary humor classification

        Returns:
            List of dicts with keys: joke, score, (optionally) is_humor, word_pair
        """
        if not jokes:
            return []

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []

        with torch.no_grad():
            for i in range(0, len(jokes), self.batch_size):
                batch_jokes = jokes[i:i + self.batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_jokes,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(**inputs)

                # Regression scores (convert bfloat16 to float32 for numpy)
                batch_scores = outputs["pred_reg"].float().cpu().numpy()
                batch_scores = np.clip(batch_scores, 0.0, 4.0)

                # Classification if requested
                if return_classifications:
                    cls_logits = outputs["logits_cls"].float().cpu().numpy()
                    cls_preds = np.argmax(cls_logits, axis=1)

                # Build result dicts
                for j, (joke, score) in enumerate(zip(batch_jokes, batch_scores)):
                    result = {
                        "joke": joke,
                        "score": float(score)
                    }

                    if return_classifications:
                        result["is_humor"] = int(cls_preds[j])

                    if word_pairs and (i + j) < len(word_pairs):
                        result["word_pair"] = word_pairs[i + j]

                    results.append(result)

        return results


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the Mistral scorer independently"""

    # Configuration
    CHECKPOINT_PATH = "./checkpoints_ministral3_multitask"

    # Test jokes
    test_jokes = [
        "Why did the banana go to space? Because it wanted to be a satellite dish!",
        "This is not funny at all.",
        "What do you call an angry teacup? A storm in a teacup!",
        "The weather is nice today.",
        "Why don't scientists trust atoms? Because they make up everything!"
    ]

    print("="*80)
    print("MISTRAL HUMOR SCORER - STANDALONE TEST")
    print("="*80)
    print()

    # Initialize and load
    scorer = MistralHumorScorer(
        base_model_id="mistralai/Ministral-3-8B-Base-2512",
        checkpoint_path=CHECKPOINT_PATH,
        num_labels=2,
        max_length=512,
        batch_size=4
    )

    try:
        scorer.load_model()

        # Test basic scoring
        print("\n📊 Testing basic scoring:")
        print("-"*80)
        scores = scorer.score_jokes(test_jokes, verbose=True)

        print("\n📈 Results Summary:")
        print("-"*80)
        for joke, score in zip(test_jokes, scores):
            print(f"Score: {score:.2f}/4 - {joke}")

        # Test with metadata
        print("\n\n📋 Testing with metadata:")
        print("-"*80)
        word_pairs = [("banana", "satellite"), ("test", "test"), ("angry", "teacup"),
                     ("weather", "nice"), ("scientists", "atoms")]

        results = scorer.batch_score_with_metadata(
            test_jokes,
            word_pairs=word_pairs,
            return_classifications=True
        )

        for r in results:
            print(f"Score: {r['score']:.2f}/4 | Humor: {r['is_humor']} | "
                  f"Pair: {r.get('word_pair', 'N/A')} | {r['joke'][:60]}...")

        print("\n✅ Standalone test complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
