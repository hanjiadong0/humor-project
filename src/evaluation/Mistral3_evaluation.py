# mistral_evaluator.py
"""
Mistral-based Humor Evaluator
Fine-tuned Mistral model for scoring jokes (0-4 scale)
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import torch.nn as nn
import torch.nn.functional as F


def get_text_backbone(model):
    """
    Extract text backbone from potentially multimodal wrapper.
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


class MultiTaskRegCls(nn.Module):
    """
    Multi-task classification + regression model.
    Same architecture as used in fine-tuning.
    """
    def __init__(self, backbone, num_labels: int, loss_w_reg: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        h = self._get_hidden_size(backbone.config)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(h, num_labels)
        self.reg_head = nn.Linear(h, 1)
        self.loss_w_reg = loss_w_reg
        self.num_labels = num_labels

    def _get_hidden_size(self, cfg) -> int:
        """Extract hidden size from config (handles Mistral3Config wrapper)"""
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

    def _last_token_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool the last real token (not padding).
        
        Args:
            last_hidden_state: [B, T, H]
            attention_mask: [B, T]
        
        Returns:
            [B, H]
        """
        if attention_mask is None:
            return last_hidden_state[:, -1, :]
        
        lengths = attention_mask.long().sum(dim=1)  # [B]
        idx = torch.clamp(lengths - 1, min=0)       # [B]
        batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        return last_hidden_state[batch_idx, idx, :]  # [B, H]

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
        
        pooled = self._last_token_pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        
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


class MistralHumorScorer:
    """
    Humor scorer using fine-tuned Mistral model.
    
    Outputs regression scores in range [0, 4] based on your training data.
    """
    
    def __init__(
        self,
        base_model_id: str = "mistralai/Ministral-3-8B-Base-2512",
        checkpoint_path: str = "./checkpoints_ministral3_multitask",
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
        self.checkpoint_path = checkpoint_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the fine-tuned Mistral model with LoRA weights"""
        print(f"🔧 Loading Mistral evaluator from {self.checkpoint_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            use_fast=True,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print("   Loading base model backbone...")
        # Load base model backbone (using bfloat16 for inference)
        backbone = AutoModel.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Extract text backbone if multimodal wrapper
        text_backbone = get_text_backbone(backbone)
        
        print("   Wrapping in MultiTaskRegCls...")
        # Wrap in MultiTaskRegCls architecture
        model = MultiTaskRegCls(
            text_backbone,
            num_labels=self.num_labels,
            loss_w_reg=1.0,
            dropout=0.1
        )
        
        print("   Loading LoRA weights...")
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(model, self.checkpoint_path)
        self.model.eval()
        self.model.to(self.device)
        
        print(f"✓ Mistral evaluator loaded successfully on {self.device}")
        print(f"   Model: {self.base_model_id}")
        print(f"   Checkpoint: {self.checkpoint_path}")
        print(f"   Max length: {self.max_length}")
        print(f"   Output range: [0, 4]\\n")
        
    def score_jokes(
        self,
        jokes: List[str],
        word1: str = "",
        word2: str = "",
        verbose: bool = False
    ) -> List[float]:
        """
        Score a list of jokes.
        
        Args:
            jokes: List of joke strings
            word1: First word (kept for API consistency, not used directly)
            word2: Second word (kept for API consistency, not used directly)
            verbose: Print detailed scoring info
            
        Returns:
            List of scores in range [0, 4]
        """
        if not jokes:
            return []
            
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        scores = []
        
        # Process in batches for efficiency
        with torch.no_grad():
            for i in range(0, len(jokes), self.batch_size):
                batch_jokes = jokes[i:i + self.batch_size]
                
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
                
                # Extract regression predictions
                batch_scores = outputs["pred_reg"].cpu().numpy()
                
                # Clip to valid range [0, 4] and convert to list
                batch_scores = np.clip(batch_scores, 0.0, 4.0).tolist()
                scores.extend(batch_scores)
                
                if verbose:
                    for joke, score in zip(batch_jokes, batch_scores):
                        print(f"  Mistral score: {score:.2f}/4 - {joke[:80]}...")
        
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
                
                # Regression scores
                batch_scores = outputs["pred_reg"].cpu().numpy()
                batch_scores = np.clip(batch_scores, 0.0, 4.0)
                
                # Classification if requested
                if return_classifications:
                    cls_logits = outputs["logits_cls"].cpu().numpy()
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
    CHECKPOINT_PATH = "./checkpoints_ministral3_multitask"  # Update this
    
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
        print("\\n📊 Testing basic scoring:")
        print("-"*80)
        scores = scorer.score_jokes(test_jokes, verbose=True)
        
        print("\\n📈 Results Summary:")
        print("-"*80)
        for joke, score in zip(test_jokes, scores):
            print(f"Score: {score:.2f}/4 - {joke}")
        
        # Test with metadata
        print("\\n\\n📋 Testing with metadata:")
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
        
        print("\\n✅ Standalone test complete!")
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()