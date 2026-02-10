"""
RAG-Enhanced Humor Generation System with Multi-Step Prompt Pipeline
Combines fine-tuned Llama 3 8B with retrieval-augmented generation

DEFAULT MODE: RAG + Multi-Step Pipeline (Best Quality)
Use use_rag=False or use_pipeline=False only for testing/ablation studies

Usage:
    from using_llama_with_rag import RAGHumorGenerator, RAGConfig

    # Initialize
    config = RAGConfig()
    generator = RAGHumorGenerator(config)
    generator.load_model()
    generator.load_rag_database()

    # Simple generation (uses RAG + Pipeline automatically)
    joke = generator.generate("coffee", "deadline")
    print(joke)

    # Detailed generation with options
    result = generator.generate_with_rag(
        "algorithm", "debugging",
        use_rag=True,
        use_pipeline=True,
        return_retrieved=True
    )
"""

from __future__ import annotations

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pickle
from tqdm import tqdm
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss

# Import multi-step prompt pipeline components
from prompt_pipeline import HumorPromptPipeline, PipelineConfig, LLMClient
from patterns import JOKE_PATTERNS


# ============================================================================
# HUGGINGFACE CHAT CLIENT ADAPTER
# ============================================================================

class HFChatClient(LLMClient):
    """Adapter to use RAGHumorGenerator with HumorPromptPipeline"""

    def __init__(self, generator: "RAGHumorGenerator"):
        self.g = generator

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        """Complete a prompt using the generator's model with custom parameters"""
        return self.g._generate_joke_custom(
            prompt,
            temperature=temperature,
            max_new_tokens=max_tokens
        )


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration for RAG system - Optimized for Multi-Step Pipeline"""

    # Model paths
    BASE_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    LORA_PATH: str = r"C:\Users\Anwender\humor-project\src\generation\llama3_humor-neu\llama3-humor-lora\checkpoint-2300"

    # RAG database
    JOKES_DATASET: str = r"C:\Users\Anwender\humor-project\data\humor_RAG_data_20000_RAG.jsonl"
    RAG_INDEX_PATH: str = "./rag_index"

    # Embedding model for retrieval
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval settings (optimized for pipeline)
    TOP_K: int = 3  # ✅ OPTIMIZED: Only retrieve what we use
    CANDIDATE_K: int = 12  # ✅ OPTIMIZED: 4x TOP_K for MMR diversity
    LAMBDA: float = 0.4
    SIMILARITY_THRESHOLD: float = 0.2

    # Generation settings
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.85  # High creativity for final joke
    TOP_P: float = 0.9
    USE_WANDB: bool = False

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Pipeline optimization settings
    PIPELINE_TEMP_STEP1: float = 0.5  # ✅ OPTIMIZED: Lower for faster associations
    PIPELINE_TEMP_STEP2: float = 0.6  # ✅ OPTIMIZED: Lower for faster imagery
    PIPELINE_MAX_TOKENS_STEP1: int = 150  # ✅ OPTIMIZED
    PIPELINE_MAX_TOKENS_STEP2: int = 200  # ✅ OPTIMIZED


# ============================================================================
# RAG DATABASE BUILDER
# ============================================================================

class HumorRAGDatabase:
    """Build and query a RAG database of joke patterns and examples"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = None
        self.index = None
        self.jokes_data = []
        self.use_diversity = True
        self.lambda_param = config.LAMBDA

    def build_database(self, jokes_file: str):
        """Build the RAG database from training jokes"""
        print("🏗️  Building RAG Database...")

        print(f"  Loading embedding model: {self.config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.embedding_model.to(self.config.DEVICE)

        print(f"  Loading jokes from: {jokes_file}")
        self.jokes_data = self._load_jokes(jokes_file)
        print(f"  Loaded {len(self.jokes_data)} jokes")

        print("  Creating embeddings...")
        embeddings = self._create_embeddings()

        print("  Building FAISS index...")
        self.index = self._build_faiss_index(embeddings)

        self._save_database()
        print(f"✓ RAG Database built with {len(self.jokes_data)} jokes\n")

    def _load_jokes(self, jokes_file: str) -> List[Dict]:
        """Load jokes from JSONL file"""
        jokes = []
        with open(jokes_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                joke_text = data['messages'][1]['content']
                word1 = data.get('word1', '')
                word2 = data.get('word2', '')
                jokes.append({
                    'joke': joke_text,
                    'word1': word1,
                    'word2': word2,
                    'constraint_text': f"{word1} {word2}",
                })
        return jokes

    def _create_embeddings(self) -> np.ndarray:
        """Create embeddings for all jokes"""
        texts_to_embed = [f"{j['constraint_text']}: {j['joke']}" for j in self.jokes_data]
        embeddings = self.embedding_model.encode(
            texts_to_embed,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index

    def _save_database(self):
        """Save database to disk"""
        os.makedirs(self.config.RAG_INDEX_PATH, exist_ok=True)
        faiss.write_index(self.index, f"{self.config.RAG_INDEX_PATH}/faiss.index")
        with open(f"{self.config.RAG_INDEX_PATH}/jokes_data.pkl", 'wb') as f:
            pickle.dump(self.jokes_data, f)
        print(f"  ✓ Database saved to {self.config.RAG_INDEX_PATH}")

    def load_database(self):
        """Load pre-built database from disk"""
        print("📂 Loading RAG Database...")
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.embedding_model.to(self.config.DEVICE)
        self.index = faiss.read_index(f"{self.config.RAG_INDEX_PATH}/faiss.index")
        with open(f"{self.config.RAG_INDEX_PATH}/jokes_data.pkl", 'rb') as f:
            self.jokes_data = pickle.load(f)
        print(f"✓ Loaded database with {len(self.jokes_data)} jokes\n")

    def retrieve(self, query: str, word1: str = None, word2: str = None, k: int = None):
        """Main retrieval method with optional diversity"""
        if k is None:
            k = self.config.TOP_K
        if self.use_diversity:
            return self.retrieve_with_diversity(
                query, word1, word2, k=k,
                candidate_k=max(k * 4, self.config.CANDIDATE_K),
                lambda_param=self.lambda_param
            )
        else:
            return self._semantic_search(query, word1, word2, k=k)

    def retrieve_with_diversity(
        self, query: str, word1: str = None, word2: str = None,
        k: int = 5, candidate_k: int = 20, lambda_param: float = 0.5
    ) -> List[Dict]:
        """Retrieve jokes using Maximal Marginal Relevance (MMR)"""
        candidates = self._semantic_search(query, word1, word2, k=candidate_k)
        if len(candidates) <= k:
            return candidates
        diverse_results = self._mmr_diversify(candidates, k, lambda_param)
        return diverse_results

    def _semantic_search(self, query: str, word1: str = None, word2: str = None, k: int = 20) -> List[Dict]:
        """Pure semantic search - gets top-k most similar jokes"""
        query_text = f"{word1} {word2}: {query}" if word1 and word2 else query
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.config.SIMILARITY_THRESHOLD:
                joke_data = self.jokes_data[idx].copy()
                joke_data['similarity_score'] = float(score)
                joke_data['idx'] = idx
                candidates.append(joke_data)
        return candidates

    def _mmr_diversify(self, candidates: List[Dict], k: int, lambda_param: float = 0.5) -> List[Dict]:
        """Apply MMR diversification"""
        if len(candidates) <= k:
            return candidates

        candidate_texts = [c['joke'] for c in candidates]
        candidate_embeddings = self.embedding_model.encode(
            candidate_texts, convert_to_numpy=True, show_progress_bar=False
        )
        faiss.normalize_L2(candidate_embeddings)

        selected_indices = [0]
        selected_embeddings = [candidate_embeddings[0]]

        for _ in range(k - 1):
            best_score = -np.inf
            best_candidate_idx = None

            for idx, candidate in enumerate(candidates):
                if idx in selected_indices:
                    continue

                relevance = candidate['similarity_score']
                candidate_emb = candidate_embeddings[idx].reshape(1, -1)
                selected_emb_matrix = np.array(selected_embeddings)
                similarities_to_selected = np.dot(selected_emb_matrix, candidate_emb.T).flatten()
                max_similarity_to_selected = np.max(similarities_to_selected)

                mmr_score = (lambda_param * relevance - (1 - lambda_param) * max_similarity_to_selected)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate_idx = idx

            if best_candidate_idx is not None:
                selected_indices.append(best_candidate_idx)
                selected_embeddings.append(candidate_embeddings[best_candidate_idx])

        return [candidates[idx] for idx in selected_indices]


# ============================================================================
# RAG-ENHANCED MODEL WITH MULTI-STEP PIPELINE
# ============================================================================

class RAGHumorGenerator:
    """
    Combines fine-tuned Llama with RAG and multi-step prompt pipeline

    DEFAULT: Uses RAG + Multi-Step Pipeline for best quality
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.rag_db = None

    def load_model(self):
        """Load fine-tuned model with LoRA"""
        print("🤖 Loading Fine-tuned Model...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.LORA_PATH, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print("  ✓ Tokenizer loaded")

        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  ✓ Base model loaded")

        self.model = PeftModel.from_pretrained(base_model, self.config.LORA_PATH)
        self.model.eval()
        print("  ✓ LoRA weights loaded")
        print(f"  ✓ Model ready on {self.config.DEVICE}\n")

    def load_rag_database(self, build_if_missing: bool = True):
        """Load or build RAG database"""
        self.rag_db = HumorRAGDatabase(self.config)
        if os.path.exists(f"{self.config.RAG_INDEX_PATH}/faiss.index"):
            self.rag_db.load_database()
        elif build_if_missing:
            print("⚠️  RAG database not found, building new one...")
            self.rag_db.build_database(self.config.JOKES_DATASET)
        else:
            raise FileNotFoundError(f"RAG database not found at {self.config.RAG_INDEX_PATH}")

    def generate(self, word1: str, word2: str = "") -> str:
        """
        MAIN GENERATION METHOD - Uses best quality settings by default

        Generate a joke using RAG + Multi-Step Pipeline (default, best quality).
        For testing other modes, use generate_with_rag() directly.

        Args:
            word1: First constraint word (or headline text if word2 is empty)
            word2: Second constraint word (empty string for headline mode)

        Returns:
            Generated joke string

        Example:
            >>> joke = generator.generate("coffee", "deadline")
            >>> joke = generator.generate("Scientists discover coffee is sentient")
        """
        result = self.generate_with_rag(word1=word1, word2=word2, use_rag=True, use_pipeline=True)
        return result['best_joke']

    def generate_with_rag(
        self,
        word1: str,
        word2: str = "",
        use_rag: bool = True,       # DEFAULT: RAG enabled
        use_pipeline: bool = True,  # DEFAULT: Pipeline enabled
        num_candidates: int = 1,
        return_retrieved: bool = False
    ) -> Dict:
        """
        Generate joke with configurable RAG and pipeline settings

        DEFAULT SETTINGS: use_rag=True, use_pipeline=True (Best Quality)
        Only change these for testing/ablation studies!

        Supports two modes:
        - Word-pair mode (word2 non-empty): joke connecting both words
        - Headline mode (word2=""): funny commentary on the headline in word1

        Args:
            word1: First constraint word (or headline text if word2 is empty)
            word2: Second constraint word (empty string for headline mode)
            use_rag: Whether to use RAG (default=True for best quality)
            use_pipeline: Whether to use multi-step pipeline (default=True for best quality)
            num_candidates: Number of jokes to generate
            return_retrieved: Whether to return retrieved examples

        Returns:
            Dictionary with generated joke(s) and metadata
        """
        retrieved_jokes = []

        if use_rag and self.rag_db is not None:
            if word2:
                query = f"joke with {word1} and {word2}"
                retrieved_jokes = self.rag_db.retrieve(query=query, word1=word1, word2=word2, k=self.config.TOP_K)
            else:
                # Headline mode: retrieve jokes by headline content
                query = f"funny commentary on: {word1}"
                retrieved_jokes = self.rag_db.retrieve(query=query, k=self.config.TOP_K)

        generated_jokes = []
        if use_pipeline:
            for _ in range(num_candidates):
                joke = self._generate_with_pipeline(word1, word2, retrieved_jokes)
                generated_jokes.append(joke)
        else:
            prompt = self._construct_prompt(word1, word2, retrieved_jokes)
            for _ in range(num_candidates):
                joke = self._generate_joke(prompt)
                generated_jokes.append(joke)

        result = {
            'word1': word1,
            'word2': word2,
            'generated_jokes': generated_jokes,
            'best_joke': generated_jokes[0] if generated_jokes else "",
            'used_rag': use_rag,
            'used_pipeline': use_pipeline,
            'num_retrieved': len(retrieved_jokes)
        }
        if return_retrieved:
            result['retrieved_examples'] = retrieved_jokes
        return result

    def _generate_with_pipeline(self, word1: str, word2: str, retrieved_jokes: List[Dict]) -> str:
        """Generate joke using multi-step prompt pipeline"""
        retrieved_block = ""
        if retrieved_jokes:
            retrieved_block = "\n".join([f"- {j['joke']}" for j in retrieved_jokes[:3]])

        text = f"{word1} {word2}".strip()
        pattern = random.choice(JOKE_PATTERNS)  # Random pattern selection

        llm = HFChatClient(self)
        pipe = HumorPromptPipeline(llm, PipelineConfig(
            temperature_step1=self.config.PIPELINE_TEMP_STEP1,
            temperature_step2=self.config.PIPELINE_TEMP_STEP2,
            temperature_step3=self.config.TEMPERATURE,
            max_tokens_step1=self.config.PIPELINE_MAX_TOKENS_STEP1,
            max_tokens_step2=self.config.PIPELINE_MAX_TOKENS_STEP2,
            max_tokens_step3=self.config.MAX_NEW_TOKENS,
        ))

        return pipe.run(
            pattern=pattern,
            text=text,
            retrieved_examples=retrieved_block if retrieved_block else None
        )

    def _generate_joke_custom(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        """
        Custom generation method with per-call temperature and max_tokens.
        Used by HFChatClient adapter for multi-step pipeline.
        """
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.config.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in generated:
            return generated.split("assistant")[-1].strip()
        return generated[len(formatted_prompt):].strip()

    def _construct_prompt(self, word1: str, word2: str, retrieved_jokes: List[Dict]) -> str:
        """Construct prompt with optional RAG context (traditional single-step)"""
        if word2:
            base_instruction = f"Generate a funny joke that naturally includes both of these words: '{word1}' and '{word2}'. The joke should be creative, humorous, and incorporate both words seamlessly."
        else:
            base_instruction = f"Generate a funny joke or humorous commentary about this headline: \"{word1}\". The joke should be creative and add a comedic twist."

        if not retrieved_jokes:
            return base_instruction

        context = """You are a comedian. You are generating humor.

A joke is funny because multiple cognitive mechanisms work together at once.
Do NOT copy wording, phrasing, or structure from the examples.
Instead, analyze WHY they work.

For each example below, humor may come from one or more of the following factors:
- Expectation violation (what the listener predicts vs what actually happens)
- Internal contradiction or incongruity
- Reversal of roles, status, or intent
- Literal interpretation of figurative language
- Treating something serious as trivial, or trivial as serious
- Absurd logic presented as reasonable
- Social norm violation that is harmless
- Overly confident reasoning applied to something wrong
- Sudden re-interpretation of earlier information
- Misdirection followed by a clean punchline
- Emotional contrast (calm setup -> extreme or petty conclusion)
- Timing and brevity (a sharp, minimal punchline)

Below are jokes that successfully combine several of these mechanisms.
For each one, mentally identify:
- What expectation is created
- What assumption is broken
- What logical or social rule is violated
- Why the violation feels safe or playful rather than confusing

Examples:"""

        for i, joke_data in enumerate(retrieved_jokes[:3], 1):
            context += f"Example {i}: {joke_data['joke']}\n"

        if word2:
            context += f"""\n- Create a NEW joke that naturally includes BOTH of these words: "{word1}" and "{word2}"
- The joke must use at least TWO humor mechanisms from the list above
- The punchline should reframe the setup in an unexpected but internally consistent way
- The humor should come from meaning, not wordplay alone
- Do NOT explain the joke
- Do NOT reference the examples
- DO NOT explain why is the joke funny.
- DO NOT elaborate on joke.

Generate only the joke text and nothing else., {base_instruction.lower()}"""
        else:
            context += f"""\n- Create a NEW funny reaction or commentary about this headline: "{word1}"
- The joke must use at least TWO humor mechanisms from the list above
- The punchline should reframe the headline in an unexpected but internally consistent way
- The humor should come from meaning, not wordplay alone
- Do NOT explain the joke
- Do NOT reference the examples
- DO NOT explain why is the joke funny.
- DO NOT elaborate on joke.

Generate only the joke text and nothing else."""
        return context

    def _generate_joke(self, prompt: str) -> str:
        """Generate a single joke (traditional single-step)"""
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in generated:
            return generated.split("assistant")[-1].strip()
        return generated[len(formatted_prompt):].strip()

    def batch_generate(
        self,
        word_pairs: List[Tuple[str, str]],
        use_rag: bool = True,      # ✅ DEFAULT: RAG enabled
        use_pipeline: bool = True  # ✅ DEFAULT: Pipeline enabled
    ) -> List[Dict]:
        """
        Generate jokes for multiple word pairs

        ⚠️  DEFAULT: use_rag=True, use_pipeline=True (Best Quality)
        Only change for testing!
        """
        results = []
        mode_str = "Pipeline" if use_pipeline else "Single-step"
        rag_str = "with RAG" if use_rag else "without RAG"
        print(f"🎭 Generating jokes ({mode_str}, {rag_str}) for {len(word_pairs)} word pairs...")

        for word1, word2 in tqdm(word_pairs):
            result = self.generate_with_rag(word1, word2, use_rag=use_rag, use_pipeline=use_pipeline)
            results.append(result)
        return results


# ============================================================================
# EVALUATION & COMPARISON (FOR TESTING ONLY)
# ============================================================================

class RAGEvaluator:
    """
    Compare RAG vs non-RAG and pipeline vs single-step generation
    ⚠️  FOR TESTING/ABLATION STUDIES ONLY - Production should use defaults
    """

    @staticmethod
    def compare_generations(generator: RAGHumorGenerator, test_pairs: List[Tuple[str, str]]):
        """Compare different generation strategies"""
        print("🔬 Comparing Generation Strategies (Testing Only)\n")
        print("="*80)

        for word1, word2 in test_pairs:
            print(f"\n📝 Words: {word1}, {word2}")
            print("-"*80)

            # Strategy 1: Without RAG, single-step (TESTING ONLY)
            result_basic = generator.generate_with_rag(
                word1, word2, use_rag=False, use_pipeline=False, return_retrieved=False
            )
            print(f"\n[TEST: Basic (No RAG, Single-step)]")
            print(f"{result_basic['best_joke']}")

            # Strategy 2: With RAG, single-step (TESTING ONLY)
            result_rag = generator.generate_with_rag(
                word1, word2, use_rag=True, use_pipeline=False, return_retrieved=True
            )
            print(f"\n[TEST: RAG + Single-step] (Retrieved {result_rag['num_retrieved']} examples)")
            print(f"{result_rag['best_joke']}")

            # Strategy 3: With RAG, multi-step pipeline (✅ DEFAULT/PRODUCTION)
            result_pipeline = generator.generate_with_rag(
                word1, word2, use_rag=True, use_pipeline=True, return_retrieved=True
            )
            print(f"\n[✅ PRODUCTION DEFAULT: RAG + Multi-step Pipeline] (Retrieved {result_pipeline['num_retrieved']} examples)")
            print(f"{result_pipeline['best_joke']}")

            if result_rag.get('retrieved_examples'):
                print(f"\n[Retrieved Examples:]")
                for i, ex in enumerate(result_rag['retrieved_examples'][:2], 1):
                    print(f"  {i}. {ex['joke'][:100]}... (score: {ex['similarity_score']:.3f})")

            print("="*80)


# ============================================================================
# MAIN (for direct execution)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RAG HUMOR GENERATOR - Multi-Step Pipeline")
    print("=" * 80)

    # Initialize
    config = RAGConfig()
    generator = RAGHumorGenerator(config)

    print("\nLoading model and RAG database...")
    generator.load_model()
    generator.load_rag_database()

    print("\n" + "=" * 80)
    print("QUICK TEST")
    print("=" * 80)

    # Simple test
    test_words = [("coffee", "algorithm"), ("python", "debugging")]

    for word1, word2 in test_words:
        print(f"\nGenerating joke for: {word1} + {word2}")
        joke = generator.generate(word1, word2)
        print(f"Result: {joke}\n")

    print("=" * 80)
    print("✅ Test completed!")
