# prompt_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any, List

from patterns import PATTERN_DEFINITIONS  # style library :contentReference[oaicite:8]{index=8}


class LLMClient(Protocol):
    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        ...


def prompt_step1_associations(text: str) -> str:
    return f"""
Task: Step 1 (Associations).
Input: {text}

You are very creative and can quickly think of many different related ideas.
Generate exactly 10 associations for the input.
Mix: objects, emotions, places, actions, metaphors, social situations.
Keep it SAFE and neutral (no hate, no slurs, no sexual content).

Output format (exact):
1) ...
2) ...
...
10) ...
""".strip()


def prompt_step2_imagery(text: str, assoc_text: str) -> str:
    return f"""
Task: Step 2 (Positive / implicit imagery).
Input: {text}

Associations:
{assoc_text}

You are a funny story teller who can turn ideas into vivid mental images.
Rewrite each association into a short vivid image / humorous picture language.
Keep it positive/implicit (no insults, no cruelty).
Keep each item <= 12 words.

Output format (exact):
1) ...
2) ...
...
10) ...
""".strip()


def prompt_step3_final_joke(
    pattern: str,
    text: str,
    imagery_text: str,
    *,
    retrieved_examples: Optional[str] = None,
) -> str:
    # NOTE: In your generator_pip.py you do definition = PATTERN_DEFINITIONS.get(pattern,"")
    # but PATTERN_DEFINITIONS entries are dicts, not strings. :contentReference[oaicite:9]{index=9}
    # Here we format them safely.
    definition = PATTERN_DEFINITIONS.get(pattern, {})
    prompt_template = ""
    if isinstance(definition, dict):
        prompt_template = definition.get("prompt_template", "")

    rag_block = ""
    if retrieved_examples:
        rag_block = f"""
Inspiration examples (do NOT copy wording/structure):
{retrieved_examples}

""".strip()

    return f"""
You are a professional comedy writer.

Joke style: {pattern}
Style guide: {prompt_template}

Core humor rule (must apply):
- TRUTH first: a relatable observation.
- PRINCIPLE → SURPRISE:
  PRINCIPLE = expected rule/interpretation.
  SURPRISE  = twist that breaks expectation but still connects logically.
- Punchline must be last.

Input: {text}

{rag_block}

Imagery candidates:
{imagery_text}

Constraints:
- If input is a headline, write a humorous comment or punchline (not a parody of the headline).
- If input is two words, use both naturally in your joke.

Task:
- Follow the constraints strictly.
- Build 1 joke (1–10 sentences).
- Choose at least 3 imagery items from the list (by index) mentally.

Writing Instructions:
- Be specific, not generic — concrete details increase funniness.
- Avoid clichés and generic templates like "Why did ...".
- No explanation, no labels.

Output format (exact):
- Output only the joke text.
""".strip()


@dataclass
class PipelineConfig:
    temperature_step1: float = 0.7
    temperature_step2: float = 0.8
    temperature_step3: float = 0.9
    max_tokens_step1: int = 200
    max_tokens_step2: int = 250
    max_tokens_step3: int = 200


class HumorPromptPipeline:
    def __init__(self, llm: LLMClient, cfg: PipelineConfig = PipelineConfig()):
        self.llm = llm
        self.cfg = cfg

    def run(
        self,
        *,
        pattern: str,
        text: str,
        retrieved_examples: Optional[str] = None,
        return_steps: bool = False,
    ):
        p1 = prompt_step1_associations(text)
        s1 = self.llm.complete(p1, temperature=self.cfg.temperature_step1, max_tokens=self.cfg.max_tokens_step1)

        p2 = prompt_step2_imagery(text, s1)
        s2 = self.llm.complete(p2, temperature=self.cfg.temperature_step2, max_tokens=self.cfg.max_tokens_step2)

        p3 = prompt_step3_final_joke(pattern, text, s2, retrieved_examples=retrieved_examples)
        s3 = self.llm.complete(p3, temperature=self.cfg.temperature_step3, max_tokens=self.cfg.max_tokens_step3).strip()

        if return_steps:
            return s3, {"step1": s1, "step2": s2, "step3": s3}
        return s3
