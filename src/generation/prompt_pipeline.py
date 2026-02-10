"""
Multi-Step Prompt Pipeline for Humor Generation

This module implements a three-stage pipeline for generating jokes:
1. Associations: Generate related concepts from input text
2. Imagery: Convert associations to vivid mental images
3. Final Joke: Generate joke using pattern-specific prompts

The pipeline uses the PRINCIPLE → SURPRISE framework where jokes establish
an expectation and then subvert it in a logical but unexpected way.

Example:
    from prompt_pipeline import HumorPromptPipeline, PipelineConfig, LLMClient

    pipeline = HumorPromptPipeline(llm_client, PipelineConfig())
    joke = pipeline.run(pattern="Pun (re-interpretation)", text="coffee deadline")
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Tuple, Union

from patterns import PATTERN_DEFINITIONS

# Set up module logger
logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """
    Protocol defining the interface for LLM clients.

    Any client implementing this protocol can be used with HumorPromptPipeline.
    Examples: HuggingFace transformers, OpenAI API, NVIDIA API, etc.
    """

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        """
        Generate text completion for a prompt.

        Args:
            prompt: The input prompt text
            temperature: Sampling temperature (higher = more random)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text string (may be empty if generation fails)
        """
        ...


def prompt_step1_associations(text: str) -> str:
    """
    Generate prompt for Step 1: Association Generation.

    Creates a prompt asking the LLM to generate 10 diverse associations
    for the input text, mixing objects, emotions, places, actions, etc.

    Args:
        text: Input text (typically a word pair or short phrase)

    Returns:
        Formatted prompt string for association generation

    Example:
        >>> prompt_step1_associations("coffee deadline")
        'Task: Step 1 (Associations)...\\nInput: coffee deadline\\n...'
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to prompt_step1_associations")
        text = "[empty input]"

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
    """
    Generate prompt for Step 2: Imagery Creation.

    Creates a prompt asking the LLM to convert associations into vivid,
    humorous mental images suitable for joke construction.

    Args:
        text: Original input text (for context)
        assoc_text: Associations generated in Step 1

    Returns:
        Formatted prompt string for imagery generation

    Example:
        >>> prompt_step2_imagery("coffee", "1) energy\\n2) morning...")
        'Task: Step 2 (Positive / implicit imagery)...\\n...'
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to prompt_step2_imagery")
        text = "[empty input]"

    if not assoc_text or not assoc_text.strip():
        logger.warning("Empty associations provided to prompt_step2_imagery")
        assoc_text = "1) [no associations available]"

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
    """
    Generate prompt for Step 3: Final Joke Generation.

    Creates a prompt that combines the pattern definition, imagery, and
    optional RAG examples to generate the final joke.

    Args:
        pattern: Humor pattern name (e.g., "Pun (re-interpretation)")
        text: Original input text
        imagery_text: Imagery generated in Step 2
        retrieved_examples: Optional RAG examples for inspiration

    Returns:
        Formatted prompt string for final joke generation

    Raises:
        ValueError: If pattern is not found in PATTERN_DEFINITIONS

    Example:
        >>> prompt_step3_final_joke("Pun", "coffee", "1) steaming mug...")
        'You are a professional comedy writer...\\n...'
    """
    # Validate inputs
    if not text or not text.strip():
        logger.warning("Empty text provided to prompt_step3_final_joke")
        text = "[empty input]"

    if not imagery_text or not imagery_text.strip():
        logger.warning("Empty imagery provided to prompt_step3_final_joke")
        imagery_text = "1) [no imagery available]"

    # Get pattern definition safely
    definition = PATTERN_DEFINITIONS.get(pattern, {})
    if not definition:
        logger.warning(f"Pattern '{pattern}' not found in PATTERN_DEFINITIONS")
        prompt_template = "Create a humorous joke."
    else:
        if isinstance(definition, dict):
            prompt_template = definition.get("prompt_template", "")
        else:
            logger.error(f"Pattern definition for '{pattern}' is not a dict: {type(definition)}")
            prompt_template = "Create a humorous joke."

    # Build RAG examples block if provided
    rag_block = ""
    if retrieved_examples and retrieved_examples.strip():
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
    """
    Configuration for HumorPromptPipeline.

    Attributes:
        temperature_step1: Sampling temperature for associations (0.0-2.0)
        temperature_step2: Sampling temperature for imagery (0.0-2.0)
        temperature_step3: Sampling temperature for final joke (0.0-2.0)
        max_tokens_step1: Maximum tokens for associations
        max_tokens_step2: Maximum tokens for imagery
        max_tokens_step3: Maximum tokens for final joke

    Note:
        Lower temperatures (0.5-0.7) for steps 1-2 can speed up generation
        with minimal quality impact. Step 3 benefits from higher temperature
        (0.8-1.0) for creative joke generation.
    """

    temperature_step1: float = 0.7
    temperature_step2: float = 0.8
    temperature_step3: float = 0.9
    max_tokens_step1: int = 200
    max_tokens_step2: int = 250
    max_tokens_step3: int = 200

    def __post_init__(self):
        """Validate configuration values."""
        # Validate temperature ranges
        for step, temp in [
            ("step1", self.temperature_step1),
            ("step2", self.temperature_step2),
            ("step3", self.temperature_step3),
        ]:
            if not 0.0 <= temp <= 2.0:
                logger.warning(
                    f"temperature_{step}={temp} is outside recommended range [0.0, 2.0]"
                )

        # Validate max_tokens
        for step, tokens in [
            ("step1", self.max_tokens_step1),
            ("step2", self.max_tokens_step2),
            ("step3", self.max_tokens_step3),
        ]:
            if tokens < 10:
                raise ValueError(
                    f"max_tokens_{step}={tokens} is too low (minimum 10 recommended)"
                )
            if tokens > 4096:
                logger.warning(
                    f"max_tokens_{step}={tokens} is very high (may be slow/expensive)"
                )


class HumorPromptPipeline:
    """
    Multi-step prompt pipeline for generating humorous content.

    This pipeline implements a three-stage process:
    1. Generate associations from input text
    2. Convert associations to vivid imagery
    3. Generate final joke using pattern-specific prompt

    The pipeline is designed to work with any LLM client implementing
    the LLMClient protocol.

    Attributes:
        llm: LLM client for text generation
        cfg: Pipeline configuration

    Example:
        >>> from prompt_pipeline import HumorPromptPipeline, PipelineConfig
        >>> pipeline = HumorPromptPipeline(my_llm_client)
        >>> joke = pipeline.run(pattern="Pun", text="coffee deadline")
        >>> print(joke)
        "I had to quit my job at the coffee shop due to a brewing deadline crisis..."
    """

    def __init__(
        self,
        llm: LLMClient,
        cfg: PipelineConfig = None,
    ):
        """
        Initialize the humor generation pipeline.

        Args:
            llm: LLM client implementing the LLMClient protocol
            cfg: Pipeline configuration (uses defaults if None)
        """
        self.llm = llm
        self.cfg = cfg if cfg is not None else PipelineConfig()

        logger.info(
            f"Initialized HumorPromptPipeline with temps: "
            f"[{self.cfg.temperature_step1}, {self.cfg.temperature_step2}, "
            f"{self.cfg.temperature_step3}]"
        )

    def run(
        self,
        *,
        pattern: str,
        text: str,
        retrieved_examples: Optional[str] = None,
        return_steps: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, str]]]:
        """
        Execute the full three-step pipeline to generate a joke.

        Args:
            pattern: Humor pattern name (from PATTERN_DEFINITIONS)
            text: Input text (typically a word pair or short phrase)
            retrieved_examples: Optional RAG examples for inspiration
            return_steps: If True, return intermediate steps with final joke

        Returns:
            If return_steps=False: Final joke text (str)
            If return_steps=True: Tuple of (joke, steps_dict)

        Raises:
            ValueError: If text is empty or pattern is invalid
            RuntimeError: If LLM generation fails critically

        Example:
            >>> joke = pipeline.run(pattern="Pun", text="coffee deadline")
            >>> joke, steps = pipeline.run(
            ...     pattern="Pun",
            ...     text="coffee deadline",
            ...     return_steps=True
            ... )
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        if not pattern or not pattern.strip():
            raise ValueError("Pattern cannot be empty")

        if pattern not in PATTERN_DEFINITIONS:
            logger.warning(
                f"Pattern '{pattern}' not found in PATTERN_DEFINITIONS. "
                f"Generation may use generic humor rules."
            )

        logger.debug(f"Starting pipeline for text='{text}', pattern='{pattern}'")

        # Step 1: Generate associations
        try:
            p1 = prompt_step1_associations(text)
            logger.debug(f"Step 1 prompt length: {len(p1)} chars")

            s1 = self.llm.complete(
                p1,
                temperature=self.cfg.temperature_step1,
                max_tokens=self.cfg.max_tokens_step1
            )

            # Handle None responses from LLM
            if s1 is None:
                logger.error("Step 1 returned None from LLM")
                s1 = ""

            logger.debug(f"Step 1 output length: {len(s1)} chars")

        except Exception as e:
            logger.error(f"Step 1 (associations) failed: {e}")
            s1 = "1) [generation failed]"

        # Step 2: Generate imagery
        try:
            p2 = prompt_step2_imagery(text, s1)
            logger.debug(f"Step 2 prompt length: {len(p2)} chars")

            s2 = self.llm.complete(
                p2,
                temperature=self.cfg.temperature_step2,
                max_tokens=self.cfg.max_tokens_step2
            )

            # Handle None responses from LLM
            if s2 is None:
                logger.error("Step 2 returned None from LLM")
                s2 = ""

            logger.debug(f"Step 2 output length: {len(s2)} chars")

        except Exception as e:
            logger.error(f"Step 2 (imagery) failed: {e}")
            s2 = "1) [generation failed]"

        # Step 3: Generate final joke
        try:
            p3 = prompt_step3_final_joke(pattern, text, s2, retrieved_examples=retrieved_examples)
            logger.debug(f"Step 3 prompt length: {len(p3)} chars")

            s3 = self.llm.complete(
                p3,
                temperature=self.cfg.temperature_step3,
                max_tokens=self.cfg.max_tokens_step3
            )

            # Handle None responses from LLM and strip whitespace
            if s3 is None:
                logger.error("Step 3 returned None from LLM")
                s3 = ""

            s3 = s3.strip()
            logger.debug(f"Step 3 output length: {len(s3)} chars")

            # Validate we got something
            if not s3:
                logger.warning("Step 3 produced empty joke")
                s3 = "[Generation produced no output]"

        except Exception as e:
            logger.error(f"Step 3 (final joke) failed: {e}")
            s3 = "[Generation failed]"

        logger.info(f"Pipeline complete for '{text}' - joke length: {len(s3)} chars")

        if return_steps:
            return s3, {
                "step1_associations": s1,
                "step2_imagery": s2,
                "step3_joke": s3,
                "pattern": pattern,
                "input_text": text
            }

        return s3
