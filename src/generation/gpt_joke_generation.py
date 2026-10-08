"""
GPT-OSS-120B Joke Generator
Generates creative jokes from word pairs using OpenAI's GPT-OSS-120B API
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI
import json

class GPT4JokeGenerator:
    """
    Generates creative jokes from word pairs using GPT-OSS-120B with optimized prompts.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        temperature: float = 0.9,
        top_p: float = 1.0,
        model: str = "openai/gpt-oss-120b",
        max_retries: int = 1
    ):
        """
        Initialize the GPT-OSS-120B joke generator.

        Args:
            api_key: NVIDIA API key
            temperature: Sampling temperature (0.7-1.0 recommended for creativity)
            top_p: Nucleus sampling parameter (default: 1.0)
            model: NVIDIA model to use (default: openai/gpt-oss-120b)
            max_retries: Maximum number of retry attempts on failure
        """
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key or os.getenv("NVIDIA_API_KEY")
        )
        self.temperature = temperature
        self.top_p = top_p
        self.model = model
        self.max_retries = max_retries

        # System prompt - establishes the AI's role and guidelines
        self.system_prompt = """You are a world-class comedy writer specializing in creative, unexpected humor. Your expertise includes:

• Wordplay & Puns: Creating clever double meanings and linguistic twists
• Absurdist Humor: Finding comedy in unexpected juxtapositions
• Observational Comedy: Highlighting the amusing aspects of everyday scenarios
• Conceptual Blending: Merging unrelated concepts in surprising ways

Your jokes should be:
✓ Original and creative (avoid clichés and obvious connections)
✓ Concise (1-3 sentences maximum)
✓ Family-friendly (no offensive, crude, or inappropriate content)
✓ Surprising (the punchline should be unexpected yet logical in hindsight)
✓ Self-contained (understandable without additional context)
✓ Naturally incorporate both words in a meaningful way

Focus on creating jokes that make people think "I never would have connected those ideas, but it's brilliant!"

Your goal is to find the most creative, unexpected, yet logical connections between the two words."""

    def _build_user_prompt(self, word1: str, word2: str, num_jokes: int = 10) -> str:
        """
        Constructs the user prompt for joke generation.
        Supports both word-pair mode and headline mode (word2="").

        Args:
            word1: First word (or headline text if word2 is empty)
            word2: Second word (empty string for headline mode)
            num_jokes: Number of jokes to generate

        Returns:
            Formatted prompt string
        """
        if word2:
            # Word-pair mode
            prompt = f"""Generate {num_jokes} creative, hilarious jokes that cleverly connect these two words:

Word 1: {word1}
Word 2: {word2}

Requirements:
1. Each joke MUST incorporate BOTH words naturally and meaningfully
2. Explore DIFFERENT comedic approaches across the {num_jokes} jokes:
   - Some should use wordplay or puns
   - Some should be absurdist or surreal
   - Some should be observational or relatable
   - Some should create unexpected conceptual blends
3. Avoid forced or obvious connections - find genuinely clever relationships
4. Make each joke distinct and unique from the others
5. Keep jokes concise (1-3 sentences each)
6. Ensure all jokes are appropriate and family-friendly

Think creatively about:
- What unexpected contexts could combine these words?
- What if one word had properties of the other?
- What professions, situations, or scenarios could involve both?
- What absurd scenarios would require both items?
- What puns or wordplay connect these concepts?

Format your response as a valid JSON object with a "jokes" key containing an array of strings:
{{"jokes": ["joke 1", "joke 2", "joke 3", ...]}}

Generate {num_jokes} unique, creative jokes now:"""
        else:
            # Headline mode
            prompt = f"""Generate {num_jokes} creative, hilarious jokes or humorous comments about this headline:

HEADLINE: "{word1}"

Requirements:
1. Write funny reactions, commentary, or punchlines inspired by the headline
2. Do NOT just rephrase the headline - add a comedic twist
3. Explore DIFFERENT comedic approaches:
   - Some should be witty one-liner reactions
   - Some should be absurdist takes on the situation
   - Some should be observational commentary
   - Some should use wordplay on key words in the headline
4. Keep jokes concise (1-3 sentences each)
5. Ensure all jokes are appropriate and family-friendly

Format your response as a valid JSON object with a "jokes" key containing an array of strings:
{{"jokes": ["joke 1", "joke 2", "joke 3", ...]}}

Generate {num_jokes} unique, creative jokes now:"""

        return prompt

    def generate_jokes(
        self,
        word1: str,
        word2: str,
        num_jokes: int = 10,
        verbose: bool = False
    ) -> List[str]:
        """
        Generate jokes from a word pair using GPT-OSS-120B with streaming.

        Args:
            word1: First word in the pair
            word2: Second word in the pair
            num_jokes: Number of jokes to generate (default: 10)
            verbose: If True, print streaming output in real-time

        Returns:
            List of generated joke strings

        Raises:
            Exception: If generation fails after max_retries attempts
        """
        user_prompt = self._build_user_prompt(word1, word2, num_jokes)

        for attempt in range(self.max_retries):
            try:
                # Use streaming completion
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=4096,
                    stream=True
                )

                # Collect streamed response
                full_content = ""
                reasoning_content = ""

                if verbose:
                    print("\n🤖 Model generating jokes (streaming)...\n")

                for chunk in completion:
                    if not getattr(chunk, "choices", None):
                        continue

                    # Handle reasoning content if present
                    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                    if reasoning:
                        reasoning_content += reasoning
                        if verbose:
                            print(reasoning, end="", flush=True)

                    # Handle main content
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content_chunk = chunk.choices[0].delta.content
                        full_content += content_chunk
                        if verbose:
                            print(content_chunk, end="", flush=True)

                if verbose:
                    print("\n\n✓ Generation complete, parsing response...\n")

                # Parse the complete response
                jokes = self._parse_response(full_content, num_jokes)

                # Validate we got enough jokes
                if jokes and len(jokes) >= int(num_jokes * 0.7):
                    return jokes[:num_jokes]
                elif jokes:
                    # Got some jokes but fewer than target
                    if attempt < self.max_retries - 1:
                        print(f"Attempt {attempt + 1}/{self.max_retries}: Only got {len(jokes)}/{num_jokes} jokes, retrying...")
                    else:
                        return jokes  # return what we have on last attempt
                else:
                    if attempt < self.max_retries - 1:
                        print(f"Attempt {attempt + 1}/{self.max_retries}: No jokes extracted, retrying...")

            except json.JSONDecodeError as e:
                print(f"Attempt {attempt + 1}/{self.max_retries}: JSON parsing error - {e}")
                if attempt == self.max_retries - 1:
                    # Last attempt - try fallback extraction
                    raise Exception(f"Failed to extract jokes!")

            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.max_retries}: Error - {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to generate jokes after {self.max_retries} attempts: {e}")

        raise Exception(f"Failed to generate {num_jokes} jokes after {self.max_retries} attempts")

    def _parse_response(self, content: str, num_jokes: int) -> List[str]:
        """
        Parse the model response to extract jokes.
        Tries JSON parsing first, then falls back to text extraction.

        Args:
            content: Raw response content from the model
            num_jokes: Target number of jokes

        Returns:
            List of joke strings
        """
        try:
            # Look for JSON in the content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                jokes_data = json.loads(json_str)
                jokes = self._extract_jokes_from_json(jokes_data)

                # Clean and validate
                jokes = [str(joke).strip() for joke in jokes if joke and len(str(joke).strip()) > 20]

                if len(jokes) > 0:
                    return jokes

        except (json.JSONDecodeError, ValueError):
            pass

        print('Failed to extract jokes from response')
        return []

    def _extract_jokes_from_json(self, jokes_data: dict) -> List[str]:
        """
        Extract jokes list from various JSON response formats.

        Args:
            jokes_data: Parsed JSON data from API response

        Returns:
            List of joke strings
        """
        # If the response is already a list
        if isinstance(jokes_data, list):
            return jokes_data

        # If it's a dict, try common key names
        if isinstance(jokes_data, dict):
            for key in ['jokes', 'responses', 'results', 'output', 'data', 'items']:
                if key in jokes_data and isinstance(jokes_data[key], list):
                    return jokes_data[key]

            # If no standard key found, try to find any list value
            for value in jokes_data.values():
                if isinstance(value, list) and len(value) > 0:
                    return value

        raise ValueError("Could not extract jokes list from JSON response")

    def generate_with_metadata(
        self,
        word1: str,
        word2: str,
        num_jokes: int = 10,
        verbose: bool = False
    ) -> Dict:
        """
        Generate jokes and return with generation metadata.

        Args:
            word1: First word in the pair
            word2: Second word in the pair
            num_jokes: Number of jokes to generate
            verbose: If True, print streaming output in real-time

        Returns:
            Dictionary containing:
                - jokes: List of generated joke strings
                - metadata: Dict with generation parameters and stats
        """
        jokes = self.generate_jokes(word1, word2, num_jokes, verbose=verbose)

        return {
            "jokes": jokes,
            "metadata": {
                "word_pair": [word1, word2],
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_requested": num_jokes,
                "num_generated": len(jokes),
                "generation_successful": len(jokes) >= int(num_jokes * 0.7)
            }
        }

    def batch_generate(
        self,
        word_pairs: List[tuple],
        num_jokes_per_pair: int = 10
    ) -> Dict[str, Dict]:
        """
        Generate jokes for multiple word pairs.

        Args:
            word_pairs: List of (word1, word2) tuples
            num_jokes_per_pair: Number of jokes to generate per pair

        Returns:
            Dictionary mapping "word1_word2" to generation results
        """
        results = {}

        for word1, word2 in word_pairs:
            key = f"{word1}_{word2}"
            try:
                results[key] = self.generate_with_metadata(word1, word2, num_jokes_per_pair)
                results[key]["status"] = "success"
            except Exception as e:
                results[key] = {
                    "jokes": [],
                    "metadata": {
                        "word_pair": [word1, word2],
                        "error": str(e)
                    },
                    "status": "failed"
                }

        return results