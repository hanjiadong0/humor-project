"""
LLM Humor Scorer
Evaluates joke quality on a scale of 1-10 using any OpenAI-compatible LLM API.
Default backend: NVIDIA NIM (GPT-OSS-120B). Swappable via base_url and model params.
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI
import json
import re


class LLMHumorScorer:
    """
    Evaluates joke quality using an LLM with detailed scoring criteria.
    Returns scores on a scale of 1-10.

    Backend is swappable: any OpenAI-compatible API can be used by changing
    base_url and model parameters.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        temperature: float = 0.3,
        model: str = "openai/gpt-oss-120b",
        max_retries: int = 2
    ):
        """
        Initialize the LLM humor scorer.

        Args:
            api_key: API key (defaults to NVIDIA_API_KEY env var)
            base_url: OpenAI-compatible API endpoint (default: NVIDIA NIM)
            temperature: Low temperature for consistent scoring (0.2-0.4 recommended)
            model: Model ID to use (default: openai/gpt-oss-120b)
            max_retries: Maximum number of retry attempts on failure
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("NVIDIA_API_KEY")
        )
        self.temperature = temperature
        self.model = model
        self.max_retries = max_retries

        # System prompt for consistent humor evaluation
        self.system_prompt = """You are an expert comedy critic and humor analyst with deep knowledge of comedic theory, wordplay, and joke structure. Your role is to objectively evaluate jokes based on multiple criteria.

Evaluation Criteria (1-10 scale):

1. **Creativity & Originality** (1-10)
   - Is the joke fresh and unexpected?
   - Does it avoid clichés and obvious connections?
   - Is the concept novel and inventive?

2. **Word Integration** (1-10)
   - Are both words incorporated naturally?
   - Do the words feel essential to the joke?
   - Is the connection between words clever?

3. **Humor Impact** (1-10)
   - How funny is the punchline?
   - Does it deliver a genuine laugh or smile?
   - Is the surprise element effective?

4. **Structure & Flow** (1-10)
   - Is the setup clear and concise?
   - Does the punchline land well?
   - Is the timing and pacing good?

5. **Cleverness** (1-10)
   - Does it make you think?
   - Is there wordplay, double meaning, or wit?
   - Does it reward re-reading?

Your overall score should reflect the joke's total comedic quality, weighing all criteria appropriately."""

    def _build_scoring_prompt(self, joke: str, word1: str, word2: str) -> str:
        """
        Constructs the scoring prompt for a single joke.

        Args:
            joke: The joke to evaluate
            word1: First word in the pair (or headline text if word2 is empty)
            word2: Second word in the pair (empty string for headline mode)

        Returns:
            Formatted prompt string
        """
        if word2:
            context_line = f'Evaluate this joke that was created using the word pair: "{word1}" and "{word2}"'
        else:
            context_line = f'Evaluate this joke that was created as commentary on the headline: "{word1}"'

        prompt = f"""{context_line}

JOKE TO EVALUATE:
"{joke}"

TASK:
Provide a comprehensive evaluation including:
1. Scores for each criterion (1-10 scale)
2. Brief justification for each score
3. An overall score (1-10 scale, can include decimals like 7.5)

The overall score should be a weighted average reflecting:
- Creativity & Originality: 25%
- Word Integration: 20%
- Humor Impact: 30%
- Structure & Flow: 15%
- Cleverness: 10%

Respond ONLY with a valid JSON object in this exact format:
{{
  "creativity": <score 1-10>,
  "word_integration": <score 1-10>,
  "humor_impact": <score 1-10>,
  "structure": <score 1-10>,
  "cleverness": <score 1-10>,
  "overall_score": <weighted score 1-10, can be decimal>,
  "justification": "Brief explanation of the overall score"
}}

Be critical but fair. Reserve scores above 8.0 for truly exceptional jokes."""

        return prompt

    def score_joke(
        self,
        joke: str,
        word1: str = "",
        word2: str = "",
        verbose: bool = False
    ) -> float:
        """
        Score a single joke using the configured LLM backend.

        Args:
            joke: The joke to evaluate
            word1: First word in the pair (optional, for context)
            word2: Second word in the pair (optional, for context)
            verbose: If True, print detailed scoring information

        Returns:
            Overall humor score (1-10 scale)
        """
        scoring_prompt = self._build_scoring_prompt(joke, word1, word2)

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": scoring_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1024,
                    stream=False
                )

                response_content = completion.choices[0].message.content

                if verbose:
                    print(f"\n📊 Raw scoring response:\n{response_content}\n")

                # Parse the response
                score_data = self._parse_scoring_response(response_content)

                if verbose:
                    print(f"🎯 Detailed Scores:")
                    print(f"   Creativity: {score_data['creativity']}/10")
                    print(f"   Word Integration: {score_data['word_integration']}/10")
                    print(f"   Humor Impact: {score_data['humor_impact']}/10")
                    print(f"   Structure: {score_data['structure']}/10")
                    print(f"   Cleverness: {score_data['cleverness']}/10")
                    print(f"   Overall: {score_data['overall_score']}/10")
                    print(f"   Justification: {score_data['justification']}\n")

                return float(score_data['overall_score'])

            except Exception as e:
                if verbose:
                    print(f"⚠️ Attempt {attempt + 1}/{self.max_retries}: Error - {e}")

                if attempt == self.max_retries - 1:
                    # Return a neutral score on complete failure
                    if verbose:
                        print(f"❌ Failed to score joke, returning default score of 5.0")
                    return 5.0

        return 5.0

    def _parse_scoring_response(self, content: str) -> Dict:
        """
        Parse the scoring response to extract scores and justification.

        Args:
            content: Raw response content from the model

        Returns:
            Dictionary with score breakdown

        Raises:
            ValueError: If parsing fails
        """
        # Try JSON parsing first
        try:
            # Look for JSON in the content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                score_data = json.loads(json_str)

                # Validate required fields
                required_fields = ['creativity', 'word_integration', 'humor_impact',
                                 'structure', 'cleverness', 'overall_score']

                if all(field in score_data for field in required_fields):
                    # Ensure scores are in valid range
                    for field in required_fields:
                        score_data[field] = max(1.0, min(10.0, float(score_data[field])))

                    return score_data

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            pass

        # Fallback: extract numbers from text
        return self._fallback_score_extraction(content)

    def _fallback_score_extraction(self, content: str) -> Dict:
        """
        Fallback method to extract scores if JSON parsing fails.

        Args:
            content: Raw response content

        Returns:
            Dictionary with estimated scores
        """
        # Try to find any number that looks like a score
        numbers = re.findall(r'\b([1-9]|10)(?:\.\d+)?\b', content)

        if numbers:
            scores = [float(n) for n in numbers[:6]]  # Take first 6 numbers found

            # Pad if needed
            while len(scores) < 6:
                scores.append(5.0)

            return {
                'creativity': scores[0],
                'word_integration': scores[1],
                'humor_impact': scores[2],
                'structure': scores[3],
                'cleverness': scores[4],
                'overall_score': scores[5],
                'justification': 'Fallback extraction used'
            }

        # Complete fallback - return neutral scores
        return {
            'creativity': 5.0,
            'word_integration': 5.0,
            'humor_impact': 5.0,
            'structure': 5.0,
            'cleverness': 5.0,
            'overall_score': 5.0,
            'justification': 'Unable to parse response, using default scores'
        }

    def _build_batch_scoring_prompt(self, jokes: List[str], word1: str, word2: str) -> str:
        """Build a single prompt that scores all jokes at once."""
        jokes_block = ""
        for i, joke in enumerate(jokes, 1):
            jokes_block += f'\nJOKE {i}:\n"{joke}"\n'

        if word2:
            context_line = f'Evaluate each of the following {len(jokes)} jokes created using the word pair: "{word1}" and "{word2}"'
        else:
            context_line = f'Evaluate each of the following {len(jokes)} jokes created as commentary on the headline: "{word1}"'

        return f"""{context_line}

{jokes_block}
TASK:
For EACH joke, provide an overall score (1-10, decimals allowed) using these weights:
- Creativity & Originality: 25%
- Word Integration: 20%
- Humor Impact: 30%
- Structure & Flow: 15%
- Cleverness: 10%

Be critical but fair. Reserve scores above 8.0 for truly exceptional jokes.

Respond ONLY with a valid JSON array of exactly {len(jokes)} objects, one per joke, in order:
[
  {{"joke_number": 1, "overall_score": <score>}},
  {{"joke_number": 2, "overall_score": <score>}},
  ...
]"""

    def _parse_batch_response(self, content: str, num_jokes: int) -> List[float]:
        """Parse batch scoring response. Returns list of scores or None on failure."""
        try:
            arr_start = content.find('[')
            arr_end = content.rfind(']') + 1
            if arr_start != -1 and arr_end > arr_start:
                arr = json.loads(content[arr_start:arr_end])
                if isinstance(arr, list) and len(arr) == num_jokes:
                    scores = []
                    for item in arr:
                        s = float(item.get('overall_score', item.get('score', 5.0)))
                        scores.append(max(1.0, min(10.0, s)))
                    return scores
        except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError):
            pass
        return None

    def score_jokes(
        self,
        jokes: List[str],
        word1: str = "",
        word2: str = "",
        verbose: bool = False
    ) -> List[float]:
        """
        Score multiple jokes in a single API call (batch mode).
        Falls back to per-joke scoring if batch parsing fails.

        Args:
            jokes: List of jokes to evaluate
            word1: First word in the pair (optional)
            word2: Second word in the pair (optional)
            verbose: If True, print detailed scoring for each joke

        Returns:
            List of overall scores (1-10 scale)
        """
        if not jokes:
            return []

        # Try batch scoring (1 API call for all jokes)
        batch_prompt = self._build_batch_scoring_prompt(jokes, word1, word2)
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": batch_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=2048,
                    stream=False
                )
                response = completion.choices[0].message.content
                scores = self._parse_batch_response(response, len(jokes))
                if scores is not None:
                    if verbose:
                        for i, (joke, score) in enumerate(zip(jokes, scores), 1):
                            print(f"  [{i}] {score:.1f}/10 - {joke[:60]}...")
                    return scores
            except Exception as e:
                if verbose:
                    print(f"  Batch attempt {attempt + 1} failed: {e}")

        # Fallback: score one by one
        if verbose:
            print("  Batch scoring failed, falling back to per-joke scoring...")
        scores = []
        for joke in jokes:
            score = self.score_joke(joke, word1, word2, verbose=verbose)
            scores.append(score)
        return scores

    def score_with_details(
        self,
        joke: str,
        word1: str = "",
        word2: str = ""
    ) -> Dict:
        """
        Score a joke and return detailed breakdown.

        Args:
            joke: The joke to evaluate
            word1: First word in the pair
            word2: Second word in the pair

        Returns:
            Dictionary with detailed scoring breakdown
        """
        scoring_prompt = self._build_scoring_prompt(joke, word1, word2)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": scoring_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1024,
                stream=False
            )

            response_content = completion.choices[0].message.content
            score_data = self._parse_scoring_response(response_content)

            return {
                "joke": joke,
                "word_pair": [word1, word2],
                "scores": score_data,
                "overall_score": score_data['overall_score']
            }

        except Exception as e:
            return {
                "joke": joke,
                "word_pair": [word1, word2],
                "scores": {
                    "creativity": 5.0,
                    "word_integration": 5.0,
                    "humor_impact": 5.0,
                    "structure": 5.0,
                    "cleverness": 5.0,
                    "overall_score": 5.0,
                    "justification": f"Error during scoring: {str(e)}"
                },
                "overall_score": 5.0,
                "error": str(e)
            }