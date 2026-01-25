# ---------------------------------
# Local Ollama Joke Generator
# Pattern-aware (GOOD QUALITY)
# ---------------------------------
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import requests
from pathlib import Path
from patterns import JOKE_PATTERNS, PATTERN_DEFINITIONS
# ----------------------------
# CONFIG
# ----------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"
MAX_TOKENS = 120
TEMPERATURE = 0.9


BASE_DIR = Path(__file__).parent  # src/generation
INPUT_FILE = BASE_DIR / "task-a-en.tsv"
OUTPUT_FILE = BASE_DIR / "jokes_dataset_pip.txt"


TEST_MODE = True      # ← set False for full dataset
TEST_LINES = 10        # number of samples in test mode



# ----------------------------
# Prompt builder pipeline
# ---------------------------


def _prompt_step1_associations(text: str) -> str:
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

def _prompt_step2_imagery(text: str, assoc_text: str) -> str:
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


def _prompt_step3_final_joke(pattern: str, definition: str, text: str, imagery_text: str) -> str:
    return f"""
You are a professional comedy writer.

Joke style: {pattern}
Style definition: {definition}

Core humor rule (must apply):
- TRUTH first: a relatable observation.
- PRINCIPLE → SURPRISE:
  PRINCIPLE = expected rule/interpretation.
  SURPRISE  = twist that breaks expectation but still connects logically.
- Punchline must be last.

Input: {text}

Imagery candidates:
{imagery_text}

Constraints:
- If input is a headline, write a humorous comment or punchline (not a parody of the headline).
- If input is two words, use both naturally in your joke.

Task:
- Follow the constraints strictly.
- Build 1 joke (1–10 sentences).
- For each joke you should choose at least 3 imagery items from the list (by index).


Writing Instructions:
- Every joke starts from a truth implicitly — something real, relatable, or painfully honest.
- Build the joke around a clear principle (setup, rule, or perspective).
- Deliver a surprise that rewires the audience’s expectations.
- Keep tone committed: absurd logic must stay serious; sarcasm must stay polite.
- Be specific, not generic — concrete details increase funniness.
- Avoid clichés, especially "Why did X cross the road" or knock-knock jokes.
- Avoid generic templates like "Why did ...".
- No explanation, no labels.

Output format (exact):
- Output only the joke text.
- Do not include imagery items, prompts, all input text and pattern""".strip()

def call_LLM(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"].strip()


def generate_joke( pattern: str, text: str, *, show_steps: bool = False):
    """
    Runs a strict 3-step humor pipeline:
    1) Associations
    2) Positive / implicit imagery rewrite
    3) Final joke using Principle→Surprise + Truth + selected pattern

    Returns:
      (final_joke, steps_dict) if show_steps=True
      final_joke if show_steps=False
    """
    definition = PATTERN_DEFINITIONS.get(pattern, "")

    # Step 1
    p1 = _prompt_step1_associations(text)
    step1 = call_LLM(p1)

    # Step 2
    p2 = _prompt_step2_imagery(text, step1)
    step2 = call_LLM(p2)

    # Step 3 (final)
    p3 = _prompt_step3_final_joke(pattern, definition, text, step2)
    final_jokes = []

    for i in range(TEST_LINES):  # generate 10 jokes
        final_jokes.append(
            call_LLM(p3)
        )

    steps = {
        "step1_associations": step1,
        "step2_imagery": step2,
        "step3_final_joke": final_jokes,
    }

    if show_steps:
        return final_jokes, steps
    return final_jokes




# ----------------------------
# Load input
# ----------------------------

with open(INPUT_FILE, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# skip header
lines = lines[1:]

if TEST_MODE:
    lines = lines[:TEST_LINES]
    print(f"🧪 TEST MODE: using {len(lines)} lines\n")
else:
    print(f"🚀 FULL MODE: using {len(lines)} lines\n")

# ----------------------------
# Generate dataset
# ----------------------------

dataset = []

text = lines[0]  # oder irgendein einzelner Text, z.B. lines[idx]

for i, pattern in enumerate(JOKE_PATTERNS, start=1):
    print(f"Generating pattern {i}/{len(JOKE_PATTERNS)}: {pattern}...")

    jokes = generate_joke(pattern, text)

    dataset.append({
        "id": i,
        "input": text,
        "pattern": pattern,
        "jokes": jokes
    })

    print("✔ done\n")





# ----------------------------
# Save TXT
# ----------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in dataset:
        f.write(f"ID: {row['id']}\n")
        f.write(f"INPUT: {row['input']}\n")
        f.write(f"PATTERN: {row['pattern']}\n")
        f.write("JOKES:\n")
        for i, joke in enumerate(row["jokes"], start=1):
            f.write(f"{i}. {joke}\n")       
        f.write("=" * 60 + "\n")

print(f"\n✅ Saved {len(dataset)} jokes to '{OUTPUT_FILE}'")
