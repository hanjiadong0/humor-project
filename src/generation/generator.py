# ---------------------------------
# Local Ollama Joke Generator
# Pattern-aware (GOOD QUALITY)
# ---------------------------------

import requests

# ----------------------------
# CONFIG
# ----------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
MAX_TOKENS = 120
TEMPERATURE = 0.9

INPUT_FILE = "task-a-en.tsv"
OUTPUT_FILE = "jokes_dataset.txt"

TEST_MODE = True      # ← set False for full dataset
TEST_LINES = 3        # number of samples in test mode

# ----------------------------
# Joke patterns
# ----------------------------

JOKE_PATTERNS = [
    "Pun",
    "Absurd humor",
    "Mini funny story",
    "Sarcastic remark",
    "Mock breaking news",
    "Deadpan humor",
    "Irony",
    "Exaggeration",
    "Observational humor",
    "Fake expert explanation",
    "Self-aware joke",
    "Understatement"
]

# ----------------------------
# Pattern definitions (IMPORTANT)
# ----------------------------

PATTERN_DEFINITIONS = {
    "Pun":
        "Use wordplay, double meanings, or similar-sounding words.",

    "Absurd humor":
        "Make the joke intentionally illogical, surreal, or nonsense.",

    "Mini funny story":
        "Tell a very short story with a humorous ending.",

    "Sarcastic remark":
        "Use dry sarcasm or ironic commentary.",

    "Mock breaking news":
        "Present the joke as if it were a breaking news update.",

    "Deadpan humor":
        "State something ridiculous in a serious tone.",

    "Irony":
        "Highlight a contradiction or ironic outcome.",

    "Exaggeration":
        "Greatly overstate the importance or impact.",

    "Observational humor":
        "Make a relatable observation about the situation.",

    "Fake expert explanation":
        "Explain the situation using fake or ridiculous expertise.",

    "Self-aware joke":
        "Acknowledge that this is a joke being written.",

    "Understatement":
        "Deliberately downplay something serious."
}

# ----------------------------
# Prompt builder
# ----------------------------

def build_prompt(pattern, text):
    definition = PATTERN_DEFINITIONS.get(pattern, "")

    return f"""
You are a professional comedy writer.

Joke style:
{pattern}

Style definition:
{definition}

Input:
{text}

Instructions:
- Follow the style definition closely.
- Avoid generic "Why did" jokes.
- If input is a news headline, comment on it humorously.
- If input is two words, include both naturally.
- Write exactly one joke.
- Do not explain the joke.
- Output only the joke text.
""".strip()

# ----------------------------
# Ollama call
# ----------------------------

def generate_joke(prompt):
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

for i, text in enumerate(lines):
    pattern = JOKE_PATTERNS[i % len(JOKE_PATTERNS)]

    print(f"Generating {i+1}/{len(lines)} ({pattern})...")

    prompt = build_prompt(pattern, text)
    joke = generate_joke(prompt)

    dataset.append({
        "id": i + 1,
        "input": text,
        "pattern": pattern,
        "joke": joke
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
        f.write(f"JOKE: {row['joke']}\n")
        f.write("=" * 60 + "\n")

print(f"\n✅ Saved {len(dataset)} jokes to '{OUTPUT_FILE}'")
