from openai import OpenAI
import re
import os

assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY not set"
client = OpenAI()




def build_eval_prompt(joke: str) -> str:
    return f"""
You are an expert humor evaluator.

Task:
Rate the following joke on a scale from 0 to 5.

Criteria:
- Truth / relatability
- Surprise / punchline strength
- Brevity and clarity
- Overall funniness

Scale:
0 = not funny at all
5 = extremely funny

Joke:
{joke}

Output ONLY a single float between 0 and 5. 
""".strip()




def rate_joke_openai(joke: str, model: str = "gpt-5", debug: bool = False) -> float:
    prompt = build_eval_prompt(joke)

    response = client.responses.create(
        model=model,
        input=prompt,

        # IMPORTANT: max_output_tokens includes reasoning tokens (GPT-5 is a reasoning model).
        # If this is too small, you get empty output_text → all 0.0 scores.
        max_output_tokens=256,

        # Reduce reasoning so it doesn't eat your token budget.
        reasoning={"effort": "low"},
    )

    text = (response.output_text or "").strip()

    # Debug print if model didn't return a clean number
    if debug and (not text or not re.search(r"[0-5](?:\.\d+)?", text)):
        print("RAW_OUTPUT_TEXT:", repr(text))

    # Extract first float in [0,5]
    m = re.search(r"\b([0-5](?:\.\d+)?)\b", text)
    score = float(m.group(1)) if m else 0.0

    return max(0.0, min(5.0, score))

def select_best_joke(jokes: list[str], model: str = "gpt-5"):
    best_joke = None
    best_score = -1.0
    scores = []

    for joke in jokes:
        score = rate_joke_openai(joke, model=model)
        scores.append((score, joke))

        if score > best_score:
            best_score = score
            best_joke = joke

    return best_joke, best_score, scores


# ---------- test run ----------
if __name__ == "__main__":
    jokes = [
            "Panamanian lawmakers' diplomatic trip to Taiwan has sparked a row with China over sovereignty concerns. It's like they kept the best deals for themselves - at the expense of everyone else, trying to charge us interest on their lack of communication.",
            "Panamanian lawmakers’ Taiwan trip sparks diplomatic row with China. I guess you could say they were flagging a major issue – after all, their flags were waving wildly in protest!",
            "Panamanian lawmakers went to Taiwan with high hopes of 'draft'ing a new diplomatic plan, but ended up getting stamped out by China.",
            "Panamanian lawmakers’ Taiwan trip sparks diplomatic row with China, But honestly, who needs diplomacy when you can just collect stamps like a tourist on their passport adventure?",
            "As Panamanian lawmakers returned from their Taiwan trip, one wondered if they'd found a new way to charge up China's diplomatic batteries. It seems those banners waved high were just drafts of what was to come – a charge of faux pas galore!",
            "Panamanian lawmakers' trip to Taiwan is a diplomatic disaster. I guess that's what happens when you try to charge into international relations without drafting a good plan.",
            "Panamanian lawmakers’ trip to Taiwan is a diplomatic faux pas. Looks like they got lost in translation – and also on the map of Asia!",
            "Panamanian lawmakers' diplomatic faux pas with China's flags are a real nail-biter. It seems their trip was just a 'draft' to stir up trouble.",
            "Panamanian lawmakers’ Taiwan trip sparks diplomatic row with China, It seems like they wanted to \"draft\" a new path in diplomacy.",
            "Panamanian lawmakers' diplomatic trip to Taiwan was a charged affair, but I guess you could say it was a real 'draft' for world peace. After all, who needs a peaceful resolution when you can have a few extra stamps on your passport?"
            "China's got a diplomatic 'draft' to pull out of this situation."
        ]

    best, best_score, all_scores = select_best_joke(jokes)

    print("\n--- All scores ---")
    for s, j in sorted(all_scores, reverse=True, key=lambda x: x[0]):
        print(f"{s}/5  -  {j}")

    print("\n--- Winner ---")
    print(f"{best_score}/5  -  {best}")
