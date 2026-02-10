
# ---------------------------------
# Humor Patterns (Principle ↔ Surprise)
# ---------------------------------

# Core idea:
# 1) PRINCIPLE: establish a clear premise / rule / expectation.
# 2) SURPRISE: twist it (misdirection, reversal, escalation, literalism, etc.)
# 3) COMMIT: treat the twist seriously and end on the strongest word.

JOKE_PATTERNS = [
    "Pun (re-interpretation)",
    "Absurd Logic (serious in a weird world)",
    "Mini Story (setup → turn → punch)",
    "Sarcasm (meaning inversion)",
    "Mock Breaking News (formal frame, silly content)",
    "Deadpan (flat delivery of absurdity)",
    "Irony (expectation vs outcome)",
    "Exaggeration / Heightening (bigger, bigger, biggest)",
    "Observational (relatable truth → twist)",
    "Fake Expert (confident nonsense logic)",
    "Self-aware (meta rule-break)",
    "Understatement (tiny reaction to huge thing)",
    "Rule of Three (pattern → pattern → twist)",
    "Literalism (take figurative language literally)",
    "Status Reversal (who's 'above' flips)"
]

PATTERN_DEFINITIONS = {
    # -----------------
    # Word-level surprise
    # -----------------
    "Pun (re-interpretation)": {
        "principle": "Build one sentence that strongly implies Meaning A of a word/phrase.",
        "surprise": "End with a word/phrase that forces Meaning B (double meaning / sound-alike / re-parse).",
        "setup_steps": [
            "Choose target word with ambiguity (bank, interest, draft, charge, scale...).",
            "Write setup that locks audience into Meaning A.",
            "Punchline re-frames into Meaning B with the final word."
        ],
        "quality_checks": [
            "Punchline is LAST and short.",
            "Setup is not vague; it clearly pushes Meaning A.",
            "Reinterpretation is immediate (no explanation needed)."
        ],
        "prompt_template": (
            "Write a 1–2 line pun. First line strongly implies meaning A. "
            "Final word forces meaning B. No explanation."
        )
    },

    # -----------------
    # World-rule surprise
    # -----------------
    "Absurd Logic (serious in a weird world)": {
        "principle": "Define a bizarre but consistent rule of the world (the 'law').",
        "surprise": "Apply that rule to a normal situation with total seriousness.",
        "setup_steps": [
            "State the weird rule implicitly (through action), not as an essay.",
            "Use a normal context (office, gym, dating, university, pharmacy...).",
            "Punchline: the rule’s consequence hits unexpectedly."
        ],
        "quality_checks": [
            "The absurdity is consistent (same rule).",
            "Tone is serious (that’s the contrast).",
            "Punchline reveals the strongest consequence."
        ],
        "prompt_template": (
            "Create a joke in a world with one weird rule. Show it in action in a normal setting. "
            "Dead-serious tone. End on the consequence."
        )
    },

    # -----------------
    # Narrative surprise
    # -----------------
    "Mini Story (setup → turn → punch)": {
        "principle": "Setup: a clear situation with a believable goal.",
        "surprise": "Turn: introduce an unexpected constraint or interpretation. Punch: snap ending.",
        "setup_steps": [
            "Setup (1 sentence): who/where/goal.",
            "Turn (1 sentence): misdirection / new info / misunderstanding.",
            "Punch (short): the reveal, last word strongest."
        ],
        "quality_checks": [
            "No extra sentences after the punch.",
            "The turn changes the meaning of the setup.",
            "Specific details (not generic)."
        ],
        "prompt_template": (
            "Tell a 2–3 sentence micro-story. Setup → turn → punch. "
            "Punchline is the final short sentence."
        )
    },

    # -----------------
    # Attitude-based surprise
    # -----------------
    "Sarcasm (meaning inversion)": {
        "principle": "Audience can see the true evaluation (bad, annoying, ridiculous).",
        "surprise": "Speaker says the opposite with a confident, polite, 'of course' tone.",
        "setup_steps": [
            "Pick a clear annoyance/disaster.",
            "State praise that is obviously false.",
            "Add one tiny detail that makes it sharper."
        ],
        "quality_checks": [
            "Don’t overdo; keep it tight.",
            "Make the underlying truth obvious from context.",
            "Use 'too polite' language for contrast."
        ],
        "prompt_template": (
            "Write one sarcastic line about a clearly bad situation. "
            "Say the opposite of what you mean. Add one small specific detail."
        )
    },

    "Deadpan (flat delivery of absurdity)": {
        "principle": "Normal statement structure and calm voice.",
        "surprise": "Content is ridiculous, delivered like a weather report.",
        "setup_steps": [
            "Start like a factual report.",
            "Insert the absurd fact as if routine.",
            "Optional: add a practical follow-up that treats it as normal."
        ],
        "quality_checks": [
            "No emojis, no 'haha'.",
            "Absurd element is concrete, not abstract.",
            "Follow-up makes it funnier by commitment."
        ],
        "prompt_template": (
            "Deliver an absurd fact in a completely serious tone. "
            "Add a calm practical follow-up. 1–2 lines."
        )
    },

    # -----------------
    # Frame-based surprise
    # -----------------
    "Mock Breaking News (formal frame, silly content)": {
        "principle": "Use the serious 'breaking news' format.",
        "surprise": "Report something trivial or absurd with urgent authority.",
        "setup_steps": [
            "Headline tone ('Breaking:', 'In a stunning development...').",
            "Absurd/trivial event.",
            "Add one 'expert quote' or 'official statement' line."
        ],
        "quality_checks": [
            "The contrast is the joke: serious voice vs silly event.",
            "Keep it short like an alert.",
            "End with the funniest official-sounding phrase."
        ],
        "prompt_template": (
            "Write a breaking news alert about something ridiculous. "
            "Serious tone, 2–3 lines max, include one 'official statement'."
        )
    },

    "Fake Expert (confident nonsense logic)": {
        "principle": "Speaker sounds like a professional explaining a system.",
        "surprise": "The logic is internally structured but obviously wrong/silly.",
        "setup_steps": [
            "Choose an everyday topic.",
            "Invent a bogus 'theory' with 1–2 technical terms.",
            "End with a confident conclusion that’s hilariously impractical."
        ],
        "quality_checks": [
            "Structure feels like a real explanation (steps, terms).",
            "Nonsense stays consistent (same fake framework).",
            "Punchline is the confident conclusion."
        ],
        "prompt_template": (
            "Explain a normal thing with fake expertise and invented terms. "
            "Make it sound rigorous. End with a ridiculous confident conclusion."
        )
    },

    # -----------------
    # Expectation math
    # -----------------
    "Exaggeration / Heightening (bigger, bigger, biggest)": {
        "principle": "Start at a believable complaint/claim.",
        "surprise": "Escalate beyond reason in 2–3 steps.",
        "setup_steps": [
            "Step 1: normal.",
            "Step 2: bigger.",
            "Step 3: absurd maximum (but logically connected)."
        ],
        "quality_checks": [
            "Escalation is smooth (each step follows).",
            "Final step is the wildest and last.",
            "Specificity beats generic exaggeration."
        ],
        "prompt_template": (
            "Write a 3-beat escalation: normal → bigger → absurd. "
            "All connected. End on the biggest beat."
        )
    },

    "Understatement (tiny reaction to huge thing)": {
        "principle": "A big dramatic event is clearly implied.",
        "surprise": "Reaction is comically small/calm.",
        "setup_steps": [
            "State the huge event plainly.",
            "Respond like it’s a mild inconvenience.",
            "Optional: add a small polite action (tea, email, checklist)."
        ],
        "quality_checks": [
            "The gap (huge vs tiny) is clear.",
            "Tone stays calm throughout.",
            "No explaining why it’s funny."
        ],
        "prompt_template": (
            "Describe a big dramatic problem. Respond with a tiny polite reaction. "
            "1–2 lines."
        )
    },

    # -----------------
    # Pattern weapon
    # -----------------
    "Rule of Three (pattern → pattern → twist)": {
        "principle": "First two beats establish a predictable pattern.",
        "surprise": "Third beat breaks the pattern sharply but still fits the theme.",
        "setup_steps": [
            "Beat 1: expected.",
            "Beat 2: expected (same structure).",
            "Beat 3: twist (different category/angle)."
        ],
        "quality_checks": [
            "First two beats are parallel (same rhythm).",
            "Third is clearly different and strongest.",
            "Keep it punchy; avoid long clauses."
        ],
        "prompt_template": (
            "Write a rule-of-three joke: A, A, then twist. "
            "Make the first two parallel and the third surprising."
        )
    },

    "Literalism (take figurative language literally)": {
        "principle": "Use a common idiom/metaphor people know.",
        "surprise": "Interpret it literally with practical consequences.",
        "setup_steps": [
            "Pick idiom (e.g., 'break the ice', 'hit the books').",
            "Show literal action.",
            "End on a realistic side-effect."
        ],
        "quality_checks": [
            "Idiom is widely known (avoid obscure).",
            "Literal consequence is concrete.",
            "Punchline is the side-effect."
        ],
        "prompt_template": (
            "Take a common idiom literally. Show the literal action and its consequence. "
            "2 lines max."
        )
    },

    "Status Reversal (who's 'above' flips)": {
        "principle": "Establish a clear hierarchy (examples:  boss/employee, teacher/student, human/pet).",
        "surprise": "Flip who has control/competence.",
        "setup_steps": [
            "Show the normal hierarchy expectation.",
            "Reveal the opposite competence/control.",
            "End with a short 'new rule' line."
        ],
        "quality_checks": [
            "Hierarchy is obvious from the first line.",
            "Flip is clean and immediate.",
            "Finish on the line that proves the reversal."
        ],
        "prompt_template": (
            "Set up a clear hierarchy, then flip it. "
            "Keep it short and end with the line that proves the reversal."
        )
    },

    # -----------------
    # Situational surprise
    # -----------------
    "Irony (expectation vs outcome)": {
        "principle": "Set up a clear expectation about what should happen.",
        "surprise": "Reveal the opposite outcome in an unexpected way.",
        "setup_steps": [
            "Establish what's expected (expertise, preparation, goal).",
            "Show the ironic opposite result.",
            "End on the contrast."
        ],
        "quality_checks": [
            "The expectation is clear and believable.",
            "The ironic twist is immediate and sharp.",
            "No need to explain why it's ironic."
        ],
        "prompt_template": (
            "Set up a clear expectation, then reveal the ironic opposite. "
            "Keep it tight. End on the strongest contrast."
        )
    },

    "Observational (relatable truth → twist)": {
        "principle": "Start with a universally relatable observation or truth.",
        "surprise": "Add an unexpected angle or exaggerated conclusion.",
        "setup_steps": [
            "Pick something everyone experiences (traffic, phones, meetings...).",
            "State the relatable truth plainly.",
            "Twist with an absurd exaggeration or unexpected perspective."
        ],
        "quality_checks": [
            "The observation is genuinely relatable.",
            "The twist adds surprise, not just agreement.",
            "End with the most exaggerated or absurd element."
        ],
        "prompt_template": (
            "Start with a relatable observation everyone knows. "
            "Then twist it with an unexpected angle or absurd conclusion. 2-3 lines."
        )
    },

    "Self-aware (meta rule-break)": {
        "principle": "Acknowledge the joke-telling format itself.",
        "surprise": "Break the fourth wall or comment on the joke's own structure.",
        "setup_steps": [
            "Start like a normal joke setup.",
            "Acknowledge you're telling a joke or that the setup is artificial.",
            "End by commenting on or subverting the joke format itself."
        ],
        "quality_checks": [
            "The meta-awareness is explicit, not subtle.",
            "The rule-break feels deliberate and clever.",
            "Works best as a surprise ending."
        ],
        "prompt_template": (
            "Tell a joke that acknowledges it's a joke. "
            "Break the fourth wall. Comment on your own setup or punchline structure."
        )
    }
}
