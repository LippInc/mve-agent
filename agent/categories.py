"""Zero-remote-token task routing: regex/keyword category detection plus the
per-category prompt shape (terse-output instruction + hard completion cap).

Detection costs nothing on the proxy meter. Misrouting is benign by design:
every spec still answers the task, it just shapes verbosity — so a wrong
route costs tokens, never correctness. Default route is "factual" (the most
general shape).
"""

import re

# Ordered rules: first hit wins. Patterns run on the casefolded prompt.
_CODE_HINT = re.compile(r"```|\bdef\b|\bfunction\b|\bcode\b|\bpython\b|\bjavascript\b")
_RULES = [
    ("sentiment", re.compile(r"\bsentiment\b|classify (the )?(tone|review|feedback)")),
    ("summarization", re.compile(r"\bsummar|\bcondense\b|\btl;?dr\b|\bheadline\b|one[- ]sentence summary")),
    ("ner", re.compile(r"\bentit(y|ies)\b|named entit|extract .*(people|persons?|organi[sz]ations?|locations?|dates?)")),
    ("code-debug", re.compile(r"\b(bug|debug|broken|error|incorrect|wrong output|fails?)\b.*\b(fix|correct)|\b(fix|find)\b.*\bbug")),
    ("code-gen", re.compile(r"\b(write|implement|create|build)\b.*\b(function|method|program|script|class)\b")),
    ("math", re.compile(r"%|\bpercent|\$\d|\d+ ?(km|kg|miles|hours?|minutes?|dollars?|euros?)\b|how (many|much)|\baverage\b|\btotal\b|\bremain(s|ing)?\b|\bcost\b|\bprice\b|\brevenue\b|\bchange\b.*\bget\b|\bcalculate\b|\btimes\b|multiplied|divided by|\bplus\b|\bminus\b|\bsum of\b|what is \d[\d\s.,]*[+*/x×÷-]")),
    ("logic", re.compile(r"\bwho (owns?|is|wins?|finish|sits?|stands?)\b|each own|\bpuzzle\b|\bqueue\b|row of|\blabeled\b|\bconstraints?\b|if all .* are")),
]

# suffix: appended to the task prompt (bills ~6-12 input tokens; buys back
# hundreds of completion tokens). max_tokens: hard cap, sized with ~2x margin
# over the largest legitimate answer so truncation can't corrupt a correct one.
SPECS = {
    "math": {
        "suffix": "\n\nGive only the final numeric answer.",
        "max_tokens": 16,
    },
    "logic": {
        "suffix": "\n\nGive only the final answer, no explanation.",
        "max_tokens": 24,
    },
    "sentiment": {
        "suffix": "\n\nAnswer with the sentiment label and one brief reason.",
        "max_tokens": 48,
    },
    "summarization": {
        "suffix": "\n\nFollow the requested length and format exactly. Output only the summary itself.",
        "max_tokens": 112,
    },
    "ner": {
        "suffix": ("\n\nOutput only the entities, one per line, in the format: "
                   "entity - type. Include every person, organization, location, "
                   "date, and monetary amount. No other text."),
        "max_tokens": 112,
    },
    "factual": {
        "suffix": "\n\nAnswer directly and concisely in one or two short sentences.",
        "max_tokens": 72,
    },
    "code-gen": {
        "suffix": "\n\nOutput only the Python code, no explanation.",
        "max_tokens": 320,
    },
    "code-debug": {
        "suffix": "\n\nState the bug in one short line, then output only the corrected code.",
        "max_tokens": 320,
    },
}


def detect(prompt: str) -> str:
    p = prompt.casefold()
    has_code = bool(_CODE_HINT.search(p))
    for cat, rx in _RULES:
        if cat.startswith("code-") and not has_code:
            continue
        if rx.search(p):
            return cat
    return "factual"


def spec_for(category: str) -> dict:
    return SPECS.get(category, SPECS["factual"])
