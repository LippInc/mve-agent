"""Zero-remote-token task routing: regex/keyword category detection plus the
per-category prompt shape (terse-output instruction + hard completion cap).

Detection costs nothing on the proxy meter. Misrouting is benign by design:
every spec still answers the task, it just shapes verbosity — so a wrong
route costs tokens, never correctness. Default route is "factual" (the most
general shape).
"""

import re

# Detection signals, run on the casefolded prompt. Priority order lives in
# detect(). Misrouting INTO an answer-shape-changing category (math's bare
# number, code's code-only output) is the expensive direction, so math needs
# numeric evidence and the code rules need a code hint before they can fire.
_CODE_HINT = re.compile(r"```|\bdef\b|\bfunction\b|\bcode\b|\bpython\b|\bjavascript\b")
_SENTIMENT_RX = re.compile(r"\bsentiment\b|classify (the )?(tone|review|feedback)")
_SUMMAR_RX = re.compile(r"\bsummar|\bcondense\b|\btl;?dr\b|\bheadline\b|one[- ]sentence summary")
_NER_RX = re.compile(
    r"\bentit(y|ies)\b|named entit"
    r"|\b(extract|identify|list|find|pull out|name)\b[\s\S]{0,80}?"
    r"\b(people|persons?|organi[sz]ations?|locations?|dates?|monetary|names)\b"
    r"|who and what (is|are) mentioned")
_BUG_RX = re.compile(r"\bbug(s|gy)?\b|\bbroken\b|\berror\b|\bincorrect\b"
                     r"|wrong (output|result|value|answer)"
                     r"|(doesn'?t|does not|won'?t) work|fails? (on|when|for)")
_FIXVERB_RX = re.compile(r"\bfix\b|\bcorrect\b|\brepair\b|\bdebug\b")
_CODEGEN_RX = re.compile(
    r"\b(write|implement|create|build|design|provide|develop|give)\b[\s\S]{0,60}?"
    r"\b(function|method|program|script|class|solution|algorithm|code)\b")
_MATH_KW_RX = re.compile(
    r"%|\bpercent|\$\d|\d+ ?(km|kg|miles|hours?|minutes?|dollars?|euros?)\b"
    r"|how (many|much)|\baverage\b|\btotal\b|\bremain(s|ing)?\b|\bcost\b"
    r"|\bprice\b|\brevenue\b|\bchange\b|\bcalculate\b|\btimes\b|multiplied"
    r"|divided by|\bplus\b|\bminus\b|\bsum of\b")
_ARITH_RX = re.compile(r"%|\bpercent|\$\d|[+*/×÷^]|\btimes\b|multiplied|divided by|\bplus\b|\bminus\b")
_NUM_RX = re.compile(r"\d+(?:\.\d+)?")
# Two-part questions ("how many X, and which team...") must keep a prose
# shape: a bare PoT number would silently drop the second half of the answer.
_COMPOUND_Q_RX = re.compile(r",\s*and\s+(wh(ich|o|at|ere|en|y)|is|are|how)\b")
_LOGIC_RX = re.compile(
    r"\bwho (owns?|is|wins?|finish|sits?|stands?)\b|each own|\bpuzzle\b"
    r"|\bqueue\b|row of|\blabeled\b|\bconstraints?\b|if all .* are")


def _is_math(p: str) -> bool:
    if _COMPOUND_Q_RX.search(p) or p.count("?") >= 2:
        return False
    nums = _NUM_RX.findall(p)
    if len(nums) >= 2 and _MATH_KW_RX.search(p):
        return True
    return len(nums) >= 1 and bool(_ARITH_RX.search(p))

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
        "max_tokens": 176,
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
    if _SENTIMENT_RX.search(p):
        return "sentiment"
    if _SUMMAR_RX.search(p):
        return "summarization"
    if _NER_RX.search(p):
        return "ner"
    if has_code and _BUG_RX.search(p) and _FIXVERB_RX.search(p):
        return "code-debug"
    if has_code and _CODEGEN_RX.search(p):
        return "code-gen"
    if _is_math(p):
        return "math"
    if _LOGIC_RX.search(p):
        return "logic"
    return "factual"


def spec_for(category: str) -> dict:
    return SPECS.get(category, SPECS["factual"])
