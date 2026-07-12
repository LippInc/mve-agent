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
_CODE_HINT = re.compile(r"```|\bdef\b|\bfunction\b|\bcode\b|\bpython\b|\bjavascript\b"
                        # bare noun/verb hints _CODEGEN_RX itself anticipates
                        # ("write a script that...") — safe because the code
                        # rules still need their own verb+noun/bug evidence
                        # to fire (hunt #11); bare "method" stays out (too
                        # common as a prose word).
                        r"|\bscript\b|\bprogram\b|\balgorithm\b|\bimplement\b")
_SENTIMENT_RX = re.compile(
    r"\bsentiment\b|classify (the )?(tone|review|feedback)"
    r"|positive,? (or )?negative|positive,? negative,? or neutral"
    # tone/mood only with a review-ish object: "tone of this passage" is a
    # summarization ask, "tone of voice" advice is factual (critic-verified)
    r"|\b(tone|mood) of (this|the) (review|message|feedback|comment|email|post|reply)"
    r"|tone comes across|rate the tone|judge the (tone|sentiment)"
    r"|how (positive|negative|happy|satisfied)"
    r"|how does (the|this) (customer|reviewer|writer|author) feel"
    # valence-pair phrasings without the word "sentiment" fell to the
    # unguarded factual path (hunt #15); the pair form keeps lone
    # "favorable" (weather, terms) from hijacking factual asks
    r"|favou?rable,? (or )?unfavou?rable"
    r"|happy or upset|pleased or displeased|satisfied or dissatisfied"
    r"|upbeat or downbeat")
# "headline" only with an authoring verb: "extract entities from this
# headline" is NER, "write a headline for..." is summarization-shaped.
_SUMMAR_RX = re.compile(
    r"\bsummar|\bcondense\b|\btl;?dr\b|\bboil (this|it) down\b|\bgist of\b"
    r"|(write|give|create|craft|come up with)[\s\S]{0,24}?\bheadline\b"
    r"|one[- ]sentence summary"
    # word-limit framings signal summarization ONLY as the leading
    # instruction ("In at most 15 words, what is this saying?") — anywhere
    # else they are an answer-length constraint on another category and must
    # not hijack it (critic-verified: "Solve 12*8. Answer in 3 words or
    # fewer." must stay math)
    r"|^in (at most |no more than |under |fewer than )?\d+ words"
    r"|what (is|are) (this|these|it) saying"
    r"|^in (a single|one) (sentence|line)\b"
    # "compress/distill the following ..." and "<verb> ... into exactly N
    # sentences" are summarization asks _SUMMAR_RX missed (refresh-gauntlet
    # 2026-07-10: "compress the following into exactly two sentences" routed
    # to math and shipped a bare number). Verb-coupled + "sentences?" keeps
    # code prompts ("compress a list into a string in python") out.
    r"|\b(compress|distill)\b[\s\S]{0,24}?\b(passage|text|article|paragraph"
    r"|report|story|following)\b"
    r"|\b(compress|condense|distill|shorten|reduce|rewrite|rephrase)\b"
    r"[\s\S]{0,60}?\binto (exactly )?(one|two|three|four|five|\d+) sentences?\b")
_NER_RX = re.compile(
    r"\bentit(y|ies)\b|named entit"
    r"|\b(extract|identify|list|find|pull out|name|tag)\b[\s\S]{0,80}?"
    r"\b(people|persons?|organi[sz]ations?|locations?|dates?|monetary|names"
    r"|compan(y|ies)|firms?|cities|countries|brands?|products?"
    r"|nationalit(y|ies)|places|proper nouns?)\b"
    r"|who and what (is|are) mentioned"
    # question-phrased NER asks ("Which people ... are mentioned/named in
    # the text?") missed the verb branch and fell through to factual, which
    # has zero entity shape-checking (fresh-124 nv2-ner-L2-1 shipped a
    # hallucinated set through exactly this hole)
    r"|\b(which|what|who)\b[\s\S]{0,60}?\b(people|persons?"
    r"|organi[sz]ations?|locations?|compan(y|ies)|places)\b[\s\S]{0,60}?"
    r"\b(mentioned|named|referenced|appear)")
_SCIENCE_FACT_RX = re.compile(
    r"chemical symbol|atomic number|periodic table")
# Counting questions ("find the number of people who...") want a number, not
# an entity list — they must not trip the NER verb-branch.
_COUNT_Q_RX = re.compile(r"\bhow many\b|\bnumber of\b|\bcount of\b")
_BUG_RX = re.compile(r"\bbug(s|gy)?\b|\bbroken\b|\berror\b|\bincorrect\b"
                     r"|wrong (output|result|value|answer)"
                     r"|(doesn'?t|does not|won'?t|isn'?t) work|fails? (on|when|for)"
                     r"|\bdefects?\b|\bflaw(s|ed)?\b")
_CRASH_RX = re.compile(r"\bcrash(es|ed|ing)?\b|\bthrows?\b|\brais(es|ed|ing)\b"
                       r"|\bexception\b|\btraceback\b|stack trace")
# Inline code on the page (not just the word "function") — evidence there is
# an existing implementation to debug.
_ACTUAL_CODE_RX = re.compile(r"```|\bdef\s+\w+\s*\(|\blambda\b|=>")
_FIXVERB_RX = re.compile(r"\bfix\b|\bcorrect\b|\brepair\b|\bdebug\b"
                         r"|make it work|get it working|\bpatch\b|\bresolve\b")
_CODEGEN_RX = re.compile(
    r"\b(write|implement|create|build|design|provide|develop|give|make"
    r"|define|generate|code)\b[\s\S]{0,60}?"
    r"\b(function|method|program|script|class|solution|algorithm|code"
    r"|snippet|implementation|one[- ]liner)\b"
    r"|^\s*(write|implement|reverse|sort|merge|compute|return|find|create"
    r"|generate|count|check|parse|convert)\b[\s\S]{0,80}?\bin (python|javascript)\b")
_MATH_KW_RX = re.compile(
    r"%|\bpercent|\$\d|\d+ ?(km|kg|miles|hours?|minutes?|dollars?|euros?)\b"
    r"|how (many|much)|\baverage\b|\btotal\b|\bremain(s|ing)?\b|\bcost\b"
    r"|\bprice\b|\brevenue\b|\bchange\b|\bcalculate\b|\btimes\b|multiplied"
    r"|divided by|\bplus\b|\bminus\b|\bsum of\b")
_ARITH_RX = re.compile(r"%|\bpercent|\$\d|[+*/×÷^]|\btimes\b|multiplied|divided by|\bplus\b|\bminus\b")
_NUM_RX = re.compile(r"\d+(?:\.\d+)?")
# Spelled-out quantities count as numeric evidence ("three dozen cookies").
_NUM_WORD_RX = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    r"|twenty|thirty|forty|fifty|hundred|thousand|dozen|half|third|quarter"
    r"|twice|double|triple)\b")
# Two-part questions ("how many X, and which team...") must keep a prose
# shape: a bare PoT number would silently drop the second half of the answer.
# "and how much/many..." is still a single quantity ask — not compound.
_COMPOUND_Q_RX = re.compile(r",\s*and\s+(wh(ich|o|at|ere|en|y)|is|are)\b")
_LOGIC_RX = re.compile(
    r"\bwho (owns?|is|was|wins?|won|finish(es|ed)?|sits?|stands?|came"
    r"|gets?|got|has|had|drinks?|drives?|lives?)\b"
    r"|what (did|does|will) \w+ (get|choose|pick|have|had|order|eat|drink|own)"
    # "four friends each chose a different dessert" - assignment puzzles
    r"|(each|all)[\s\S]{0,40}?\bdifferent\b|each own"
    r"|knights?\b[\s\S]{0,60}?\bknaves?|\bknave\b|always (tells? the truth|lies?)"
    r"|\bpuzzle\b|\bqueue\b|row of|\blabeled\b|\bconstraints?\b|if all .* are"
    # Ordering/assignment puzzles phrased without "who"/"different" fell to
    # factual, whose raw path violates format constraints and rambles
    # (refresh-gauntlet 2026-07-10: "each assigned to exactly one of four
    # labs", "In what order did they finish?"). Nouns list is deliberately
    # assignment-shaped (no team/group - "which team won the cup" is factual).
    r"|in (what|which) order (did|do|does|will)\b"
    r"|\beach\b[\s\S]{0,40}?\bassigned to\b|assigned to (exactly )?one of\b"
    r"|which (lab|desk|seat|room|house|floor|table|office|locker|shelf)\b")


def _is_math(p: str) -> bool:
    if _COMPOUND_Q_RX.search(p) or p.count("?") >= 2:
        return False
    nums = _NUM_RX.findall(p) + _NUM_WORD_RX.findall(p)
    if len(nums) >= 2 and _MATH_KW_RX.search(p):
        return True
    return len(_NUM_RX.findall(p)) >= 1 and bool(_ARITH_RX.search(p))

# suffix: appended to the task prompt (bills ~6-12 input tokens; buys back
# hundreds of completion tokens). max_tokens: hard cap, sized with ~2x margin
# over the largest legitimate answer so truncation can't corrupt a correct one.
# Suffixes compressed 2026-07-10 (ladder-audit wave): every escalated call
# bills its suffix; each rewrite preserves the output-shaping semantics.
SPECS = {
    "math": {
        "suffix": "\n\nGive only the final numeric answer.",
        "max_tokens": 16,
    },
    "logic": {
        "suffix": "\n\nAnswer only, no explanation.",
        "max_tokens": 24,
    },
    "sentiment": {
        "suffix": "\n\nAnswer with the sentiment label and one brief reason.",
        "max_tokens": 48,
    },
    "summarization": {
        "suffix": "\n\nFollow the length/format exactly. Output only the summary.",
        "max_tokens": 112,
    },
    "ner": {
        "suffix": ("\n\nOutput only lines of: entity - type. Include every "
                   "person, organization, location, date, and monetary amount."),
        "max_tokens": 176,
    },
    "factual": {
        "suffix": "\n\nAnswer in 1-2 short sentences.",
        # 128 not 72: factual is always-remote with no truncation repair, and
        # two-part questions ("name X and explain why") need the headroom.
        # Models stop when done, so the wider cap costs ~nothing typically.
        "max_tokens": 128,
    },
    "code-gen": {
        "suffix": "\n\nPython code only, no explanation.",
        "max_tokens": 320,
    },
    "code-debug": {
        "suffix": "\n\nOne-line bug statement, then only the fixed code.",
        "max_tokens": 320,
    },
}


def detect(prompt: str) -> str:
    p = prompt.casefold()
    has_code = bool(_CODE_HINT.search(p))
    # Explicit "named entities" outranks everything: "extract the named
    # entities from this headline/summary/review" is NER, not the category
    # those nouns would suggest. Except a counting ask ("how many named
    # entities...") — that wants a number.
    if "named entit" in p and not _COUNT_Q_RX.search(p):
        return "ner"
    # Science-fact short-circuit BEFORE math: "chemical symbol for iron,
    # plus its atomic number" trips _MATH_KW_RX's bare \bplus\b and ships
    # arithmetic nonsense (fresh-124 nv2-factual-L1-2, twice live).
    if _SCIENCE_FACT_RX.search(p):
        return "factual"
    if _SENTIMENT_RX.search(p):
        return "sentiment"
    if _SUMMAR_RX.search(p):
        return "summarization"
    # A counting question wants a number, never an entity list.
    if _NER_RX.search(p) and not _COUNT_Q_RX.search(p):
        return "ner"
    if has_code and ((_BUG_RX.search(p) and _FIXVERB_RX.search(p))
                     or _CRASH_RX.search(p)
                     or ((_BUG_RX.search(p) or _FIXVERB_RX.search(p))
                         and _ACTUAL_CODE_RX.search(p))):
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
