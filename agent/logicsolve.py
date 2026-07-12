"""Deterministic logic-puzzle solver (LLM translates, Python solves).

The small local model is unreliable at *solving* constraint puzzles (live-
difficulty strict: 5/15) but competent at *translating* them into a tiny
structured spec. This module enumerates the full solution space of that spec
and only ships when the puzzle's ANSWER is uniquely determined — a mis-parsed
spec almost always yields zero or many answers, not a unique wrong one, so
the uniqueness gate doubles as the verification step. Zero proxy tokens,
pure stdlib.

Spec format (line-based DSL — token-cheap for a 3B model to emit):

    TYPE order|assign|knights|intsystem
    ITEMS name1 name2 ...            (people/entities, one word each)
    VALUES v1 v2 ...                 (assign only: the attribute domain)
    C <constraint ...>               (zero or more)
    Q <question ...>

Constraints by type:
    order:     C pos <e> <n>          entity is at position n (1 = first/left)
               C notpos <e> <n>
               C immbefore <a> <b>    pos(a)+1 == pos(b)
               C before <a> <b>       pos(a) < pos(b)
               C after <a> <b>        pos(a) > pos(b)
               C adjacent <a> <b>
               C notadjacent <a> <b>
    assign:    C is <e> <v>
               C not <e> <v>
    knights:   C says <speaker> role <e> knight|knave
               C says <speaker> all <e1> <e2> ... knight|knave
               C says <speaker> atleastone <e1> <e2> ... knight|knave
               C says <speaker> same <e1> <e2>
               C says <speaker> diff <e1> <e2>
    intsystem: C kind <name> <per_unit_count>     (exactly two kinds)
               C heads <n>
               C legs <n>
Questions:
    Q valueof <e>        -> "e: v" / order: position of e
    Q whohas <v>         -> entity with value v / order: entity at position v
    Q fullorder          -> "e1, e2, ..." in position order
    Q roles              -> "A is a knight, B is a knave, ..."
    Q roleof <e>         -> "e is a knight/knave."
    Q count <kindname>   -> intsystem: "<n> <kindname>"
"""

import itertools
import re


class SpecError(ValueError):
    pass


def parse_spec(text: str) -> dict:
    """Tolerant line-DSL parser: ignores blank lines, markdown fences, and
    anything before TYPE / after the last recognized line."""
    spec = {"constraints": []}
    for raw in text.splitlines():
        ln = raw.strip().strip("`").strip()
        if not ln:
            continue
        parts = ln.split()
        head = parts[0].upper()
        if head == "TYPE" and len(parts) >= 2:
            spec["type"] = parts[1].lower()
        elif head == "ITEMS":
            spec["items"] = [p.strip(",.") for p in parts[1:] if p.strip(",.")]
        elif head == "VALUES":
            spec["values"] = [p.strip(",.").lower() for p in parts[1:]
                              if p.strip(",.")]
        elif head == "C" and len(parts) >= 2:
            spec["constraints"].append([p.strip(",.") for p in parts[1:]])
        elif head == "Q" and len(parts) >= 2:
            spec["question"] = [p.strip(",.") for p in parts[1:]]
    if "type" not in spec or "question" not in spec:
        raise SpecError("missing TYPE or Q")
    return spec


def _norm(s: str) -> str:
    return s.strip().casefold()


# ---------------- order / assignment ----------------

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "sixth": 6, "seventh": 7, "eighth": 8}


def _pos_tok(tok: str, n: int) -> int:
    """Position token: digits, ordinal words, or 'last' (needs N)."""
    t = _norm(tok).rstrip(".")
    if t in _ORDINALS:
        return _ORDINALS[t]
    if t == "last":
        return n
    return int(t)


def _check_order(c, pos) -> bool:
    n = len(pos)
    k = c[0].lower()
    try:
        if k == "pos":
            return pos[_norm(c[1])] == _pos_tok(c[2], n)
        if k == "notpos":
            return pos[_norm(c[1])] != _pos_tok(c[2], n)
        if k == "immbefore":
            return pos[_norm(c[1])] + 1 == pos[_norm(c[2])]
        if k == "immafter":
            return pos[_norm(c[1])] == pos[_norm(c[2])] + 1
        if k == "before":
            return pos[_norm(c[1])] < pos[_norm(c[2])]
        if k == "after":
            return pos[_norm(c[1])] > pos[_norm(c[2])]
        if k == "adjacent":
            return abs(pos[_norm(c[1])] - pos[_norm(c[2])]) == 1
        if k == "notadjacent":
            return abs(pos[_norm(c[1])] - pos[_norm(c[2])]) != 1
    except (KeyError, ValueError, IndexError):
        raise SpecError(f"bad order constraint {c}")
    raise SpecError(f"unknown order constraint {c[0]}")


def _solve_order(spec):
    items = [_norm(i) for i in spec.get("items", [])]
    if not 2 <= len(items) <= 8 or len(set(items)) != len(items):
        raise SpecError("bad ITEMS")
    sols = []
    for perm in itertools.permutations(items):
        pos = {e: i + 1 for i, e in enumerate(perm)}
        if all(_check_order(c, pos) for c in spec["constraints"]):
            sols.append(pos)
    return sols


def _check_assign(c, val) -> bool:
    k = c[0].lower()
    try:
        if k == "is":
            return val[_norm(c[1])] == _norm(c[2])
        if k == "not":
            return val[_norm(c[1])] != _norm(c[2])
    except (KeyError, IndexError):
        raise SpecError(f"bad assign constraint {c}")
    raise SpecError(f"unknown assign constraint {c[0]}")


def _solve_assign(spec):
    items = [_norm(i) for i in spec.get("items", [])]
    values = [_norm(v) for v in spec.get("values", [])]
    if not items or len(items) != len(values):
        raise SpecError("ITEMS/VALUES mismatch")
    sols = []
    for perm in itertools.permutations(values):
        val = dict(zip(items, perm))
        if all(_check_assign(c, val) for c in spec["constraints"]):
            sols.append(val)
    return sols


# ---------------- knights & knaves ----------------

_SELF_TOKENS = {"self", "i", "me", "myself"}


def _claim_true(c, roles, speaker) -> bool:
    """c = tokens after the speaker: e.g. ['role','rune','knave'].
    'self'/'I'/'me' resolve to the speaker."""
    def who(tok):
        t = _norm(tok)
        return speaker if t in _SELF_TOKENS else t

    k = c[0].lower()
    try:
        if k == "role":
            return roles[who(c[1])] == c[2].lower()
        if k == "all":
            *ents, r = c[1:]
            return all(roles[who(e)] == r.lower() for e in ents)
        if k == "atleastone":
            *ents, r = c[1:]
            return any(roles[who(e)] == r.lower() for e in ents)
        if k == "same":
            return roles[who(c[1])] == roles[who(c[2])]
        if k == "diff":
            return roles[who(c[1])] != roles[who(c[2])]
    except (KeyError, IndexError, ValueError):
        raise SpecError(f"bad claim {c}")
    raise SpecError(f"unknown claim {c[0]}")


def _solve_knights(spec):
    items = [_norm(i) for i in spec.get("items", [])]
    if not 1 <= len(items) <= 6:
        raise SpecError("bad ITEMS")
    statements = []
    for c in spec["constraints"]:
        if c[0].lower() != "says" or len(c) < 3:
            raise SpecError(f"knights constraint must be 'says': {c}")
        statements.append((_norm(c[1]), c[2:]))
    sols = []
    for combo in itertools.product(("knight", "knave"), repeat=len(items)):
        roles = dict(zip(items, combo))
        ok = True
        for speaker, claim in statements:
            if speaker not in roles:
                raise SpecError(f"unknown speaker {speaker}")
            truth = _claim_true(claim, roles, speaker)
            if (roles[speaker] == "knight") != truth:
                ok = False
                break
        if ok:
            sols.append(roles)
    return sols


# ---------------- integer system (heads & legs) ----------------

def _solve_intsystem(spec):
    kinds, heads, legs = [], None, None
    for c in spec["constraints"]:
        k = c[0].lower()
        if k == "kind":
            kinds.append((_norm(c[1]), int(c[2])))
        elif k == "heads":
            heads = int(c[1])
        elif k == "legs":
            legs = int(c[1])
    if len(kinds) != 2 or heads is None or legs is None:
        raise SpecError("intsystem needs 2 kinds + heads + legs")
    (n1, p1), (n2, p2) = kinds
    sols = []
    for a in range(heads + 1):
        b = heads - a
        if a * p1 + b * p2 == legs:
            sols.append({n1: a, n2: b})
    return sols


# ---------------- answer projection & formatting ----------------

def _project(spec, sols):
    """Reduce full solutions to the asked answer; unique-answer acceptance."""
    q = spec["question"]
    qk = q[0].lower()
    t = spec["type"]
    answers = set()
    for sol in sols:
        if qk == "valueof":
            key = _norm(q[1])
            if key not in sol:
                raise SpecError(f"question entity {q[1]} not in solution")
            answers.add(str(sol[key]))
        elif qk == "whohas":
            want = _norm(q[1])
            if t == "order":
                try:
                    want = str(_pos_tok(want, len(sol)))
                except ValueError:
                    raise SpecError(f"bad position {q[1]}")
            hit = [e for e, v in sol.items() if str(v) == want]
            if len(hit) != 1:
                return None
            answers.add(hit[0])
        elif qk == "fullorder":
            if t != "order":
                raise SpecError("fullorder needs TYPE order")
            order = sorted(sol, key=lambda e: sol[e])
            answers.add(", ".join(e.capitalize() for e in order))
        elif qk == "fullassign":
            if t != "assign":
                raise SpecError("fullassign needs TYPE assign")
            answers.add(", ".join(
                f"{e.capitalize()}: {sol[e]}" for e in sorted(sol)))
        elif qk == "roles":
            parts = [f"{e.capitalize()} is a {sol[e]}" for e in sol]
            if len(parts) > 1:
                parts[-1] = "and " + parts[-1]
            answers.add(", ".join(parts) + ".")
        elif qk == "roleof":
            key = _norm(q[1])
            answers.add(f"{key.capitalize()} is a {sol[key]}.")
        elif qk == "whois":
            role = q[1].lower()
            hit = [e for e, r in sol.items() if r == role]
            if len(hit) != 1:
                return None
            answers.add(hit[0].capitalize())
        elif qk == "count":
            key = _norm(q[1])
            if key not in sol:
                raise SpecError(f"count kind {q[1]} not in solution")
            answers.add(f"{sol[key]} {key}")
        else:
            raise SpecError(f"unknown question {qk}")
        if len(answers) > 1:
            return None
    if len(answers) != 1:
        return None
    ans = answers.pop()
    if qk in ("valueof", "whohas") and t in ("order", "assign"):
        ans = ans.capitalize() if not ans.isdigit() else ans
    return ans


def solve(text: str):
    """Parse a spec and return (answer, n_solutions) or raise SpecError.
    answer is None when the puzzle's asked answer is not uniquely determined
    (the acceptance gate) — 0 solutions, or solutions that disagree on the
    projection."""
    spec = parse_spec(text)
    t = spec["type"]
    if t == "order":
        sols = _solve_order(spec)
    elif t == "assign":
        sols = _solve_assign(spec)
    elif t == "knights":
        sols = _solve_knights(spec)
    elif t == "intsystem":
        sols = _solve_intsystem(spec)
    else:
        raise SpecError(f"unknown TYPE {t}")
    if not sols:
        return None, 0
    return _project(spec, sols), len(sols)


# quick shape trigger for the pipeline (heads-and-legs misroutes to math)
INTSYSTEM_RX = re.compile(r"\bheads\b.{0,200}\b(legs|wheels)\b", re.I | re.S)
KNIGHTS_RX = re.compile(r"knights? always tell the truth", re.I)
