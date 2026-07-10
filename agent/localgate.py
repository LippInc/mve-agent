"""Deterministic verifiers for Stage D local NLP answers (zero proxy tokens).

Same doctrine as codegate: a local answer ships ONLY when a check we can run
deterministically says it is trustworthy; anything unverifiable falls back to
the remote pipeline. The checks here are category-specific:

- sentiment: two independently-framed local classifications must agree on a
  label from the closed label set.
- ner: every extracted entity must appear verbatim (casefolded) in the source
  text (kills hallucinated entities), and two independent extractions must
  agree on the entity set (evidence of completeness, the part a substring
  check cannot see).
- summarization: hard word-limit compliance (the judge-visible constraint).

Pure stdlib, never raises to the caller.
"""

import re

# ---------------- sentiment ----------------

SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed")


def extract_sentiment_label(text: str):
    """First closed-set label present in the reply, or None."""
    low = (text or "").casefold()
    hits = [lb for lb in SENTIMENT_LABELS if lb in low]
    return hits[0] if len(hits) == 1 else None


def sentiment_agree(reply_a: str, reply_b: str):
    """Label iff both replies contain exactly one label and they match."""
    a, b = extract_sentiment_label(reply_a), extract_sentiment_label(reply_b)
    return a if (a is not None and a == b) else None


# ---------------- ner ----------------

# "entity - type" with tolerance for the dash/colon variants small models emit.
_ENTITY_LINE_RX = re.compile(r"^\s*(?:[-*•]\s*)?(.+?)\s*(?:-|–|—|:)\s*([A-Za-z ]{2,24})\s*$")

_TYPE_NORM = {
    "person": "person", "people": "person", "per": "person", "name": "person",
    "organization": "organization", "organisation": "organization",
    "org": "organization", "company": "organization", "agency": "organization",
    "location": "location", "place": "location", "loc": "location",
    "gpe": "location", "city": "location", "country": "location",
    "date": "date", "time": "date", "year": "date",
    "money": "money", "monetary": "money", "monetary amount": "money",
    "amount": "money", "currency": "money", "price": "money",
}


def _norm_type(t: str) -> str:
    t = (t or "").strip().casefold()
    return _TYPE_NORM.get(t, t)


def _norm_entity(e: str) -> str:
    return re.sub(r"\s+", " ", (e or "").strip().strip("\"'").rstrip(".,;")).casefold()


def parse_entity_lines(text: str) -> list:
    """Parse 'entity - type' lines into [(entity, normalized_type)]. Lines that
    do not match the shape are ignored (prose preambles etc.)."""
    out = []
    for ln in (text or "").splitlines():
        m = _ENTITY_LINE_RX.match(ln)
        if not m:
            continue
        ent, typ = m.group(1).strip(), _norm_type(m.group(2))
        if ent and typ:
            out.append((ent, typ))
    return out


def ner_verify(pairs: list, source: str) -> bool:
    """Every entity must appear verbatim (casefolded, whitespace-normalized) in
    the source text; at least one entity required. Catches hallucination and
    rewritten entities — the unsafe direction. (Cannot catch misses; that is
    what the two-sample agreement is for.)"""
    if not pairs:
        return False
    src = re.sub(r"\s+", " ", (source or "")).casefold()
    return all(_norm_entity(e) in src for e, _ in pairs)


def ner_sets_agree(pairs_a: list, pairs_b: list) -> bool:
    """Two extractions agree iff the (entity, type) sets match after
    normalization. Type disagreement on the same entity fails too — the type
    is part of the graded answer."""
    sa = {(_norm_entity(e), t) for e, t in pairs_a}
    sb = {(_norm_entity(e), t) for e, t in pairs_b}
    return bool(sa) and sa == sb


def format_entities(pairs: list) -> str:
    """Canonical output shape (same as the remote spec asks for), deduped,
    source order preserved."""
    seen, lines = set(), []
    for e, t in pairs:
        key = (_norm_entity(e), t)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{e} - {t}")
    return "\n".join(lines)


# ---------------- summarization ----------------

def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'\-]+", text or ""))
