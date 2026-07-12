"""Offline factual grounding: BM25 lookup over a bundled Simple-Wikipedia
FTS5 index (stdlib sqlite3 only, zero network).

Query recipe (measured 2026-07-12; naive BM25 misses when the answer entity
is absent from the question — "capital of Australia … body of water" ranks
sea/ocean listicles over the Canberra article):
1. draft-first: the caller passes the model's unverified draft; its first
   proper noun that (a) is NOT already in the question and (b) exactly
   matches an article title is taken as the ANSWER ENTITY (the draft names
   the right entity even when it hallucinates the attribute).
2. that article's lead paragraphs (rowid order) + its best question-term
   chunks are pulled title-scoped, then unioned with plain OR-term BM25.

Every public call is exception-proofed: no index file, a corrupt DB, or any
query error returns [] and the caller ships the ungrounded draft — this
module must never be able to break the answer pipeline.
"""

import os
import re
import sqlite3
import unicodedata

_DB_PATH = os.environ.get("WIKI_DB", "/models/wiki/wiki.db")
_conn = None
_conn_failed = False

_STOP = set(
    "what is the a an of and or near which who whom whose when where why how "
    "does do did was were are be been being it its in on at to for with by "
    "from about that this these those name briefly explain difference "
    "between type used known called".split())
# Sentence-initial capitalized words that the proper-noun regex would
# otherwise treat as entities ("The", "It" ...).
_CAP_STOP = {"the", "it", "its", "this", "that", "in", "on", "a", "an", "he",
             "she", "they", "we", "i", "if", "as", "at", "by", "for"}

_PROPER_RX = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")


def _norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                   if not unicodedata.combining(c))


def _terms(q):
    words = re.sub(r"[^a-z0-9 ]", " ", q.lower()).split()
    seen, out = set(), []
    for w in words:
        if w in _STOP or w in seen or len(w) < 2:
            continue
        seen.add(w)
        out.append(w)
    return out


def _get_conn():
    global _conn, _conn_failed
    if _conn is not None or _conn_failed:
        return _conn
    try:
        if not os.path.exists(_DB_PATH):
            _conn_failed = True
            return None
        c = sqlite3.connect(_DB_PATH)
        # keep resident footprint tiny next to llama-server in the 4 GB cgroup:
        # no mmap (pages stay in reclaimable page cache), 16 MB sqlite cache.
        c.execute("PRAGMA mmap_size=0")
        c.execute("PRAGMA cache_size=-16000")
        c.execute("SELECT 1 FROM docs LIMIT 1").fetchone()
        _conn = c
    except Exception:
        _conn_failed = True
        _conn = None
    return _conn


def _title_chunks(db, title, question, k_bm=2, k_lead=2):
    ph = '"%s"' % title.replace('"', "")
    lead = db.execute(
        "SELECT title, body FROM docs WHERE docs MATCH 'title:' || ? "
        "ORDER BY rowid LIMIT ?", (ph, k_lead)).fetchall()
    qterms = " OR ".join('"%s"' % w for w in _terms(question))
    bm = db.execute(
        "SELECT title, body FROM docs WHERE docs MATCH "
        "'title:' || ? || ' AND (' || ? || ')' "
        "ORDER BY bm25(docs,10.0,1.0) LIMIT ?",
        (ph, qterms, k_bm)).fetchall() if qterms else []
    out, seen = [], set()
    for t, b in lead + bm:
        if _norm(t) != _norm(title):  # phrase match can hit longer titles
            continue
        if b[:60] in seen:
            continue
        seen.add(b[:60])
        out.append((t, b))
    return out


def _answer_entity(db, question, draft):
    qwords = set(re.sub(r"[^a-z0-9 ]", " ", _norm(question)).split())
    for m in _PROPER_RX.finditer(draft or ""):
        cand = m.group(1)
        if _norm(cand) in _CAP_STOP:
            continue
        if all(w in qwords for w in _norm(cand).split()):
            continue  # already named in the question -> not the answer entity
        row = db.execute(
            "SELECT title FROM docs WHERE docs MATCH 'title:' || ? LIMIT 1",
            ('"%s"' % cand.replace('"', ""),)).fetchone()
        if row and _norm(row[0]) == _norm(cand):
            return cand
    return None


def lookup(question, draft, k=4):
    """Best-effort passages for a factual question. Returns [(title, body)],
    possibly empty. Never raises."""
    try:
        db = _get_conn()
        if db is None:
            return []
        chunks = []
        entity = _answer_entity(db, question, draft)
        if entity:
            chunks += _title_chunks(db, entity, question)
        q = " OR ".join('"%s"' % w for w in _terms(question + " " + (draft or "")))
        if q:
            chunks += db.execute(
                "SELECT title, body FROM docs WHERE docs MATCH ? "
                "ORDER BY bm25(docs,10.0,1.0) LIMIT 3", (q,)).fetchall()
        out, seen = [], set()
        for t, b in chunks:
            if b[:60] in seen:
                continue
            seen.add(b[:60])
            out.append((t, b))
        return out[:k]
    except Exception:
        return []
