"""Per-category pipeline. Zero-cost regex routing, then a category-tuned
terse call — plus two verified local-compute layers (stdlib-only):

- math runs Program-of-Thought: the model emits a bare arithmetic expression,
  the container evaluates it locally (exact, deterministic — suppressed models
  misarithmetic multi-step in their head; measured 4/7 on the bench without
  this). Falls back to a plain terse call when the expression doesn't parse.
- summarization word limits are verified locally (word counts are free and
  deterministic); one compress-the-draft retry on violation, re-sending only
  the short draft, never the full passage.

answer_task stays the single entry point main.py talks to.
"""

import re
import sys
import time

from agent.categories import detect, spec_for


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _is_minimax(model: str) -> bool:
    return "minimax" in model.rsplit("/", 1)[-1].lower()


def _model_params(model: str) -> dict:
    # thinking is a token optimization, never load-bearing: fw.py strips it
    # on a 400 and the call still succeeds.
    if _is_minimax(model):
        return {"thinking": {"type": "disabled"}}
    return {}


def _call(client, ladder, text: str, deadline, max_tokens: int) -> str:
    """One shaped completion, walking the ladder best-first. Returns "" when
    nothing succeeded."""
    for model in list(ladder):
        if model in client.dead_models:
            continue
        res = client.chat(
            model,
            [{"role": "user", "content": text}],
            deadline,
            max_tokens=max_tokens,
            **_model_params(model),
        )
        if res.ok and res.content.strip():
            return res.content.strip()
        # model_dead → next model; transient failure with time left → next
        # model too (correlated-failure insurance); out of time → give up
        if deadline - time.monotonic() < 2.0:
            break
    return ""


# ---------------- math: Program-of-Thought with local evaluation ----------------

_MATH_POT_SUFFIX = (
    "\n\nWrite one Python arithmetic expression that computes the final "
    "answer. Output only the expression, nothing else."
)
_EXPR_ALLOWED = re.compile(r"^[0-9+\-*/().\s]+$")


def _extract_expression(text: str) -> str:
    """Pull a bare arithmetic expression out of the reply (fences stripped)."""
    t = re.sub(r"```(?:python)?", "", text).strip().strip("`").strip()
    candidates = [t] + [ln.strip() for ln in t.splitlines() if ln.strip()]
    for c in candidates:
        c = c.rstrip(";").strip()
        if not (0 < len(c) <= 200 and _EXPR_ALLOWED.match(c)
                and any(ch.isdigit() for ch in c)):
            continue
        # exponent guard: an in-process eval of e.g. 9**9**9 would hang past
        # every watchdog (eval is uninterruptible) — refuse 4+-digit exponents
        # and directly-chained powers (a**b**c); separate powers in one formula
        # (compound interest) stay legal because an operator sits between them
        if re.search(r"\*\*\s*\(?\s*\d{4,}", c) or re.search(r"\*\*[\s\d.()]*\*\*", c):
            continue
        return c
    return ""


def _format_number(value) -> str:
    if isinstance(value, float):
        value = round(value, 6)
        if value.is_integer():
            return str(int(value))
    return str(value)


def _math_task(client, ladder, prompt: str, deadline) -> str:
    reply = _call(client, ladder, prompt + _MATH_POT_SUFFIX, deadline, 48)
    expr = _extract_expression(reply)
    if expr:
        try:
            value = eval(expr, {"__builtins__": {}}, {})  # charset-validated arithmetic only
            if isinstance(value, (int, float)):
                _log(f"[pot] expr ok, local eval")
                return _format_number(value)
        except Exception:
            pass
    _log("[pot] no usable expression - plain fallback")
    return _call(client, ladder, prompt + "\n\nGive only the final numeric answer.",
                 deadline, 16)


# ---------------- summarization: local length verification ----------------

_WORD_LIMIT_RX = re.compile(
    r"(?:no more than|at most|fewer than|under|up to|maximum of|max(?:imum)?(?: of)?)\s+"
    r"(\d+)\s+words?", re.I)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'\-]+", text))


def _summarization_task(client, ladder, prompt: str, deadline, spec) -> str:
    m = _WORD_LIMIT_RX.search(prompt)
    if not m:
        return _call(client, ladder, prompt + spec["suffix"], deadline,
                     spec["max_tokens"])
    limit = int(m.group(1))
    target = max(5, (limit * 7) // 10)  # models overshoot; aim well under
    draft = _call(
        client, ladder,
        prompt + f"\n\nOutput only the summary, in at most {target} words. "
                 f"Never exceed {limit} words.",
        deadline, spec["max_tokens"])
    if not draft or _word_count(draft) <= limit:
        return draft
    _log(f"[sumlen] draft {_word_count(draft)} words > limit {limit} - compressing")
    shorter = _call(
        client, ladder,
        f"Shorten this to at most {target} words, keeping the meaning and "
        f"format. Output only the result.\n\n{draft}",
        deadline, spec["max_tokens"])
    if shorter and _word_count(shorter) <= limit:
        return shorter
    return shorter or draft


# ---------------- entry point ----------------

def answer_task(client, ladder, prompt: str, deadline) -> str:
    """Route, shape, verify. Returns the answer text or "" when nothing
    succeeded (caller applies the static fallback)."""
    category = detect(prompt)
    spec = spec_for(category)
    _log(f"[route] category={category} cap={spec['max_tokens']}")
    if category == "math":
        return _math_task(client, ladder, prompt, deadline)
    if category == "summarization":
        return _summarization_task(client, ladder, prompt, deadline, spec)
    return _call(client, ladder, prompt + spec["suffix"], deadline,
                 spec["max_tokens"])


def order_ladder(client, ladder):
    """Stable reorder: alive models first, leak suspects after clean ones."""
    alive = [m for m in ladder if m not in client.dead_models]
    clean = [m for m in alive if not client.leak_suspect(m)]
    suspect = [m for m in alive if client.leak_suspect(m)]
    return clean + suspect
