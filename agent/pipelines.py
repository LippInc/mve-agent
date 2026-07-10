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

import ast
import operator
import re
import sys
import time

from agent import codegate
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

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval(expr: str):
    """AST-walk arithmetic evaluator. Numbers and + - * / // % ** only, with
    ** magnitude-bounded from the EVALUATED operands (|exp| <= 64, |base| <=
    1e9) — a regex can never bound arithmetic magnitude (2**(999*999*999)
    passes any textual guard), so the bound has to live at the AST node."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                base, exp = ev(node.left), ev(node.right)
                if abs(exp) > 64 or abs(base) > 10 ** 9:
                    raise ValueError("power out of bounds")
                return base ** exp
            fn = _BIN_OPS.get(type(node.op))
            if fn is None:
                raise ValueError("disallowed operator")
            return fn(ev(node.left), ev(node.right))
        raise ValueError("disallowed expression")

    return ev(ast.parse(expr, mode="eval"))


def _extract_expression(text: str) -> str:
    """Pull a bare arithmetic expression out of the reply (fences stripped)."""
    t = re.sub(r"```(?:python)?", "", text).strip().strip("`").strip()
    candidates = [t] + [ln.strip() for ln in t.splitlines() if ln.strip()]
    for c in candidates:
        c = c.rstrip(";").strip()
        if (0 < len(c) <= 200 and _EXPR_ALLOWED.match(c)
                and any(ch.isdigit() for ch in c)):
            return c
    return ""


_MONEY_RX = re.compile(r"\$|dollars?|cents?|euros?|price|cost|owes?|pay|change")


def _format_number(value, prompt: str = "") -> str:
    if isinstance(value, float):
        # money reads as cents, not 6-decimal precision
        value = round(value, 2 if _MONEY_RX.search(prompt.casefold()) else 6)
        if value.is_integer():
            return str(int(value))
    return str(value)


def _math_task(client, ladder, prompt: str, deadline) -> str:
    reply = _call(client, ladder, prompt + _MATH_POT_SUFFIX, deadline, 48)
    expr = _extract_expression(reply)
    if expr:
        try:
            value = _safe_eval(expr)
            if isinstance(value, (int, float)):
                _log("[pot] expr ok, local eval")
                return _format_number(value, prompt)
        except Exception:
            pass
    # Salvage: the model often answers with the number itself instead of an
    # expression — that IS the answer, no second call needed.
    if reply:
        m = re.fullmatch(r"[^\d\-]{0,12}(-?\d[\d,]*(?:\.\d+)?)\s*%?[.\s]{0,4}", reply.strip())
        if m:
            _log("[pot] salvaged bare number from reply")
            return m.group(1).replace(",", "")
    # Last resort: a possible misroute — answer in the general prose shape
    # rather than forcing a bare number onto a question that may not want one.
    _log("[pot] no usable expression - prose fallback")
    fb = spec_for("factual")
    return _call(client, ladder, prompt + fb["suffix"], deadline, fb["max_tokens"])


# ---------------- summarization: local length verification ----------------

_WORD_LIMIT_RX = re.compile(
    r"(?:no more than|at most|fewer than|under|up to|maximum of|max(?:imum)?(?: of)?)\s+"
    r"(\d+)\s+words?", re.I)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'\-]+", text))


_PER_ITEM_RX = re.compile(r"\beach\b|\bper\s+(bullet|point|item|line|entity|sentence)\b", re.I)


def _summarization_task(client, ladder, prompt: str, deadline, spec) -> str:
    m = _WORD_LIMIT_RX.search(prompt)
    # A per-item limit ("3 bullets, each under 8 words") must NOT be verified
    # against the whole answer — the compress-retry would destroy a correct
    # multi-item structure. Look for per-item markers only in the constraint
    # clause itself: passage text says "each" all the time ("each day").
    clause = prompt[max(0, m.start() - 48): m.end() + 48] if m else ""
    if not m or _PER_ITEM_RX.search(clause):
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


# ---------------- Stage C: verified local code (zero proxy tokens) ----------

_LOCAL_CODE_BUDGET_S = 22.0   # local generation must leave room for a remote fallback

_CODE_LOCAL_SUFFIX = {
    "code-debug": "\n\nState the bug in one short line, then give the corrected "
                  "Python code in a ```python code block.",
    "code-gen": "\n\nGive the Python code in a ```python code block. No explanation.",
}
_SELFTEST_SUFFIX = (
    "\n\nWrite 3 Python assert statements that test the required behaviour of "
    "the function above, using only literal inputs and expected outputs. Output "
    "only the asserts, one per line, no code fences."
)


def _extract_asserts(text: str) -> list:
    """Keep only well-formed, NON-DEGENERATE example asserts: `assert CALL ==
    LITERAL`, where the right side is a plain Python literal (not another call,
    and not a reference to the function under test). This rejects vacuous
    self-tests like `assert f(x) == f(x)` or `assert True` that would let wrong
    code pass its own gate."""
    out, seen = [], set()
    for ln in (text or "").splitlines():
        ln = ln.strip().rstrip(";")
        if not (ln.startswith("assert ") and "==" in ln):
            continue
        body = ln[len("assert "):]
        try:
            node = ast.parse(body, mode="eval").body
        except Exception:
            continue
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)):
            continue
        left, right = node.left, node.comparators[0]
        if not isinstance(left, ast.Call):          # LHS must exercise the code
            continue
        try:
            ast.literal_eval(right)                  # RHS must be a bare literal
        except Exception:
            continue
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out[:5]


def _gate_timeout(local_deadline) -> float:
    """Seconds a gate subprocess may run, clamped to the local budget so the
    whole local attempt never eats into the remote-fallback margin."""
    return max(0.0, min(6.0, local_deadline - time.monotonic()))


def _try_local_code(local, prompt: str, category: str, deadline) -> str:
    """Generate code locally and ship it ONLY if it passes a deterministic
    gate. Returns the answer text on success, or None to fall back to remote.
    Records zero proxy tokens on success (no Fireworks call made).

    The entire local attempt (generation + gates) is bounded by local_deadline,
    which sits inside the per-task deadline so a remote fallback always fits."""
    local_deadline = min(deadline, time.monotonic() + _LOCAL_CODE_BUDGET_S)
    reply = local.chat(prompt + _CODE_LOCAL_SUFFIX[category], max_tokens=384,
                       deadline=local_deadline)
    code = codegate.extract_code(reply)
    if not code.strip():
        return None

    # Gate 1 (always): tests derived from the prompt's own worked examples.
    gt = _gate_timeout(local_deadline)
    if gt >= 1.0:
        passed, n_examples = codegate.gate_code(prompt, code, timeout=gt)
        if passed:
            _log(f"[local-code] {category} shipped (example-gate, {n_examples} tests)")
            return reply.strip()
    else:
        n_examples = len(codegate.extract_example_tests(
            prompt, func=codegate.first_func_name(code)))

    # Gate 2 ("code+" only): model-authored self-tests. Hardened against the
    # "confidently-wrong" failure so it is safer on a zero-margin gate:
    #   - >=3 non-degenerate asserts (LHS call == RHS literal)
    #   - the candidate code passes all of them
    #   - for code-debug: the ORIGINAL buggy code must FAIL >=1 test, proving the
    #     tests actually discriminate the bug (else the tests are vacuous).
    if (local.mode == "code+" and n_examples == 0
            and local_deadline - time.monotonic() > 3.0):
        tset = local.chat(code + _SELFTEST_SUFFIX, max_tokens=160,
                          deadline=local_deadline)
        tests = _extract_asserts(tset)
        gt = _gate_timeout(local_deadline)
        if len(tests) >= 3 and gt >= 1.0 and codegate.run_tests(code, tests, timeout=gt):
            discriminates = True
            if category == "code-debug":
                original = codegate.extract_code(prompt)
                gt = _gate_timeout(local_deadline)
                # tests discriminate iff the original (buggy) code does NOT pass them
                discriminates = (bool(original.strip()) and gt >= 1.0
                                 and not codegate.run_tests(original, tests, timeout=gt))
            if discriminates:
                _log(f"[local-code] {category} shipped (self-test gate, {len(tests)} tests)")
                return reply.strip()

    _log(f"[local-code] {category} unverified -> remote fallback")
    return None


# ---------------- entry point ----------------

def answer_task(client, ladder, prompt: str, deadline, local=None) -> str:
    """Route, shape, verify. Returns the answer text or "" when nothing
    succeeded (caller applies the static fallback)."""
    category = detect(prompt)
    spec = spec_for(category)
    _log(f"[route] category={category} cap={spec['max_tokens']}")

    if (local is not None and local.mode != "off"
            and category in ("code-debug", "code-gen")):
        ans = _try_local_code(local, prompt, category, deadline)
        if ans:
            return ans  # verified locally -> zero proxy tokens

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
