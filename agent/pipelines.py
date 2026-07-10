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
from agent import localgate
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
_COUNT_Q_RX = re.compile(r"how many|number of|count of", re.I)


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
_SELFTEST_SUFFIX2 = (
    "\n\nWrite 4 assert statements checking this function on typical and edge "
    "inputs. Each assert must compare a direct call against a literal expected "
    "value, e.g. assert f(2) == 4. Asserts only, one per line."
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


def _try_local_code(local, prompt: str, category: str, deadline,
                    ship_unverified: bool = False) -> str:
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
        if len(tests) < 3 and local_deadline - time.monotonic() > 3.0:
            # one rephrased retry: the first generation often yields non-literal
            # or malformed asserts that the extractor (correctly) rejects.
            tset2 = local.chat(code + _SELFTEST_SUFFIX2, max_tokens=200,
                               deadline=local_deadline)
            tests = _extract_asserts((tset or "") + "\n" + (tset2 or ""))
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

    if ship_unverified and reply.strip():
        _log(f"[local-code] {category} unverified but LOCAL_ONLY - shipped raw")
        return reply.strip()
    _log(f"[local-code] {category} unverified -> remote fallback")
    return None


# ---------------- Stage D: verified local NLP (zero proxy tokens) ----------

_LOCAL_MATH_BUDGET_S = 16.0
_LOCAL_NLP_BUDGET_S = 18.0

_MATH_POT_SUFFIX2 = (
    "\n\nTranslate this problem into a single Python arithmetic expression "
    "using only numbers and + - * / ( ). Reply with the expression alone."
)
_MATH_POT_SUFFIX3 = (
    "\n\nSolve by writing one arithmetic expression that evaluates to the "
    "answer (no words, no explanation). Expression only."
)


def _eval_expression(reply: str):
    """Expression text -> numeric value, or None."""
    expr = _extract_expression(reply)
    if not expr:
        return None
    try:
        value = _safe_eval(expr)
    except Exception:
        return None
    return value if isinstance(value, (int, float)) else None


def _num_close(a, b) -> bool:
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))


# Program-of-Thought, program form: single expressions cannot express "solve
# a system" or "brute-force the assignment" — a tiny printed-answer program
# can, and small models write brute-force loops far more reliably than they
# do mental algebra. Executed in codegate's hardened sandbox.
_POT_PROG_SUFFIX = (
    "\n\nWrite a short Python program that computes the answer and prints "
    "ONLY the final answer (no labels, no explanation). Use plain Python "
    "(math/itertools allowed, no input(), no files, no network). Output only "
    "the code."
)


_COMPUTED_RX = re.compile(r"\bfor\b|\bwhile\b|\bif\b|itertools|range\(")


def _pot_program_answer(local, prompt: str, deadline):
    """Model writes a tiny program; we run it sandboxed. Returns
    (last_stdout_line, computed) or (None, False). `computed` is True when
    the program actually computes (loops/branches) rather than just printing
    a belief — a computed answer is stronger evidence than a terse reply."""
    if deadline - time.monotonic() < 3.0:
        return None, False
    reply = local.chat(prompt + _POT_PROG_SUFFIX, max_tokens=288,
                       deadline=deadline)
    code = codegate.extract_code(reply)
    if not code.strip() or len(code) > 2400 or "input(" in code:
        return None, False
    out = codegate.run_capture(code, timeout=min(6.0, max(1.0, deadline - time.monotonic())))
    if not out or not out.strip():
        return None, False
    return out.strip().splitlines()[-1].strip(), bool(_COMPUTED_RX.search(code))


def _try_local_math(local, prompt: str, deadline, local_only: bool = False) -> str:
    """Independently-framed local PoT derivations; ship as soon as any two
    evaluate to the same number (2-of-3). Different phrasings decorrelate
    surface slips; a residual correlated setup error is the accepted
    (bench-measured) risk. In LOCAL_ONLY a bounded CoT pass joins as a
    tiebreaker, and the best single candidate ships rather than nothing."""
    budget = 26.0 if local_only else _LOCAL_MATH_BUDGET_S
    reserve = 7.0 if local_only else 0.0
    local_deadline = min(deadline - reserve, time.monotonic() + budget)

    def _guard(v):
        # A negative answer to a "how many/number of" question is always a
        # sign-slip (measured: derivations agreeing on -8 rabbits). The
        # magnitude is what was derived — salvage abs() rather than reject:
        # a wrong magnitude stays wrong either way, a backwards subtraction
        # becomes the right answer.
        if v is not None and float(v) < 0 and _COUNT_Q_RX.search(prompt):
            _log("[local-math] negative count -> abs salvage")
            return abs(float(v))
        return v

    def _expr_derivation(suffix):
        return _guard(_eval_expression(local.chat(prompt + suffix, max_tokens=64,
                                                  deadline=local_deadline)))

    def _prog_derivation():
        ans, _ = _pot_program_answer(local, prompt, local_deadline)
        m = _NUM_IN_TEXT_RX.search(ans or "")
        if not m:
            return None
        try:
            return _guard(float(m.group(0).replace(",", "")))
        except ValueError:
            return None

    # Method diversity first: an expression and an executed program agreeing
    # is stronger evidence than two same-method samples (mental-algebra slips
    # correlate; expression-vs-program errors don't).
    derivations = [lambda: _expr_derivation(_MATH_POT_SUFFIX),
                   _prog_derivation,
                   lambda: _expr_derivation(_MATH_POT_SUFFIX2)]
    if local_only:
        derivations.append(lambda: _expr_derivation(_MATH_POT_SUFFIX3))

    values = []
    for derive in derivations:
        if local_deadline - time.monotonic() < 1.5:
            break
        v = derive()
        if v is None:
            continue
        for prev in values:
            if _num_close(prev, v):
                _log("[local-math] two derivations agree - shipped local")
                return _format_number(v, prompt)
        values.append(v)
    if local_only:
        # Hard tail: spend a real thinking budget — local tokens are free,
        # only time is budgeted (bounded by the caller's per-task deadline
        # and the global watchdog; total runtime is the binding contract
        # limit, not per-task time).
        think_deadline = min(deadline - reserve, time.monotonic() + 70.0)
        cot_v = None
        if think_deadline - time.monotonic() > 8.0:
            reply = local.chat(prompt + _MATH_COT_SUFFIX, max_tokens=704,
                               deadline=think_deadline)
            m = _NUM_IN_TEXT_RX.search(_extract_final_answer(reply))
            if m:
                try:
                    cot_v = _guard(float(m.group(0).replace(",", "")))
                except ValueError:
                    cot_v = None
        if cot_v is not None:
            for prev in values:
                if _num_close(prev, cot_v):
                    _log("[local-math] thinking pass confirms a derivation - shipped local")
                    return _format_number(prev, prompt)
        if values:
            # no agreement: terse-first single (measured better than
            # CoT-preference on the borderline set)
            _log("[local-math] no agreement, LOCAL_ONLY - shipped terse single")
            return _format_number(values[0], prompt)
        if cot_v is not None:
            _log("[local-math] only the thinking pass produced a number - shipped")
            return _format_number(cot_v, prompt)
    _log("[local-math] no two derivations agree -> remote")
    return None


_SENT_LOCAL_A = "\n\nAnswer with exactly one word - positive, negative, neutral, or mixed."
_SENT_LOCAL_B = ("\n\nWhat is the overall sentiment? Reply with only one label "
                 "from: positive, negative, neutral, mixed.")


def _try_local_sentiment(local, prompt: str, deadline) -> str:
    local_deadline = min(deadline, time.monotonic() + _LOCAL_NLP_BUDGET_S)
    a = local.chat(prompt + _SENT_LOCAL_A, max_tokens=8, deadline=local_deadline)
    if not a:
        return None
    b = local.chat(prompt + _SENT_LOCAL_B, max_tokens=8, deadline=local_deadline)
    label = localgate.sentiment_agree(a, b)
    if not label:
        _log("[local-sent] framings disagree -> remote")
        return None
    reason = local.chat(
        prompt + f"\n\nThe sentiment is {label}. State the key reason in at "
                 "most 10 words.",
        max_tokens=24, deadline=local_deadline)
    reason = (reason or "").splitlines()[0].strip() if reason else ""
    _log("[local-sent] agreed label - shipped local")
    return f"{label} - {reason}" if reason else label


_NER_LOCAL_B = ("\n\nExtract all named entities from the text. Output one per "
                "line as: entity - type (person, organization, location, date, "
                "or money). No other text.")
_NER_LOCAL_C = ("\n\nList each named entity in the text on its own line in the "
                "form 'entity - type', covering every person, organization, "
                "location, date, and monetary amount. Output nothing else.")


def _try_local_ner(local, prompt: str, deadline) -> str:
    """Up to three independently-framed extractions; ship as soon as any two
    substring-verified samples agree on the full (entity, type) set. Every
    shipped entity is verbatim-in-source (kills hallucination); the set
    agreement is the completeness evidence."""
    local_deadline = min(deadline, time.monotonic() + _LOCAL_NLP_BUDGET_S + 6.0)
    spec = spec_for("ner")
    verified = []
    for suffix in (spec["suffix"], _NER_LOCAL_B, _NER_LOCAL_C):
        if local_deadline - time.monotonic() < 2.0:
            break
        pairs = localgate.parse_entity_lines(
            local.chat(prompt + suffix, max_tokens=spec["max_tokens"],
                       deadline=local_deadline))
        if not localgate.ner_verify(pairs, prompt):
            continue
        for prev in verified:
            if localgate.ner_sets_agree(prev, pairs):
                _log(f"[local-ner] {len(pairs)} entities agreed - shipped local")
                return localgate.format_entities(pairs)
        verified.append(pairs)
    _log("[local-ner] no two verified samples agree -> remote")
    return None


_SHORT_ANSWER_SUFFIXES = (
    "\n\nGive only the final answer, no explanation.",
    "\n\nAnswer with just the final answer and nothing else.",
    "\n\nState the final answer alone, without reasoning or commentary.",
)

# Bounded local chain-of-thought: in LOCAL_ONLY, tokens are free and only
# time is budgeted, so a thinking pass is a legitimate extra derivation for
# the hard reasoning tail (math word problems, logic puzzles).
_COT_SHORT_SUFFIX = (
    "\n\nThink through this step by step briefly, then give only the final "
    "answer on the last line in the form: ANSWER: <answer>"
)
_MATH_COT_SUFFIX = (
    "\n\nSolve this step by step briefly, then give only the final numeric "
    "answer on the last line in the form: ANSWER: <number>"
)
_ANSWER_LINE_RX = re.compile(r"answer\s*[:=]\s*(.+)", re.I)
_NUM_IN_TEXT_RX = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _extract_final_answer(text: str) -> str:
    for ln in reversed((text or "").splitlines()):
        m = _ANSWER_LINE_RX.search(ln)
        if m:
            return m.group(1).strip()
    return ""


def _norm_short(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().strip("\"'`")).rstrip(".!")
    t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.I)
    return t.casefold()


def _try_local_short_agree(local, prompt: str, deadline, tag: str,
                           max_words: int = 8, local_only: bool = False) -> str:
    """2-of-3 agreement on a short final answer (logic / factual). Long or
    rambling replies never count as agreement evidence — a short exact match
    across independently-framed asks is the confidence signal. In LOCAL_ONLY
    a bounded CoT pass joins as a tiebreaker and the best single candidate
    ships rather than nothing."""
    budget = 26.0 if local_only else _LOCAL_NLP_BUDGET_S
    # In LOCAL_ONLY always leave room for the raw-fallback shot: an empty
    # answer is the one outcome worse than an unverified one (measured:
    # budget exhaustion shipped "" on two tasks).
    reserve = 7.0 if local_only else 0.0
    local_deadline = min(deadline - reserve, time.monotonic() + budget)

    def _terse(suffix):
        return local.chat(prompt + suffix, max_tokens=32,
                          deadline=local_deadline), False

    # Method diversity: a brute-force program's printed answer agreeing with
    # a terse reply is stronger evidence than two terse replies (a model's
    # wrong belief repeats across framings — measured on the pet-puzzle task —
    # but doesn't survive execution). A COMPUTED program answer also vetoes a
    # terse-terse agreement that contradicts it.
    samples = [lambda: _terse(_SHORT_ANSWER_SUFFIXES[0]),
               lambda: _pot_program_answer(local, prompt, local_deadline),
               lambda: _terse(_SHORT_ANSWER_SUFFIXES[1]),
               lambda: _terse(_SHORT_ANSWER_SUFFIXES[2])]

    seen = []          # (norm, reply, computed)
    computed_norm = None
    for sample in samples:
        if local_deadline - time.monotonic() < 1.5:
            break
        reply, computed = sample()
        norm = _norm_short(reply)
        if not norm or len(norm.split()) > max_words:
            continue
        if computed and computed_norm is None:
            computed_norm = norm
        for prev_norm, prev_reply, prev_computed in seen:
            if prev_norm != norm:
                continue
            if (computed_norm is not None and norm != computed_norm
                    and not (computed or prev_computed)):
                _log(f"[{tag}] terse agreement vetoed by computed answer")
                continue
            _log(f"[{tag}] two samples agree - shipped local")
            return prev_reply.strip().strip("\"'`")
        seen.append((norm, reply, computed))
    cot_ans = ""
    if local_only:
        think_deadline = min(deadline - reserve, time.monotonic() + 70.0)
        if think_deadline - time.monotonic() > 8.0:
            reply = local.chat(prompt + _COT_SHORT_SUFFIX, max_tokens=704,
                               deadline=think_deadline)
            cot_ans = _extract_final_answer(reply)
            norm = _norm_short(cot_ans)
            if norm and len(norm.split()) <= max_words:
                for prev_norm, prev_reply, _pc in seen:
                    if prev_norm == norm:
                        _log(f"[{tag}] thinking pass confirms a sample - shipped local")
                        return prev_reply.strip().strip("\"'`")
            else:
                cot_ans = ""
    if local_only and (cot_ans or seen):
        computed = [r for _n, r, c in seen if c]
        best = computed[0] if computed else (seen[0][1] if seen else cot_ans)
        if not best:
            best = cot_ans
        _log(f"[{tag}] no agreement, LOCAL_ONLY - shipped best single")
        return best.strip().strip("\"'`")
    _log(f"[{tag}] no short-answer agreement -> remote")
    return None


def _try_local_sum(local, prompt: str, deadline) -> str:
    """Local summaries ship ONLY when the task states a whole-answer word
    limit we can verify deterministically; unconstrained summaries have no
    local verifier and always stay remote."""
    m = _WORD_LIMIT_RX.search(prompt)
    if not m:
        return None
    clause = prompt[max(0, m.start() - 48): m.end() + 48]
    if _PER_ITEM_RX.search(clause):
        return None
    limit = int(m.group(1))
    target = max(5, (limit * 7) // 10)
    local_deadline = min(deadline, time.monotonic() + _LOCAL_NLP_BUDGET_S)
    draft = local.chat(
        prompt + f"\n\nOutput only the summary, in at most {target} words. "
                 f"Never exceed {limit} words.",
        max_tokens=112, deadline=local_deadline)
    if draft and localgate.word_count(draft) <= limit:
        _log("[local-sum] within limit - shipped local")
        return draft
    if draft:
        shorter = local.chat(
            f"Shorten this to at most {target} words, keeping the meaning and "
            f"format. Output only the result.\n\n{draft}",
            max_tokens=112, deadline=local_deadline)
        if shorter and localgate.word_count(shorter) <= limit:
            _log("[local-sum] compressed within limit - shipped local")
            return shorter
    _log("[local-sum] over limit -> remote")
    return None


def _try_local(local, category: str, prompt: str, deadline,
               local_only: bool = False) -> str:
    """Dispatch to the verified local path for this category, if the active
    server's role and the enabled feature set cover it. None -> remote."""
    if local is None or not local.available:
        return None
    feats = local.features
    if category in ("code-debug", "code-gen") and local.mode != "off":
        return _try_local_code(local, prompt, category, deadline,
                               ship_unverified=local_only)
    if local.role == "coder":
        if category == "math" and "math" in feats:
            return _try_local_math(local, prompt, deadline, local_only=local_only)
        return None
    # general model phase
    if category == "math" and "math" in feats and local_only:
        # LOCAL_ONLY phases math onto the general model (stronger reasoner).
        return _try_local_math(local, prompt, deadline, local_only=True)
    if category == "sentiment" and "sentiment" in feats:
        return _try_local_sentiment(local, prompt, deadline)
    if category == "ner" and "ner" in feats:
        return _try_local_ner(local, prompt, deadline)
    if category == "summarization" and "sum" in feats:
        return _try_local_sum(local, prompt, deadline)
    if category == "logic" and "logic" in feats:
        return _try_local_short_agree(local, prompt, deadline, "local-logic",
                                      local_only=local_only)
    # factual deliberately has NO verified path: agreement games truncate
    # multi-part answers and program answers are nonsense for recall
    # (measured: factual 7/7 via the rich raw path vs 4/7 via short-agree).
    # In LOCAL_ONLY answer_task falls through to _local_raw for it.
    return None


# ---------------- entry point ----------------

def _local_raw(local, category: str, prompt: str, deadline, spec) -> str:
    """LOCAL_ONLY last resort: the best unverified local answer. Zero proxy
    tokens is the whole point of that mode; an unverified local answer beats
    a static fallback string."""
    if local is None or not local.available:
        return ""
    local_deadline = min(deadline, time.monotonic() + 70.0)
    if category in ("math", "logic"):
        # Reasoning tail: think, then extract the answer line. NEVER reuse the
        # remote-tuned tiny caps here — a local model's preamble would truncate
        # into garbage (measured: math answers chopped at 16 tokens). Factual
        # is NOT in this branch: answer-line extraction strips required detail
        # ("Canberra" instead of the two-sentence answer the judge wants).
        reply = local.chat(prompt + _COT_SHORT_SUFFIX, max_tokens=704,
                           deadline=local_deadline)
        ans = _extract_final_answer(reply)
        if ans:
            _log(f"[local-raw] {category} thinking answer shipped")
            return ans
    reply = local.chat(prompt + spec["suffix"],
                       max_tokens=max(spec["max_tokens"], 96),
                       deadline=local_deadline)
    if reply:
        _log(f"[local-raw] {category} raw local answer shipped")
    return (reply or "").strip()


def answer_task(client, ladder, prompt: str, deadline, local=None,
                category: str = None, local_only: bool = False) -> str:
    """Route, shape, verify. Returns the answer text or "" when nothing
    succeeded (caller applies the static fallback)."""
    if category is None:
        category = detect(prompt)
    spec = spec_for(category)
    _log(f"[route] category={category} cap={spec['max_tokens']}")

    ans = _try_local(local, category, prompt, deadline, local_only=local_only)
    if ans:
        return ans  # verified locally -> zero proxy tokens

    if local_only:
        # NEVER call remote in this mode - the token score must stay 0.
        return _local_raw(local, category, prompt, deadline, spec)

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
