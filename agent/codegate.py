"""Execution test-gate for locally-generated code.

A local model's code answer ships ONLY when it passes tests we can derive
deterministically from the prompt itself (worked examples the task states).
No derivable test -> we do NOT trust the local answer and fall back to the
remote pipeline. This keeps the local code path accuracy-safe on a zero-margin
accuracy gate: a shipped local answer has been executed and matches the
prompt's own stated behaviour; anything unverifiable pays remote tokens and
stays as correct as the remote model.

Pure stdlib. Everything here is best-effort and never raises to the caller.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile

_POSIX = os.name == "posix"
if _POSIX:
    import signal
    import resource


def _limits():
    # Bound a child that runs model-generated code: address space, CPU seconds,
    # process count (fork-bomb guard), file size. POSIX only.
    os.setsid()
    mb = 512 * 1024 * 1024
    for res, lim in ((resource.RLIMIT_AS, mb), (resource.RLIMIT_CPU, 8),
                     (resource.RLIMIT_FSIZE, 1 << 20), (resource.RLIMIT_NPROC, 64)):
        try:
            resource.setrlimit(res, (lim, lim))
        except Exception:
            pass


def _exec(script: str, timeout: float) -> str:
    """Run a Python script in a hardened, disposable subprocess. Returns stdout
    (capped) on clean exit, or None on non-zero exit / timeout / any failure.
    Never raises. Env is scrubbed of the API key; on POSIX the whole process
    group is killed on timeout so orphaned grandchildren can't linger."""
    if timeout < 0.5:
        return None
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
    kw = dict(stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
              cwd=tempfile.gettempdir(), env=env)
    if _POSIX:
        kw["preexec_fn"] = _limits
    else:
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        p = subprocess.Popen([sys.executable, "-I", "-X", "utf8", "-c", script], **kw)
    except Exception:
        return None
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if _POSIX:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            else:
                p.kill()
        except Exception:
            pass
        try:
            p.communicate(timeout=1)
        except Exception:
            pass
        return None
    except Exception:
        return None
    if p.returncode != 0:
        return None
    try:
        return (out or b"").decode("utf-8", "replace")[:65536]
    except Exception:
        return None

# name(args) [==|=|->|returns|should return|gives|yields|outputs] value
_CALL_EXPECT = re.compile(
    r"([A-Za-z_]\w*)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*"
    r"(?:==|=|->|=>|returns?|should\s+return|gives?|yields?|outputs?|produces?|:)\s*"
    r"([^\n;]+?)\s*(?=[\n;.]|$)",
    re.I)
# doctest style: >>> name(args)\n value
_DOCTEST = re.compile(r">>>\s*([A-Za-z_]\w*\s*\([^\n]*\))\s*\n\s*([^\n]+)")


def _valid_literal(s: str) -> bool:
    try:
        ast.literal_eval(s)
        return True
    except Exception:
        return False


def extract_example_tests(prompt: str, func: str = None, limit: int = 6) -> list:
    """Return a list of `EXPR == VALUE` assertion bodies pulled from worked
    examples in the prompt. Only keeps pairs whose call and expected value both
    parse (so we never build a broken or fabricated assertion)."""
    tests = []
    seen = set()

    def add(call: str, val: str):
        call, val = call.strip(), val.strip().rstrip(".")
        # strip surrounding prose like "the list [1, 2]" -> "[1, 2]" is hard;
        # require the expected value to be a bare Python literal to stay safe.
        if val.endswith(")") and not _valid_literal(val):
            val = val  # leave; may be a call, rejected below
        if not _valid_literal(val):
            return
        try:
            node = ast.parse(call.strip(), mode="eval")
        except Exception:
            return
        if not isinstance(node.body, ast.Call) or not isinstance(node.body.func, ast.Name):
            return
        if func and node.body.func.id != func:
            return
        key = (call, val)
        if key in seen:
            return
        seen.add(key)
        tests.append(f"assert ({call}) == ({val})")

    for m in _DOCTEST.finditer(prompt):
        add(m.group(1), m.group(2))
    for m in _CALL_EXPECT.finditer(prompt):
        add(f"{m.group(1)}({m.group(2)})", m.group(3))
    return tests[:limit]


def extract_code(answer: str) -> str:
    """Pull the Python code out of a model reply. Prefers fenced blocks that
    actually define something; falls back to the tail from the first def/class/
    import; else the whole answer."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.S | re.I)
    if blocks:
        def_blocks = [b for b in blocks if re.search(r"(?m)^\s*(def|class)\s+", b)]
        return "\n\n".join(def_blocks or blocks)
    m = re.search(r"(?m)^(def |class |import |from )", answer)
    if m:
        return answer[m.start():]
    return answer


def first_func_name(code: str) -> str:
    m = re.search(r"(?m)^\s*def\s+(\w+)\s*\(", code)
    return m.group(1) if m else None


def run_tests(code: str, tests: list, timeout: float = 6.0) -> bool:
    """Execute `code` followed by the assert lines in a hardened subprocess.
    True iff it exits cleanly with all assertions passing."""
    if not code or not code.strip() or not tests:
        return False
    script = code + "\n\n" + "\n".join(tests) + "\nprint('GATE_OK')\n"
    out = _exec(script, timeout)
    return out is not None and "GATE_OK" in out


def gate_code(prompt: str, code: str, timeout: float = 6.0) -> tuple:
    """Return (passed, n_tests). passed is True only if >=1 prompt-derived test
    was found AND the code executes them all successfully. (False, 0) means
    'no prompt-derived test -> unverifiable by this gate'."""
    if not code or not code.strip():
        return False, 0
    func = first_func_name(code)
    tests = extract_example_tests(prompt, func=func)
    if not tests:
        return False, 0
    return run_tests(code, tests, timeout), len(tests)
