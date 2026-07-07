"""Entry point. Contract: read /input/tasks.json, answer through the
Fireworks API (FIREWORKS_BASE_URL, ALLOWED_MODELS), write /output/results.json
as a list of {"task_id": str, "answer": str}, exit 0.

Never-regress rails implemented here:
- results.json is valid, complete JSON from the first seconds of the run
  (fallback-filled, atomically rewritten after every answered task)
- global watchdog: 9.5-minute soft budget with a reserved write margin
- fair-share per-task budget, capped at 28 s, spanning remote + retry
- answers are always non-empty strings; exit code 0 whenever output exists
"""

import json
import os
import sys
import time

BOOT = time.monotonic()
print(f"[boot] agent starting", file=sys.stderr, flush=True)

from agent.config import (  # noqa: E402
    FALLBACK_ANSWER,
    INPUT_PATH,
    OUTPUT_DIR,
    PER_TASK_CAP_S,
    PER_TASK_FLOOR_S,
    Settings,
    TOTAL_BUDGET_S,
    WRITE_MARGIN_S,
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_tasks(path: str) -> list:
    """Tolerates task_id/id drift and a {"tasks": [...]} wrapper; returns
    [] (not an exception) on anything unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        log(f"[input-error] {type(e).__name__}: cannot read {path}")
        return []
    if isinstance(data, dict):
        data = data.get("tasks", [])
    if not isinstance(data, list):
        log("[input-error] tasks payload is not a list")
        return []
    tasks = []
    for i, t in enumerate(data):
        if not isinstance(t, dict):
            continue
        tid = t.get("task_id") or t.get("id") or f"task-{i + 1}"
        prompt = t.get("prompt") or t.get("question") or t.get("input") or ""
        tasks.append({"task_id": str(tid), "prompt": str(prompt)})
    return tasks


def atomic_write_results(results_in_order: list) -> bool:
    """Write the full results list atomically so a kill at any moment leaves
    valid, complete JSON on disk."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tmp = os.path.join(OUTPUT_DIR, ".results.tmp")
        final = os.path.join(OUTPUT_DIR, "results.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results_in_order, f, ensure_ascii=False)
        os.replace(tmp, final)
        return True
    except OSError as e:
        log(f"[output-error] {type(e).__name__}: {e}")
        return False


def main() -> int:
    settings = Settings()
    log(f"[config] {settings.describe()}")

    tasks = read_tasks(INPUT_PATH)
    log(f"[tasks] n={len(tasks)}")

    # Insurance first: a complete, valid results file exists before any
    # network call is attempted.
    answers = {t["task_id"]: FALLBACK_ANSWER for t in tasks}
    order = [t["task_id"] for t in tasks]

    def results_list():
        return [{"task_id": tid, "answer": answers[tid]} for tid in order]

    wrote = atomic_write_results(results_list())

    client = None
    ladder = []
    if settings.online:
        # Deferred import keeps boot instant even if httpx grows deps later.
        from agent.fw import FireworksClient
        from agent import pipelines
        client = FireworksClient(settings)
        ladder = list(settings.models)
    else:
        log("[offline] missing env config - shipping fallback answers")

    deadline_global = BOOT + TOTAL_BUDGET_S
    for i, t in enumerate(tasks):
        if client is None:
            break
        now = time.monotonic()
        remaining = deadline_global - now - WRITE_MARGIN_S
        n_left = len(tasks) - i
        if remaining < PER_TASK_FLOOR_S:
            log(f"[watchdog] global budget exhausted at task {i + 1}/{len(tasks)}")
            break
        budget = min(PER_TASK_CAP_S, max(PER_TASK_FLOOR_S, 0.9 * remaining / n_left))
        try:
            current = pipelines.order_ladder(client, ladder)
            if not current:
                log("[watchdog] no models left alive")
                break
            ans = pipelines.answer_task(client, current, t["prompt"], now + budget)
            if ans:
                answers[t["task_id"]] = str(ans)
        except Exception as e:  # per-task isolation: one bad task never kills the run
            log(f"[task-error] {t['task_id']} {type(e).__name__}: {e}")
        atomic_write_results(results_list())

    wrote = atomic_write_results(results_list()) or wrote

    if client is not None:
        led = client.ledger
        log(f"[ledger] calls={led['calls']} retries={led['retries']} "
            f"errors={led['errors']} prompt={led['prompt_tokens']} "
            f"completion={led['completion_tokens']} total={led['total_tokens']}")
    log(f"[done] wall={time.monotonic() - BOOT:.1f}s wrote={wrote}")
    return 0 if wrote else 1


if __name__ == "__main__":
    sys.exit(main())
