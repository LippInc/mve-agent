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
        tid = t.get("task_id")
        if tid is None:
            tid = t.get("id")
        if tid is None:
            tid = f"task-{i + 1}"
        prompt = t.get("prompt")
        if prompt is None:
            prompt = t.get("question")
        if prompt is None:
            prompt = t.get("input")
        if prompt is None:
            prompt = ""
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
            json.dump(results_in_order, f, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, final)
        return True
    except Exception as e:  # any write failure must not crash the exit-0 rail
        log(f"[output-error] {type(e).__name__}")
        return False


def main() -> int:
    settings = Settings()
    log(f"[config] {settings.describe()}")

    tasks = read_tasks(INPUT_PATH)
    log(f"[tasks] n={len(tasks)}")

    # Insurance first: a complete, valid results file exists before any
    # network call is attempted.
    answers = {t["task_id"]: FALLBACK_ANSWER for t in tasks}
    order = []
    _seen = set()
    for t in tasks:
        tid = t["task_id"]
        if tid not in _seen:
            _seen.add(tid)
            order.append(tid)

    def results_list():
        return [{"task_id": tid, "answer": answers[tid]} for tid in order]

    wrote = atomic_write_results(results_list())

    client = None
    ladder = []
    pipelines = None
    if settings.online:
        # Deferred import keeps boot instant even if httpx grows deps later.
        # Guarded so a broken import/init can never defeat the exit-0 rail:
        # on any failure we ship the fallback answers already on disk.
        try:
            from agent.fw import FireworksClient
            from agent import pipelines as _pl
            pipelines = _pl
            client = FireworksClient(settings)
            ladder = list(settings.models)
        except Exception as e:
            client = None
            log(f"[init-error] {type(e).__name__} - shipping fallback answers")
    else:
        log("[offline] missing env config - shipping fallback answers")
    if pipelines is None and settings.local_only:
        # LOCAL_ONLY needs no remote path at all - load the pipeline anyway.
        try:
            from agent import pipelines as _pl
            pipelines = _pl
        except Exception as e:
            log(f"[init-error] {type(e).__name__} (local-only)")

    # A run is workable with a remote client OR in local-only mode.
    runnable = pipelines is not None and (client is not None or settings.local_only)

    # Local-inference layer (off unless LOCAL_LAYER is set). Two model roles,
    # one server alive at a time (4 GB box): tasks are processed in two phases
    # — coder-model categories first (code, math), then everything else with
    # the general model. Fully guarded so it can never defeat the exit-0 rail;
    # only spawned when we actually have a remote path to fall back on.
    local = None
    coder_needed = general_needed = False
    if runnable and settings.local_features:
        try:
            from agent.config import CODER_FEATURES, GENERAL_FEATURES
            coder_needed = bool(settings.local_features & CODER_FEATURES)
            general_needed = bool(settings.local_features & GENERAL_FEATURES)
        except Exception:
            pass

    # Phase order: coder-category tasks first, original order inside each
    # phase. Output order is unaffected (results keyed by task_id).
    # In LOCAL_ONLY, math rides the GENERAL phase: the bigger instruct model
    # is the stronger word-problem reasoner and there is no remote to catch
    # coder slips (measured: coder-PoT 5/7 vs general-model target 7/7).
    CODER_CATS = (("code-debug", "code-gen") if settings.local_only
                  else ("code-debug", "code-gen", "math"))
    if runnable:
        try:
            from agent.categories import detect
            for t in tasks:
                t["category"] = detect(t["prompt"])
        except Exception:
            for t in tasks:
                t["category"] = None
        if coder_needed or general_needed:
            tasks = ([t for t in tasks if t["category"] in CODER_CATS]
                     + [t for t in tasks if t["category"] not in CODER_CATS])

    def start_local(role):
        try:
            from agent.local_llm import LocalLLM
            inst = LocalLLM(settings, role=role)
            return inst if inst.available else None
        except Exception as e:
            log(f"[local] init guard ({role}): {type(e).__name__} - remote only")
            return None

    if runnable and coder_needed:
        local = start_local("coder")

    deadline_global = BOOT + TOTAL_BUDGET_S
    general_started = False
    for i, t in enumerate(tasks):
        if not runnable:
            break
        # Phase boundary: first non-coder-category task -> swap servers.
        if (general_needed and not general_started
                and t.get("category") not in CODER_CATS):
            general_started = True
            if local is not None:
                local.shutdown()
                local = None
            # The swap is the one block that can overrun the global budget
            # unwatched (shutdown <=5s + boot <=local_boot_budget_s). Boot
            # (and retry a failed boot once) only while the remaining budget
            # can absorb a worst-case boot; otherwise skip the swap so the
            # wall stays inside the runtime cap.
            swap_floor = settings.local_boot_budget_s + 20.0
            if deadline_global - time.monotonic() - WRITE_MARGIN_S > swap_floor:
                local = start_local("general")
                if (local is None and deadline_global - time.monotonic()
                        - WRITE_MARGIN_S > swap_floor):
                    log("[local] general boot failed - one retry")
                    local = start_local("general")
            else:
                log("[watchdog] skipping general swap (budget too low)")
        now = time.monotonic()
        remaining = deadline_global - now - WRITE_MARGIN_S
        n_left = len(tasks) - i
        if remaining < PER_TASK_FLOOR_S:
            log(f"[watchdog] global budget exhausted at task {i + 1}/{len(tasks)}")
            break
        budget = min(settings.per_task_cap_s,
                     max(PER_TASK_FLOOR_S, 0.9 * remaining / n_left))
        try:
            current = []
            if client is not None:
                current = pipelines.order_ladder(client, ladder)
                if not current and not settings.local_only:
                    log("[watchdog] no models left alive")
                    break
            ans = pipelines.answer_task(client, current, t["prompt"], now + budget,
                                        local=local, category=t.get("category"),
                                        local_only=settings.local_only,
                                        hybrid_policy=settings.hybrid_policy)
            if ans:
                answers[t["task_id"]] = str(ans)
        except Exception as e:  # per-task isolation: one bad task never kills the run
            log(f"[task-error] {t['task_id']} {type(e).__name__}")
        atomic_write_results(results_list())

    wrote = atomic_write_results(results_list()) or wrote

    if local is not None:
        local.shutdown()
    if client is not None:
        led = client.ledger
        log(f"[ledger] calls={led['calls']} retries={led['retries']} "
            f"errors={led['errors']} prompt={led['prompt_tokens']} "
            f"completion={led['completion_tokens']} total={led['total_tokens']}")
    log(f"[done] wall={time.monotonic() - BOOT:.1f}s wrote={wrote}")
    return 0 if wrote else 1


if __name__ == "__main__":
    sys.exit(main())
