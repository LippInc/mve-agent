"""Floor pipeline: one terse single-shot call per task.

The eight per-category pipelines (math/logic/code/sentiment/NER/summarization/
factual/debug with local verification) slot in here without touching main.py:
answer_task is the single entry point and already receives everything it needs.
"""

import time

from agent.config import MAX_TOKENS_DEFAULT


def _is_minimax(model: str) -> bool:
    return "minimax" in model.rsplit("/", 1)[-1].lower()


def _model_params(model: str) -> dict:
    # thinking is a token optimization, never load-bearing: fw.py strips it
    # on a 400 and the call still succeeds.
    if _is_minimax(model):
        return {"thinking": {"type": "disabled"}}
    return {}


def answer_task(client, ladder, prompt: str, deadline) -> str:
    """Try models best-first within the deadline. Returns the answer text or
    "" when nothing succeeded (caller applies the static fallback)."""
    for model in list(ladder):
        if model in client.dead_models:
            continue
        res = client.chat(
            model,
            [{"role": "user", "content": prompt}],
            deadline,
            max_tokens=MAX_TOKENS_DEFAULT,
            **_model_params(model),
        )
        if res.ok and res.content.strip():
            return res.content.strip()
        # model_dead → next model; transient failure with time left → next
        # model too (correlated-failure insurance); out of time → give up
        if deadline - time.monotonic() < 2.0:
            break
    return ""


def order_ladder(client, ladder):
    """Stable reorder: alive models first, leak suspects after clean ones."""
    alive = [m for m in ladder if m not in client.dead_models]
    clean = [m for m in alive if not client.leak_suspect(m)]
    suspect = [m for m in alive if client.leak_suspect(m)]
    return clean + suspect
