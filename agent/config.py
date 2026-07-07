"""Runtime configuration.

Everything comes from the environment at container start (contract:
FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS). Nothing in this
package hardcodes a model id or endpoint; the constants below are routing
preferences that get resolved against whatever ALLOWED_MODELS actually says.
"""

import os

# Preferred routing order, best-first, resolved against ALLOWED_MODELS at
# runtime (suffix match; entries absent from ALLOWED_MODELS are dropped).
# minimax-m3 leads while it is the only verified-reachable serverless model;
# the planned flip to gemma-4-31b-it-first happens after the Wednesday bench
# confirms gemma is reachable through the graded proxy.
# kimi-k2p7-code is deliberately last: its mandatory thinking makes it a
# token liability, so it is reachable only when everything else is gone.
MODEL_PREFERENCE = [
    "minimax-m3",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it-nvfp4",
    "kimi-k2p7-code",
]

TOTAL_BUDGET_S = 570.0    # 9.5 min of the 10-min cap; the rest is exit margin
WRITE_MARGIN_S = 8.0      # always reserve this much for the final write
PER_TASK_CAP_S = 28.0     # hard per-task ceiling: remote + local + retry
PER_TASK_FLOOR_S = 3.0
MAX_TOKENS_DEFAULT = 512

FALLBACK_ANSWER = "Unable to complete this task."

# Container-contract paths; the MVE_* overrides exist only for host-side dev
# runs on Windows and are never set by the harness.
INPUT_PATH = os.environ.get("MVE_INPUT", "/input/tasks.json")
OUTPUT_DIR = os.environ.get("MVE_OUTPUT_DIR", "/output")


def _last_segment(model_id: str) -> str:
    return model_id.rstrip("/").rsplit("/", 1)[-1].strip().lower()


def resolve_models(allowed_raw: str) -> list:
    """Map ALLOWED_MODELS (comma-separated; bare ids or full account paths)
    onto MODEL_PREFERENCE via suffix match. Returns the allowed entries
    verbatim, best-first. Allowed entries we don't recognize go last rather
    than being dropped — never discard a legal option."""
    entries = [e.strip() for e in (allowed_raw or "").split(",") if e.strip()]
    by_suffix = {}
    for e in entries:
        by_suffix.setdefault(_last_segment(e), e)
    ladder = [by_suffix[p.lower()] for p in MODEL_PREFERENCE if p.lower() in by_suffix]
    known = set(ladder)
    ladder += [e for e in entries if e not in known]
    return ladder


class Settings:
    def __init__(self, env=None):
        env = env if env is not None else os.environ
        self.api_key = env.get("FIREWORKS_API_KEY", "").strip()
        self.base_url = env.get("FIREWORKS_BASE_URL", "").strip().rstrip("/")
        self.allowed_raw = env.get("ALLOWED_MODELS", "").strip()
        self.models = resolve_models(self.allowed_raw)

    @property
    def online(self) -> bool:
        return bool(self.api_key and self.base_url and self.models)

    def describe(self) -> str:
        # Never include the key itself anywhere in logs.
        return (
            f"base_url={'set' if self.base_url else 'MISSING'} "
            f"api_key={'set' if self.api_key else 'MISSING'} "
            f"models={self.models if self.models else 'MISSING'}"
        )
