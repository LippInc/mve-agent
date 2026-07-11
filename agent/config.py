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


def _to_callable(model_id: str) -> str:
    """Fireworks expects the full 'accounts/fireworks/models/<name>' path.
    ALLOWED_MODELS may arrive as bare names or full account paths; normalize
    bare names to the full path and pass anything already containing an
    account path through untouched. fw.py toggles the two forms at call time
    if this guess 404s, so we are robust to whichever the graded proxy wants."""
    m = model_id.strip().rstrip("/")
    if "/" in m:
        return m
    return f"accounts/fireworks/models/{m}"


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
    return [_to_callable(m) for m in ladder]


# Local-inference layer (kill-switchable). Default OFF: the image only runs
# local inference when LOCAL_LAYER is explicitly set, so the safe fallback
# behaviour (remote-only) is what ships unless we opt in.
#
# LOCAL_LAYER is a comma-separated feature set; each feature is independently
# kill-switchable so accuracy risk is dialed per category:
#   code   — local code answers gated on prompt-derived example tests (coder model)
#   code+  — code, plus model-authored self-test gates (higher hit rate)
#   math   — local Program-of-Thought, two independent derivations must agree
#   sentiment — local label, two independent framings must agree (general model)
#   ner    — local extraction, substring-verified + two-sample set agreement
#   sum    — local summaries, hard word-limit verified (riskiest; margin-gated)
# Legacy single values "off" / "code" / "code+" keep their Stage C meaning.
LOCAL_LAYER_DEFAULT = "off"
LOCAL_SERVER_DEFAULT = "/opt/llamacpp/llama-server"
LOCAL_MODEL_DEFAULT = "/models/model.gguf"
LOCAL_MODEL2_DEFAULT = "/models/general.gguf"

_KNOWN_FEATURES = {"code", "code+", "math", "sentiment", "ner", "sum",
                   "logic", "factual"}

# Which features run on which model server. Coder-model features run in
# phase 1, general-model features in phase 2 (one server lives at a time —
# the 4 GB grading box cannot hold both models resident).
CODER_FEATURES = {"code", "code+", "math"}
GENERAL_FEATURES = {"sentiment", "ner", "sum", "logic", "factual"}


def parse_local_features(raw: str) -> frozenset:
    """'off'/'' -> empty set; otherwise the recognized feature names present.
    Unknown tokens are dropped (a typo can never accidentally widen the risk
    surface)."""
    toks = {t.strip().casefold() for t in (raw or "").split(",")}
    return frozenset(t for t in toks if t in _KNOWN_FEATURES)


class Settings:
    def __init__(self, env=None):
        env = env if env is not None else os.environ
        self.api_key = env.get("FIREWORKS_API_KEY", "").strip()
        self.base_url = env.get("FIREWORKS_BASE_URL", "").strip().rstrip("/")
        self.allowed_raw = env.get("ALLOWED_MODELS", "").strip()
        self.models = resolve_models(self.allowed_raw)

        self.local_layer = env.get("LOCAL_LAYER", LOCAL_LAYER_DEFAULT).strip().lower()
        self.local_features = parse_local_features(self.local_layer)
        # LOCAL_ONLY=1: the zero-proxy-token play. NEVER call the remote API —
        # ship the best local answer (verified path first, raw fallback second).
        # The token score is exactly 0; accuracy rides entirely on the bundled
        # models, so this only ever ships when the bench says the local stack
        # clears the gate with margin.
        self.local_only = env.get("LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes")
        # HYBRID_POLICY (Basket B, ignored when LOCAL_ONLY): "" = legacy
        # hybrid; "h1" = insured-local (factual always-remote, logic ships
        # only on agreement); "h2" = h1 but logic also ships its best single
        # local candidate instead of escalating (cheaper, slightly riskier);
        # "h3" = h2 everywhere except logic, whose escalation runs
        # thinking-ENABLED (fresh-puzzle probe 2026-07-11: m3 terse 0/6 vs
        # m3 thinking 6/6 @ ~522 tok/task - the suppression, not the model,
        # was the fresh-logic failure).
        self.hybrid_policy = env.get("HYBRID_POLICY", "").strip().lower()
        if self.hybrid_policy not in ("", "h1", "h2", "h3"):
            self.hybrid_policy = ""
        # Per-task ceiling: 28 s keeps every remote API call comfortably under
        # the 30 s/request rule. LOCAL_ONLY makes zero API calls, so only the
        # 10-minute total binds — allow long local thinking on the hard tail
        # (the global watchdog + fair-share budgeting still bound the total).
        self.per_task_cap_s = 75.0 if self.local_only else PER_TASK_CAP_S
        self.local_server = env.get("LOCAL_SERVER", LOCAL_SERVER_DEFAULT).strip()
        self.local_model = env.get("LOCAL_MODEL", LOCAL_MODEL_DEFAULT).strip()
        self.local_model2 = env.get("LOCAL_MODEL2", LOCAL_MODEL2_DEFAULT).strip()
        try:
            self.local_port = int(env.get("LOCAL_PORT", "8080"))
        except ValueError:
            self.local_port = 8080
        try:
            self.local_threads = int(env.get("LOCAL_THREADS", "2"))
        except ValueError:
            self.local_threads = 2
        self.local_ctx = 4096
        self.local_boot_budget_s = 40.0

    @property
    def code_mode(self) -> str:
        """Stage C compatibility: the code-gate strictness implied by the
        feature set ("off" | "code" | "code+")."""
        if "code+" in self.local_features:
            return "code+"
        if "code" in self.local_features:
            return "code"
        return "off"

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
