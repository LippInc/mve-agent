"""Generic Fireworks chat-completions wrapper.

Rails honored here:
- every request goes to {FIREWORKS_BASE_URL}/chat/completions and nowhere else
- the API key is sent in the Authorization header and never logged
- server-side params (thinking / response_format / stop) are token
  optimizations only: a 400 naming one of them triggers one retry without them
- per-call usage goes to the stderr ledger; a large gap between
  usage.completion_tokens and the visible answer length marks the model as a
  leak suspect (thinking billed invisibly) — two strikes demote it
"""

import sys
import time

import httpx


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _visible_token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


class CallResult:
    __slots__ = ("ok", "content", "model_dead")

    def __init__(self, ok=False, content="", model_dead=False):
        self.ok = ok
        self.content = content
        self.model_dead = model_dead


_STRIPPABLE = ("thinking", "response_format", "stop", "reasoning_effort")


def _alternate_model_form(model_id: str) -> str:
    """Toggle between the bare name and the full Fireworks account path, so a
    model-not-found can be retried in the form the graded proxy expects."""
    m = model_id.strip().rstrip("/")
    if "/" in m:
        return m.rsplit("/", 1)[-1]
    return f"accounts/fireworks/models/{m}"


class FireworksClient:
    def __init__(self, settings):
        self.s = settings
        self.http = httpx.Client(timeout=httpx.Timeout(22.0, connect=4.0))
        self.ledger = {
            "calls": 0, "retries": 0, "errors": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        self.leak_strikes = {}   # model id -> strike count
        self.dead_models = set()
        self.model_form = {}     # ladder id -> the id form that actually worked

    def leak_suspect(self, model: str) -> bool:
        return self.leak_strikes.get(model, 0) >= 2

    def chat(self, model, messages, deadline, max_tokens=512, temperature=0.0,
             stop=None, thinking=None, response_format=None,
             expect_reasoning=False) -> CallResult:
        """One chat completion bounded by an absolute time.monotonic deadline.
        Retries transient failures with backoff while time remains; strips
        optional params once on a 400 that names them."""
        body = {
            "model": self.model_form.get(model, model),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            body["stop"] = stop
        if thinking:
            body["thinking"] = thinking
        if response_format:
            body["response_format"] = response_format

        attempt = 0        # counts transient retries only (bounds the loop)
        stripped = False
        tried_alt = False
        while True:
            remaining = deadline - time.monotonic()
            # One transient retry only: retries re-bill on 200s, and 0/400+
            # bench calls ever needed more than one (ladder audit 2026-07-10).
            if remaining < 1.5 or attempt >= 2:
                return CallResult()
            t0 = time.monotonic()
            try:
                r = self.http.post(
                    f"{self.s.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.s.api_key}"},
                    json=body,
                    # connect + read are additive; keep the sum < 30 s/request
                    timeout=httpx.Timeout(
                        min(remaining - 0.5, 22.0),
                        connect=min(4.0, remaining - 0.5),
                    ),
                )
            except httpx.HTTPError as e:
                attempt += 1
                self.ledger["errors"] += 1
                self.ledger["retries"] += 1
                log(f"[call-error] model={model} attempt={attempt} {type(e).__name__}")
                self._backoff(attempt, deadline)
                continue

            elapsed = time.monotonic() - t0
            if r.status_code == 200:
                if body["model"] != model:
                    self.model_form[model] = body["model"]
                return self._handle_200(r, model, elapsed, expect_reasoning)
            if r.status_code in (400, 404, 422):
                tl = r.text[:500].lower()
                # 1) Strip only the optional param the error actually names.
                # Never strip a thinking-disable: dropping it silently re-arms
                # reasoning and bills a hidden-CoT token bomb on the retry.
                if not stripped:
                    named = [k for k in _STRIPPABLE
                             if k != "thinking" and k in body and k in tl]
                    if named:
                        for k in named:
                            body.pop(k, None)
                        stripped = True
                        log(f"[param-strip] model={model} http={r.status_code} "
                            f"dropped={','.join(named)}")
                        continue
                # 2) Model-not-found: a 404 always means the id form is wrong;
                # a 400/422 only if it names the model. Drive off the status
                # code, not a body substring (some proxies omit "model").
                if r.status_code == 404 or "model" in tl:
                    alt = _alternate_model_form(body["model"])
                    if not tried_alt and alt != body["model"]:
                        log(f"[model-form-retry] {body['model']} -> {alt}")
                        body["model"] = alt
                        tried_alt = True
                        continue
                    self.dead_models.add(model)
                    log(f"[model-dead] model={model} http={r.status_code}")
                    return CallResult(model_dead=True)
                self.ledger["errors"] += 1
                log(f"[call-fail] model={model} http={r.status_code}")
                return CallResult()
            # 429 / 5xx / anything else: transient, back off and retry
            attempt += 1
            self.ledger["errors"] += 1
            self.ledger["retries"] += 1
            log(f"[call-retryable] model={model} http={r.status_code} attempt={attempt}")
            self._backoff(attempt, deadline)

    def _backoff(self, attempt, deadline):
        pause = min(0.5 * (2 ** (attempt - 1)), 4.0)
        pause = min(pause, max(0.0, deadline - time.monotonic() - 1.0))
        if pause > 0:
            time.sleep(pause)

    def _handle_200(self, r, model, elapsed, expect_reasoning=False) -> CallResult:
        try:
            data = r.json()
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, ValueError, TypeError):
            self.ledger["errors"] += 1
            log(f"[bad-response-shape] model={model}")
            return CallResult()
        content = (msg.get("content") or "").strip()

        usage = data.get("usage") or {}
        try:
            p = int(usage.get("prompt_tokens") or 0)
            c = int(usage.get("completion_tokens") or 0)
            t = int(usage.get("total_tokens") or (p + c))
        except (ValueError, TypeError):
            # a malformed usage field must not discard an otherwise-good answer
            p = c = t = 0
        self.ledger["calls"] += 1
        self.ledger["prompt_tokens"] += p
        self.ledger["completion_tokens"] += c
        self.ledger["total_tokens"] += t

        vis = _visible_token_estimate(content)
        log(f"[call] model={model} t={elapsed:.1f}s prompt={p} completion={c} "
            f"total={t} visible~{vis}")
        # Leak detection on the VISIBLE content (before the reasoning fallback),
        # so a mandatory-thinking model (empty content, huge completion) is still
        # correctly flagged as billing hidden reasoning. Calls that DELIBERATELY
        # buy reasoning (h3 logic escalation) are exempt - their token bill is
        # priced in, and a strike here would wrongly demote the model for every
        # later terse call.
        if not expect_reasoning and c > max(3 * vis, vis + 100):
            self.leak_strikes[model] = self.leak_strikes.get(model, 0) + 1
            log(f"[leak-suspect] model={model} strikes={self.leak_strikes[model]} "
                f"completion={c} visible~{vis}")
        # Best-effort recovery: if the answer came back only in reasoning_content
        # (some mandatory-thinking models leave content empty), use it rather than
        # shipping the static fallback. Never fires on the validated minimax path.
        if not content:
            content = (msg.get("reasoning_content") or "").strip()
        return CallResult(ok=True, content=content)
