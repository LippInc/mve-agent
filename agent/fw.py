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


class FireworksClient:
    def __init__(self, settings):
        self.s = settings
        self.http = httpx.Client(timeout=httpx.Timeout(25.0, connect=5.0))
        self.ledger = {
            "calls": 0, "retries": 0, "errors": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        self.leak_strikes = {}   # model id -> strike count
        self.dead_models = set()

    def leak_suspect(self, model: str) -> bool:
        return self.leak_strikes.get(model, 0) >= 2

    def chat(self, model, messages, deadline, max_tokens=512, temperature=0.0,
             stop=None, thinking=None, response_format=None) -> CallResult:
        """One chat completion bounded by an absolute time.monotonic deadline.
        Retries transient failures with backoff while time remains; strips
        optional params once on a 400 that names them."""
        body = {
            "model": model,
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

        attempt = 0
        stripped = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining < 1.5 or attempt >= 4:
                return CallResult()
            attempt += 1
            t0 = time.monotonic()
            try:
                r = self.http.post(
                    f"{self.s.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.s.api_key}"},
                    json=body,
                    timeout=httpx.Timeout(
                        max(1.0, min(remaining - 0.5, 25.0)),
                        connect=max(1.0, min(5.0, remaining - 0.5)),
                    ),
                )
            except httpx.HTTPError as e:
                self.ledger["errors"] += 1
                log(f"[call-error] model={model} attempt={attempt} {type(e).__name__}")
                self._backoff(attempt, deadline)
                self.ledger["retries"] += 1
                continue

            elapsed = time.monotonic() - t0
            if r.status_code == 200:
                return self._handle_200(r, model, elapsed)
            if r.status_code in (400, 404, 422):
                text = r.text[:500]
                if not stripped and any(k in body for k in _STRIPPABLE) \
                        and any(k in text for k in _STRIPPABLE):
                    for k in _STRIPPABLE:
                        body.pop(k, None)
                    stripped = True
                    log(f"[param-strip] model={model} http={r.status_code}")
                    continue
                if "model" in text.lower():
                    self.dead_models.add(model)
                    log(f"[model-dead] model={model} http={r.status_code}")
                    return CallResult(model_dead=True)
                self.ledger["errors"] += 1
                log(f"[call-fail] model={model} http={r.status_code}")
                return CallResult()
            # 429 / 5xx / anything else: transient, back off and retry
            self.ledger["errors"] += 1
            log(f"[call-retryable] model={model} http={r.status_code} attempt={attempt}")
            self._backoff(attempt, deadline)
            self.ledger["retries"] += 1

    def _backoff(self, attempt, deadline):
        pause = min(0.5 * (2 ** (attempt - 1)), 4.0)
        pause = min(pause, max(0.0, deadline - time.monotonic() - 1.0))
        if pause > 0:
            time.sleep(pause)

    def _handle_200(self, r, model, elapsed) -> CallResult:
        try:
            data = r.json()
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, ValueError, TypeError):
            self.ledger["errors"] += 1
            log(f"[bad-response-shape] model={model}")
            return CallResult()

        usage = data.get("usage") or {}
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        t = int(usage.get("total_tokens") or (p + c))
        self.ledger["calls"] += 1
        self.ledger["prompt_tokens"] += p
        self.ledger["completion_tokens"] += c
        self.ledger["total_tokens"] += t

        vis = _visible_token_estimate(content)
        log(f"[call] model={model} t={elapsed:.1f}s prompt={p} completion={c} "
            f"total={t} visible~{vis}")
        # Leak detection: completion tokens far beyond the visible answer
        # means hidden reasoning is being billed.
        if c > max(3 * vis, vis + 100):
            self.leak_strikes[model] = self.leak_strikes.get(model, 0) + 1
            log(f"[leak-suspect] model={model} strikes={self.leak_strikes[model]} "
                f"completion={c} visible~{vis}")
        return CallResult(ok=True, content=content)
