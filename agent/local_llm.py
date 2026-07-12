"""Local llama.cpp server manager (Stages C+D).

Spawns the bundled runtime-dispatch `llama-server` as a subprocess and drives
it over localhost HTTP (OpenAI-compatible). In-container local inference
records ZERO Fireworks proxy tokens.

Stage D runs up to two models — a coder (code + math-PoT) and a general
instruct model (sentiment / ner / summarization) — but only ONE server lives
at a time: the 4 GB grading box cannot hold both resident. main.py phases the
tasks and swaps servers at the phase boundary.

Everything here degrades gracefully: if the layer is disabled, the binary or
model is missing, the server fails to become healthy, or a request errors, the
manager reports `available=False`/returns "" and the agent falls back to the
remote pipeline. It can never make the container worse than remote-only.
"""

import json
import os
import subprocess
import sys
import time

import httpx


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class LocalLLM:
    """One llama-server instance for one model role.

    role="coder"   -> settings.local_model  (code, code+, math features)
    role="general" -> settings.local_model2 (sentiment, ner, sum features)
    """

    def __init__(self, settings, role: str = "coder"):
        self.available = False
        self.gen_tps = None
        self.proc = None
        self.role = role
        self.features = settings.local_features
        self.code_mode = settings.code_mode
        self.base = f"http://127.0.0.1:{settings.local_port}"
        self._client = None

        model = settings.local_model if role == "coder" else settings.local_model2
        server = settings.local_server
        if not self.features:
            _log("[local] layer disabled (LOCAL_LAYER=off)")
            return
        if not (server and os.path.exists(server)):
            _log(f"[local] server binary absent -> {role} disabled")
            return
        if not (model and os.path.exists(model)):
            _log(f"[local] {role} model absent -> disabled")
            return
        try:
            self.proc = subprocess.Popen(
                [server, "--model", model, "--threads", str(settings.local_threads),
                 "--threads-batch", str(settings.local_threads),
                 "--ctx-size", str(settings.local_ctx),
                 "--host", "127.0.0.1", "--port", str(settings.local_port),
                 "--no-webui"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            _log(f"[local] {role} spawn failed: {type(e).__name__} -> disabled")
            self.proc = None
            return

        self._client = httpx.Client(timeout=httpx.Timeout(30.0, connect=2.0))
        if self._wait_health(settings.local_boot_budget_s):
            self.available = True
            _log(f"[local] {role} server healthy, layer={','.join(sorted(self.features))}")
            self._probe_speed()
        else:
            _log(f"[local] {role} server did not become healthy -> disabled (remote only)")
            self.shutdown()

    # generation speed (tokens/second, EWMA across calls). The grading box's
    # vCPUs can be far slower than the bench threads (--cpus=1 replay of the
    # 9/19 live fail: 11/19 deadline-cut fragments) — pipelines consult
    # `slow` to skip optional passes and shrink token asks so generations
    # COMPLETE instead of getting cut.
    _SLOW_TPS = 13.0

    @property
    def slow(self) -> bool:
        return self.gen_tps is not None and self.gen_tps < self._SLOW_TPS

    def _probe_speed(self) -> None:
        """One tiny generation right after boot so task 1 already knows the
        speed mode. ~1-3 s; inside the run budget, not the boot budget."""
        self.gen_tps = None
        try:
            self.chat("Count from 1 to 30, digits separated by spaces.",
                      max_tokens=24, deadline=time.monotonic() + 8.0)
            if self.gen_tps is not None:
                _log(f"[local] {self.role} speed probe: {self.gen_tps:.1f} tok/s"
                     f"{' (SLOW mode)' if self.slow else ''}")
        except Exception:
            pass

    @property
    def mode(self) -> str:
        """Stage C compatibility: code-gate strictness while this server is
        the active one ("off" unless the coder is up)."""
        return self.code_mode if (self.available and self.role == "coder") else "off"

    def _wait_health(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:  # server process already exited
                return False
            try:
                r = self._client.get(f"{self.base}/health", timeout=2.0)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.4)
        return False

    def chat(self, prompt: str, max_tokens: int, deadline: float,
             temperature: float = 0.0) -> str:
        """One local completion bounded by an absolute time.monotonic deadline.
        Returns "" (or the partial text) on any failure.

        STREAMING on purpose: a non-streaming call that hits the client
        timeout leaves llama-server generating the full completion on its
        single slot, wedging every follow-up call — observed as empty
        answers (fresh-124: 3 logic tasks whose grace retries all queued
        behind a dead 704-token CoT) and, at 2 vCPU where generation is
        minutes-slow, a poisoned-task cascade. Closing the stream aborts
        the server-side generation within a token."""
        if not self.available:
            return ""
        budget = deadline - time.monotonic()
        if budget < 1.0:
            return ""
        parts = []
        first_tok_t = last_tok_t = None
        try:
            with self._client.stream(
                    "POST", f"{self.base}/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens,
                          "temperature": temperature,
                          "stream": True},
                    timeout=httpx.Timeout(min(budget, 28.0), connect=2.0)) as r:
                if r.status_code != 200:
                    return ""
                for line in r.iter_lines():
                    if time.monotonic() > deadline:
                        break  # exit the with-block -> server aborts
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"]
                    except (ValueError, KeyError, IndexError):
                        continue
                    parts.append(delta.get("content") or "")
                    last_tok_t = time.monotonic()
                    if first_tok_t is None:
                        first_tok_t = last_tok_t
            self._record_speed(len(parts), first_tok_t, last_tok_t)
            return "".join(parts).strip()
        except httpx.ConnectError:
            # server died mid-run (e.g. an inference-time crash) -> stop trying
            # it; remaining tasks skip straight to remote.
            if self.proc is None or self.proc.poll() is not None:
                self.available = False
            return "".join(parts).strip()
        except Exception:
            # partial text beats empty - every caller re-verifies content
            self._record_speed(len(parts), first_tok_t, last_tok_t)
            return "".join(parts).strip()

    def _record_speed(self, n_tokens: int, first_t, last_t) -> None:
        """EWMA of pure-generation speed (first chunk -> last chunk)."""
        if first_t is None or last_t is None or n_tokens < 8:
            return
        dt = last_t - first_t
        if dt <= 0.05:
            return
        tps = (n_tokens - 1) / dt
        prev = getattr(self, "gen_tps", None)
        self.gen_tps = tps if prev is None else (0.6 * prev + 0.4 * tps)

    def shutdown(self) -> None:
        self.available = False
        if self.proc is not None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    self.proc.kill()
            except Exception:
                pass
            self.proc = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
