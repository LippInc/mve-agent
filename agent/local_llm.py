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
        else:
            _log(f"[local] {role} server did not become healthy -> disabled (remote only)")
            self.shutdown()

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
        Returns "" on any failure (caller falls back to remote)."""
        if not self.available:
            return ""
        budget = deadline - time.monotonic()
        if budget < 1.0:
            return ""
        try:
            r = self._client.post(
                f"{self.base}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=httpx.Timeout(min(budget, 28.0), connect=2.0))
            if r.status_code != 200:
                return ""
            data = r.json()
            return (data["choices"][0]["message"].get("content") or "").strip()
        except httpx.ConnectError:
            # server died mid-run (e.g. an inference-time crash) -> stop trying
            # it; remaining tasks skip straight to remote.
            if self.proc is None or self.proc.poll() is not None:
                self.available = False
            return ""
        except Exception:
            return ""

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
