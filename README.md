# minimum viable effort — track 1 agent

solo entry for the AMD Developer Hackathon: ACT II, track 1 (general-purpose AI agent).

a small agent. it reads `/input/tasks.json`, answers each task, and writes
`/output/results.json`. the current build is a **verified-local hybrid**:
prompts are routed by category and answered first by bundled llama.cpp
models; a local answer ships only after passing a deterministic check, and
tasks that can't be verified locally escalate as terse calls through
`FIREWORKS_BASE_URL` to a model from `ALLOWED_MODELS`.

## how it works

- **routing**: a zero-cost regex router classifies each prompt into one of 8
  task shapes (math, logic, sentiment, summarization, NER, code-gen,
  code-debug, factual) and picks the response shape and budget.
- **local models** (bundled, served one-at-a-time by `llama-server` on
  localhost): a coder model for code tasks and a general instruct model for
  everything else.
- **verify before shipping**: math runs program-of-thought — the model writes
  an arithmetic expression *and* a tiny program; the container evaluates both
  locally and ships only on cross-method agreement. code ships only after
  passing execution self-tests in a hardened sandbox. NER entities are
  verified verbatim-in-source; summaries are word-limit-verified; logic
  puzzles are brute-forced by generated constraint-enumeration programs.
  unverifiable answers fall back to the best available sample — never an
  empty answer.
- **escalation policy** (`HYBRID_POLICY=h3`): factual recall always goes
  remote (small local models hallucinate on open-domain facts); code, math,
  NER, sentiment and summaries ship locally only when their verification
  passes; summaries below a content floor re-draft remotely; logic
  escalations run the remote model with thinking enabled. escalated calls
  stay terse — capped completions, no system-prompt mass.
- **budget discipline**: a global watchdog fair-shares the runtime budget so
  the batch always completes inside the contest limits, and `results.json`
  is written atomically after every task (always valid JSON, exit 0).

the same codebase also builds a pure zero-token configuration (local
inference only — **no network calls at all**, validated with
`--network none`) selected at build time via `LOCAL_LAYER` / `LOCAL_ONLY`
build args. the submitted image is the hybrid build.

## docker image

```
ghcr.io/lippinc/mve-agent:final3
```

## how it runs

```
docker run --rm \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=... \
  -e ALLOWED_MODELS=... \
  ghcr.io/lippinc/mve-agent:final3
```

all configuration comes from the environment. no keys or answers are baked
into the image.

## bundled models & licenses

- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
  (Q4_K_M GGUF) — Apache License 2.0
- [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
  (Q4_K_M GGUF) — Apache License 2.0
- [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` — MIT License

## submission items

demo video and slide deck accompany the lablab submission.
