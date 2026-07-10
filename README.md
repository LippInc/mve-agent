# minimum viable effort — track 1 agent

solo entry for the AMD Developer Hackathon: ACT II, track 1 (general-purpose AI agent).

a small agent. it reads `/input/tasks.json`, answers each task, and writes
`/output/results.json`. the current build answers every task with **local
inference inside the container** (zero proxy tokens): prompts are routed by
category, answered by bundled llama.cpp models, and shipped only after
deterministic verification wherever one exists.

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
- **budget discipline**: a global watchdog fair-shares the runtime budget so
  the batch always completes inside the contest limits, and `results.json`
  is written atomically after every task (always valid JSON, exit 0).

the same codebase also supports a hybrid mode (verified-local answers first,
remote fallback through `FIREWORKS_BASE_URL` with models from
`ALLOWED_MODELS`) selected at build time via `LOCAL_LAYER` / `LOCAL_ONLY`
build args. in the zero-token build the container makes **no network calls
at all** (validated with `--network none`).

## docker image

```
ghcr.io/lippinc/mve-agent:latest
```

## how it runs

```
docker run --rm \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=... \
  -e ALLOWED_MODELS=... \
  ghcr.io/lippinc/mve-agent:latest
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
