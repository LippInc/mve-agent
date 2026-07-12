# minimum viable effort — track 1 agent

solo entry for the AMD Developer Hackathon: ACT II, track 1 (general-purpose AI agent).

a small agent. it reads `/input/tasks.json`, answers each task, and writes
`/output/results.json`. the submitted build is the **pure local
configuration**: every task is answered by bundled llama.cpp models running
inside the container — **zero network calls, zero API tokens** (validated
with `--network none`). prompts are routed by category, drafted locally, and
pass deterministic verification before shipping.

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
  verified verbatim-in-source; summaries are format-verified against the
  task's stated constraints (word limits, sentence counts, bullet counts,
  per-bullet word caps — deterministically enforced) and regenerated once
  when they drop the passage's key figures; logic puzzles are brute-forced
  by generated constraint-enumeration programs; two-sided reviews are
  labeled mixed only after two independent confirmations. unverifiable
  answers fall back to the best available sample — never an empty answer
  (a deterministic extractive summary is the summarization floor).
- **offline factual grounding**: factual questions retrieve passages from a
  bundled Simple English Wikipedia full-text index (sqlite FTS5/BM25, built
  at image-build time). the model drafts, the draft's entities steer the
  lookup, and a grounded second pass answers from the retrieved text —
  reading comprehension instead of small-model recall. still zero network:
  the encyclopedia ships inside the image.
- **escalation policy (hybrid builds only)**: the same codebase also builds
  a verified-local hybrid (`HYBRID_POLICY=h3`) where factual recall and
  failed verifications escalate as terse calls through `FIREWORKS_BASE_URL`
  to a model from `ALLOWED_MODELS`. in the submitted zero-token build nothing
  escalates — every category resolves locally, with best-available fallbacks
  instead of remote calls.
- **budget discipline**: a global watchdog fair-shares the runtime budget so
  the batch always completes inside the contest limits, and `results.json`
  is written atomically after every task (always valid JSON, exit 0).

configurations are selected at build time via `LOCAL_LAYER` / `LOCAL_ONLY` /
`HYBRID_POLICY` / `LOCAL_FINAL` build args. the submitted image is the pure
zero-token local build; earlier submissions from this repo were the hybrid.

## docker image

```
ghcr.io/lippinc/mve-agent:final7
```

## how it runs

```
docker run --rm \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=... \
  -e ALLOWED_MODELS=... \
  ghcr.io/lippinc/mve-agent:final7
```

all configuration comes from the environment. no keys or answers are baked
into the image.

## bundled models & licenses

- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
  (Q4_K_M GGUF) — Apache License 2.0
- [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
  (Q4_K_M GGUF) — Apache License 2.0
- [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` — MIT License
- Simple English Wikipedia article text (snapshot 2023-11-01, via the
  [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
  dataset), indexed offline — CC-BY-SA 3.0

## submission items

demo video and slide deck accompany the lablab submission.
