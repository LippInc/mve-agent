# minimum viable effort — track 1 agent

solo entry for the AMD Developer Hackathon: ACT II, track 1 (general-purpose AI agent).

a small agent. it reads `/input/tasks.json`, answers each task through the
Fireworks AI API, and writes `/output/results.json`. built around gemma 4 on
fireworks.

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

all configuration comes from the environment. all inference goes through
`FIREWORKS_BASE_URL` using models from `ALLOWED_MODELS`. the container makes
no other network calls.

## submission items

demo video and slide deck accompany the lablab submission.
