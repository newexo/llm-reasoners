# Running the Blocksworld demo (CoT vs. ToT vs. RAP) in Docker

This documents how to run `demo.ipynb` — the tutorial notebook comparing Chain-of-Thought, Tree-of-Thought,
and RAP (MCTS) on the Blocksworld planning task — inside the GPU-enabled Docker image built from this
branch (`reuben/docker-gpu-experiment`).

## Prerequisites

- A Linux host with an NVIDIA GPU, the driver installed, and the NVIDIA Container Toolkit configured for
  Docker (check with `docker info | grep -i nvidia`; you should see `nvidia` listed under `Runtimes`).
  12GB+ VRAM is enough for any of the model options below at 4-bit quantization.
- Submodules initialized — the build fails fast with a clear error if not:
  ```bash
  git submodule update --init
  ```
- (Optional, only if using the gated Llama-3.1-8B option) A Hugging Face account with the
  [Llama-3.1-8B license accepted](https://huggingface.co/meta-llama/Llama-3.1-8B), and an access token.
- A `.env` file in the repo root with your HF token, if you're using it (Docker Compose loads this
  automatically — no other config needed):
  ```
  HF_TOKEN=hf_...
  ```
  This isn't required for the two ungated model options below.

## Build

```bash
docker compose build
```

Rebuilds only when `pyproject.toml`/`poetry.lock`/`poetry.toml` or the repo source changes — the CUDA base
image layer and the VAL-from-source build layer are cached independently and don't need to redo.

## Run

```bash
docker compose up -d
docker compose logs | grep token=
```

Open the printed `http://127.0.0.1:8888/lab?token=...` URL in a browser. The container binds Jupyter to
`127.0.0.1` only (not reachable from your local network). The repo directory is bind-mounted into the
container (`.:/workspace`), so notebook edits and outputs persist to disk — only dependency changes need a
rebuild, not source/notebook edits.

## Choosing a model

Open `demo.ipynb`. The model-setup section near the top has several alternative cells — **run exactly
one**:

| Cell | Model | Gated? | Notes |
|---|---|---|---|
| ExLlama | ~~`TheBloke/Llama-2-70B-GPTQ`~~ | — | **Removed from this fork.** Left commented out for reference only; do not run. |
| HFModel | `meta-llama/Llama-3.1-8B` | Yes — needs `HF_TOKEN` + license acceptance | `quantized='nf4'`, matches the notebook's original saved outputs most closely (same model family as the 70B reference run) |
| HFModel | `Qwen/Qwen2.5-7B` | No | Apache-2.0, generally stronger reasoning benchmarks than Llama-3.1-8B |
| HFModel | `mistralai/Mistral-7B-v0.3` | No | Apache-2.0, well-established Llama drop-in since 2023 |
| SGLang | `meta-llama/Llama-3.1-8B` | Yes | Needs a separately-running SGLang server; not covered by this Docker setup |

All three `HFModel` options use `bitsandbytes` 4-bit (`nf4`) quantization — each fits comfortably within
12GB VRAM (~4-5GB weights) with headroom for the MCTS search's KV cache.

## Running the notebook

Run cells top to bottom after the model cell:
1. Loads one Blocksworld example + a 4-shot ICL prompt pool.
2. **Chain-of-Thought** — raw prompt, no search. Expect an invalid plan.
3. **Tree-of-Thought** — `BeamSearch`, still tracks only action history as state (no real world model). Also
   typically produces an invalid plan.
4. **RAP** — `MCTS`, tracks the actual block configuration as state via an LLM-simulated `step()`. This is
   the notebook's payoff: where CoT/ToT fail, RAP is expected to find a valid plan.
5. **Visualization** — uploads the search tree and prints a link to `llm-reasoners.net/visualizer/...`.
6. **Full evaluator run** (last cell) — runs the whole `BWEvaluator` pipeline over the dataset, including
   real VAL plan validation (`Plan valid` / `Plan failed to execute` in the output).

### Expected outcome, and a caveat

The notebook's saved output cells demonstrate the full narrative (CoT fails → ToT fails → RAP succeeds)
using the original `Llama-2-70B-GPTQ` run. At 7-8B parameters (any of the options above), the framework
mechanics all work identically — but per this repo's own analysis, RAP's advantage over CoT/ToT scales
with base model capability, so a smaller model may not reproduce the *exact* "small model still gets it
right via RAP" contrast on the first example. If example `evaluator.full_dataset[1]` doesn't show a clear
split, try a few other indices, or increase `MCTS(n_iters=...)` in the RAP cell.

## Verifying VAL independently

If you want to sanity-check the plan validator without running the full notebook:

```bash
docker compose exec llm-reasoners bash -c '$VAL/validate <domain.pddl> <problem.pddl> <plan.pddl>'
```

A valid plan's output contains the line `Plan valid`; an invalid one contains `Plan failed to execute`.

## Teardown

```bash
docker compose down          # stop and remove the container
docker compose down -v       # also delete the downloaded model weights (hf-cache volume)
```

## Troubleshooting

- **Build fails with "LLMs-Planning submodule not initialized"**: run `git submodule update --init` on the
  host before `docker compose build` — Docker's build context is a plain file copy, it doesn't fetch
  submodules itself.
- **`403`/access errors downloading Llama-3.1-8B**: your `HF_TOKEN` doesn't have the model's license
  accepted yet — visit https://huggingface.co/meta-llama/Llama-3.1-8B and accept it, or use one of the
  ungated alternatives instead.
- **`nvidia-smi` not found / GPU not visible inside the container**: confirm `docker info` lists `nvidia`
  as an available runtime on the host; if not, the NVIDIA Container Toolkit needs to be installed/configured
  there first.
