# Cocktail JEPA

Self-supervised energy-based modeling of cocktail structure. A Joint-Embedding
Predictive Architecture (JEPA) learns the latent structure of well-formed
cocktails from ~6,600 unlabeled recipes; its prediction error becomes an energy
function used for coherence scoring, constrained generation, and transfer.

## The two-machine model

- **MacBook = cockpit.** Code, notebooks, data prep, debugging, small test
  runs. Apple Silicon `mps` handles shape-checks and single-batch overfitting.
- **Cloud GPU = engine.** Only real training (Stage 3) and ablations. Rent,
  run, pull results, release.

The git repo is the only thing that syncs between them. Code goes up via git;
checkpoints come back via cloud storage. Never edit files on the remote box.

## Setup (one time)

Requires [`uv`](https://docs.astral.sh/uv/) and `git`.

```bash
# 1. install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. from the project root, create the env and install everything
uv sync --extra dev

# 3. drop the corpus in place (git-ignored, synced separately)
#    corpus/recipes.jsonl  and  corpus/vocabulary.json

# 4. verify the station works
uv run python scripts/check.py
```

`check.py` should end with `STATION READY`.

## Layout

```
src/cocktail_jepa/
  config.py     device + paths, env-var overridable for cloud portability
  data/         Stage 1 -- dataset, encoding, splits, perturbation set
  model/        Stage 2 -- context encoder, predictor, EMA target, loss
  train/        Stage 3 -- training loop, collapse diagnostics
  energy/       Stage 4 -- energy function, evaluation, ablations
  generate/     Stage 5 -- energy-descent constrained sampler
  transfer/     Stage 6 -- SFT classification head
scripts/        thin CLI entrypoints (run headless on cloud)
notebooks/      marimo notebooks -- exploration and the final demo
corpus/         the dataset (git-ignored)
runs/           checkpoints + logs (git-ignored)
tests/          unit tests + collapse diagnostics
```

## Build stages

| Stage | What | Exit criterion |
|-------|------|----------------|
| 1 | Encoding + splits | DataLoader yields masked batches; perturbation set saved |
| 2 | JEPA model | Forward + backward pass runs without shape errors |
| 3 | Training | Trained model, collapse diagnostics stay healthy |
| 4 | Energy + eval | Trusted AUROC; ablations degrade when broken |
| 5 | Generation | Partial recipe -> completed low-energy recipe |
| 6 | SFT + demo | Frozen encoder beats from-scratch; demo runs |

Stages 1->2->3 are a hard chain. Stage 3 (collapse) is the gate. Stages 4, 5,
6 are independent of each other once a trained model exists.

## Running things

```bash
uv run python scripts/check.py        # smoke test
uv run marimo edit notebooks/01_explore_corpus.py   # exploration notebook
uv run pytest                         # tests
```

## Experiment tracking (Weights & Biases)

Logging goes through `cocktail_jepa.logging.get_logger`, which wraps W&B.
Every stage calls `logger.log({...})` without caring whether W&B is on:

- **W&B off** (default, Stage 1 plumbing): calls become console prints, no
  network dependency.
- **W&B on** (Stage 3 training): metrics stream to wandb.ai.

To turn it on, copy `.env.example` to `.env`, set `COCKTAIL_WANDB=1` and your
`WANDB_API_KEY` (from https://wandb.ai/authorize), then `set -a; source .env;
set +a` before running. If W&B is ever unreachable, the logger falls back to
console mode automatically -- a logging failure never kills a run.

## Cloud (Stage 3 only)

Training reads paths and the W&B flag from env vars so it is
provider-agnostic:

```bash
export COCKTAIL_DATA_DIR=/path/to/synced/corpus
export COCKTAIL_RUN_DIR=/path/to/persistent/storage
export COCKTAIL_WANDB=1
export WANDB_API_KEY=...
```

The cloud provider is deliberately not chosen yet -- the code stays portable
so it becomes a 20-minute decision at Stage 3, not an architecture commitment.
