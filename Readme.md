# LLDC — Part A (Thesis Code)

**LLMs as Data Compressors**: reproducible research code for **Predictive Masking (PM)** & **Vector-Quantization (VQ)** compression pipelines.

This README is the quick, human-friendly guide to get you from zero → running experiments.

---

## What’s inside (at a glance)

* `configs/` — Hydra configs for **compute**, **data**, **model**, **eval**, **experiment** (e.g., `experiment/e1a_wiki103.yaml`, `data/wikitext-2.yaml`, `model/gpt2.yaml`).
* `lldc/` — source code (compression, decompression, metrics, models, scripts, utils).
* `artifacts/` — where **checkpoints**, **logs**, **runs**, **rd\_curves** are written.
* `tests/` — unit tests for compression & metrics.
* `external/` — optional external tools (e.g., `cmix`, `deepzip`).
* `Makefile`, `pyproject.toml`, `uv.lock` — tooling & deps.

---

## 1) One-time setup

### (A) Make your `.env`

```bash
cp .env.example .env
```

Open `.env` and add your token (example):

```bash
HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
```

**How to actually export it (so your code sees it):**

* For one shell session:

  ```bash
  export $(grep -v '^#' .env | xargs)
  ```

  (Re-run this in new terminals.)

* Or, a safer POSIX approach:

  ```bash
  set -a
  . ./.env
  set +a
  ```

> Tip: If you prefer Python to auto-load `.env`, add `python-dotenv` and call:
>
> ```python
> from dotenv import load_dotenv; load_dotenv()
> ```
>
> Otherwise, stick to the `export` commands above.

### (B) Install dependencies with `uv`

> Requires Python **3.10+** and [`uv`](https://github.com/astral-sh/uv) installed.

Dev environment (project + linters + tests):

```bash
make dev
```

This creates `.venv/` and installs everything from `pyproject.toml` (including dev tools).

### (C) External dependency: **KenLM**

`uv` won’t install KenLM for you. Build/install KenLM separately and set:

```bash
# e.g., in ~/.bashrc or ~/.zshrc
export KENLM_BIN="$HOME/kenlm/build/bin"
```

---

## 2) Run experiments (the main workflow)

I use a single Make target that forwards **Hydra** overrides via `$(ARGS)`:

```make
run-exp:
	uv run python -m lldc.scripts.sweeps $(ARGS)
```

### Common runs

* **Experiment 1A (WikiText-103, main compression pipeline)**

  ```bash
  make run-exp ARGS="experiment=e1a_wiki103"
  ```

* **Experiment 2A (Pruning)**

  ```bash
  make run-exp ARGS="experiment=e2a_pruning"
  ```

* **Experiment 2B (Channel analysis)**

  ```bash
  make run-exp ARGS="experiment=e2b_channel"
  ```

* **Change dataset (example: WikiText-2)**

  ```bash
  make run-exp ARGS="experiment=e1a_wiki103 data=wikitext-2"
  ```

Hydra will merge from `configs/defaults.yaml` and your overrides. Artifacts land under `artifacts/` (checkpoints, logs, runs, RD curves).

---

## 3) Useful CLI entry points

These are registered in `pyproject.toml` and run inside the `uv` environment:

```bash
# Stage 1: specialize models
uv run specialise

# Stage 2: compress with PM or VQ
uv run compress_pm
uv run compress_vq

# (If present in your tree) Stage 3: reconstruct
uv run reconstruct_pm
uv run reconstruct_vq

# Evaluate suite
uv run evaluate

# Full sweeps (same as Make target under the hood)
uv run sweeps experiment=e1a_wiki103
```

> Prefer `make run-exp ARGS="..."` for day-to-day; it’s concise and reproducible.

---

## 4) Development & QA

* **Format + Lint**

  ```bash
  make fmt
  make lint
  ```
* **Run tests**

  ```bash
  make test          # quiet
  make test-verbose  # verbose
  make test-cov      # coverage
  ```
* **Everything (CI-ish)**

  ```bash
  make all   # sync + lint + test
  ```

---

## 5) Configuration quick tips

* **Compute**: `configs/compute/{local_dev.yaml,crescent_v100.yaml}`
* **Data**: `configs/data/{wikitext-2.yaml,wikitext-103.yaml,the-stack.yaml,mathematics.yaml}`
* **Model**: `configs/model/{gpt2.yaml,gpt2-medium.yaml,gpt2-large.yaml,bert-base-cased.yaml,roberta-base.yaml,distilroberta.yaml}`
* **Eval**: `configs/eval/{fidelity.yaml,efficiency.yaml,crumpled_paper.yaml}`

Override anything via CLI, e.g.:

```bash
make run-exp ARGS="experiment=e1a_wiki103 model=gpt2-medium data=wikitext-2 compute=local_dev"
```

---

## 6) Where things go

* **Checkpoints / logs / runs** → `artifacts/`
* **Rate–distortion curves** → `artifacts/rd_curves`
* **Figures & tables (reports)** → `reports/{figures,tables}`

---

## 7) Troubleshooting

* **My `.env` isn’t picked up**
  Run one of:

  ```bash
  export $(grep -v '^#' .env | xargs)
  # or
  set -a; . ./.env; set +a
  ```

  Then re-run your command in the same shell.

* **`bitsandbytes` install issues on non-Linux**
  It’s optional and gated to Linux in `pyproject.toml`; you can ignore on macOS/Windows.

* **KenLM not found**
  Ensure `KENLM_BIN` points to the directory that contains `lmplz`, `build_binary`, etc.
