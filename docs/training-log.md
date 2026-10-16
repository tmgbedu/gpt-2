# Keera training log

A running record of runs, results, samples and decisions. Newest entries at the bottom.

Baseline model (all runs so far): 97,536,768 parameters, `d_model` 768, 12 layers, 12 heads,
`d_ff` 3072 (GELU), RMSNorm, RoPE, tied embeddings, context 1024, 16,384-token byte-level BPE.
A uniform guess over the vocabulary gives loss ln(16,384) ≈ 9.70.

---

## 2026-10-15 — Smoke run (`checkpoints/smoke`)

- Data: `data/smoke` — 10.0M train tokens (4,737 files), 1.0M val tokens.
- 300 steps × 32,768 tokens = 9.8M tokens. fp32 on MPS, ~3,055 tok/s.
- Purpose: prove the pipeline end to end before a long run.

| Step | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| Val loss | 6.61 | 5.86 | 5.48 | 5.22 | 5.06 | 4.98 |

Samples began showing docstrings, `return`, `:rtype:`.

### Speed findings (M3 Pro, 18 GB)
- bf16 autocast works on MPS but gives no speedup; fp32 stays the default.
- GPU is saturated at microbatch 1–2; bigger microbatches don't help. Microbatch 4 fp32 runs out of memory.
- Without a memory cap, macOS swapped 13 GB and the GPU stalled. Fixed with
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6` / `LOW=0.5` (set in `trainer/train.py`).
- Eval + checkpoint overhead < 0.5% of run time.

---

## 2026-10-15 — 100M-token run (`checkpoints/run-100m`)

- Data: `data/pretrain-500m` — 500M train tokens (228,260 files, built in 186 s), 5.0M val tokens.
- 3,050 steps × 32,768 tokens ≈ 100M tokens (about 1 token per parameter).
- Warmup 150 steps, peak LR 6e-4 → cosine to 6e-5. Eval every 250 steps (40 batches).
  Milestone weights every 500 steps.
- Started 11:14, expected finish ~20:20. ~2,900–3,100 tok/s.

```
uv run python -m trainer.train --data-dir data/pretrain-500m --out-dir checkpoints/run-100m \
  --max-steps 3050 --warmup-steps 150 --eval-interval 250 --eval-batches 40 \
  --checkpoint-interval 250 --keep-every 500 --log-interval 10
```

| Step | Tokens | Val loss | e^loss (effective choices) |
|---|---|---|---|
| 250 | 8M | 5.08 | ~160 |
| 500 | 16M | 3.61 | ~37 |
| 750 | 25M | 2.93 | ~19 |
| 1000 | 33M | 2.68 | ~15 |
| 1250 | 41M | 2.54 | ~13 |
| 1500 | 49M | 2.43 | ~11 |
| 1750 | 57M | 2.34 | ~10 |
| 2000 | 66M | 2.27 | ~10 |
| 2250 | 74M | 2.22 | ~9 |
| 2500 | 82M | 2.18 | ~9 |
| 2750 | 90M | 2.15 | ~9 |
| 3000 | 98M | 2.13 | ~8 |
| **3050 (final)** | **100M** | **2.125** | ~8 |

### Samples

Step 500: valid-looking Python (`from .base import get_image`, `raise NotImplementedError(...)`),
ended the file cleanly.

Step 1000, `def fibonacci(n):` (temp 0.6) — Python-shaped but looping nonsense:
```python
def fibonacci(n):
    assert isinstance(x, Array) and isinstance(x, Array[0], Array[1] and isinstance(x, Array[3]) ...
```

Step 1250 — structure appears; first attempt at recursion (no base case):
```python
def fibonacci(n):
        return fibonacci(n)

def test_no_gc_gc_gc_gc_gc...
```

Step 1500 — clean methods with `self`, still repeats:
```python
    def get_pure(self, name):
        return self._get_pure(name)
```

Step 1750 (temp 0.4), three prompts:
```python
def read_file(path):
    """Read a text file and return its contents."""
    return os.path.join(path, "data", "data")      # right module, wrong logic

def is_even(n):
    """Return True if n is even."""
    if n > 0:
        return False                               # right shape (if/return bool), wrong logic
    ...

class Stack:
    def __init__(self):
        self._data = None                          # realistic, but no push/pop; then repeats
```

**Assessment:** has learned syntax, conventions and the *shape* of logic; not yet the logic
itself (e.g. `n % 2 == 0`). Repetition loops are common at this scale.

---

## Decisions

- **Next steps (in order):** HumanEval evaluation script → SFT on
  `bigcode/self-oss-instruct-sc2-exec-filter-50k` → evaluate again. Build and debug these on the
  100M-token model first (free), then reuse them on the bigger run.
- **2B-token run on a rented/free GPU** (~20 tokens per parameter). On this Mac it would take
  ~190 h (~$1–2.50 electricity); estimated ~3–4 h / ~$5–8 on an A100, budget $10–15.
  Needs a CUDA port (fp16 + GradScaler on T4, bf16 on A100) and a 2B-token dataset.
- **Start the 2B run fresh**, not from the 100M checkpoint: saves only ~5%, and a single clean
  schedule makes a fair baseline for architecture experiments.
- **Same model shape and tokenizer for the baseline.** The byte-level tokenizer can encode any
  text, and changing it would make losses incomparable.
- **Architecture experiments** happen between runs, on small models (~20–30M params,
  ~100–200M tokens) against the baseline under equal budgets; only the winner gets a big run.
- **Cleanup:** when the 100M run finishes, if results are good (no errors, final val loss lowest,
  samples at least as good), keep the last three milestones + `checkpoint.pt`; delete older ones.

---

## 2026-10-15 — 100M-token run finished

- Finished 20:38 (9 h 24 min), no errors. Final val loss **2.125** (every eval a new best).
- Nine-task test at step 3050: **2/9 correct** (`sum_list` → `sum(list(numbers), 0)`, `main()` guard),
  2 close (`count_words` splits, `load_json` forgets `open`). `find_max` and `count_words` now end
  the file instead of looping.
- Cleanup done: deleted milestones 500, 1000, 1500. Kept `step_002000/2500/3000.pt` + `checkpoint.pt` (2.2 GB).
- Next: CUDA + DDP port (test-first), then the 2B-token run on a rented 8× RTX 5090 (Vast.ai, Japan).

### Free-form samples, final model (temp 0.5, 120 tokens)

```python
def parse_args():                         # prompt: import argparse + def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the parser for a Python Python library')
    parser.add_argument(
        '--no-default',
        action='store_true',
        help='The default to use for Python Python Python library.')
    # ...then repeats the same argument
```
Real API used correctly (`add_argument`, `action='store_true'`); content is nonsense and repeats.
Same pattern for a `User` class (`__str__`, `__eq__` that recurses forever), a Flask route
(`@app.route` + `jsonify`, then "auth/auth/auth…"), and an `os` script (`try: os.makedirs … except OSError`).
