---
title: Training a GPT-2-sized (97M) model from scratch
published: false
description: A 97.5M-parameter Python coding model, pretrained for $1.40 on a rented GPU and fine-tuned on a MacBook.
tags: machinelearning, python, llm, deeplearning
---

*Bedram Tamang · October 2026 · code: [github.com/tmgbedu/gpt-2](https://github.com/tmgbedu/gpt-2)*

I built a small Python coding model end to end to learn how it works. I wrote my own tokenizer, model, training loop, fine-tuning and benchmark. Pre-training ran on one rented GPU for 2 h 23 min. Fine-tuning ran on my MacBook.

> ### 🏁 The result
> **5.5% on HumanEval (pass@1)** with **97.5M parameters** and **$1.40** of GPU rent.
> That is above GPT-Neo 1.3B (4.8%) and codeparrot 1.5B (4.0%), a model **15× larger** trained on the same data.

| | |
|---|---|
| Parameters | 97.5M |
| Pre-training | 2B tokens of Python, 2 h 23 min on one RTX 5090 |
| Fine-tuning | 50k instruction examples, 4 h 40 min on a MacBook (M3 Pro) |
| HumanEval pass@1 | 1.8% → **5.5%** |

---

## The architecture

![Keera architecture](https://raw.githubusercontent.com/tmgbedu/gpt-2/main/architecture.png)

It has the same shape as GPT-2 small: a stack of 12 identical blocks that reads tokens and predicts the next one. Each block does two things, each with a residual connection:

```
x = x + Attention(RMSNorm(x))    # tokens look at earlier tokens
x = x + MLP(RMSNorm(x))          # each token is processed on its own
```

| Part | Parameters | Share |
|---|---:|---:|
| 12 blocks (attention 4·768² + MLP 8·768² each) | 84.9M | 87% |
| Embedding 16,384 × 768 (shared with the output layer) | 12.6M | 13% |
| **Total** | **97.5M** | |

---

## Why these choices

Most choices copy what worked for GPT-2 and GPT-3 small. Where I changed something, I picked a newer part that is cheaper or works better at the same size.

| Choice | Picked | Why |
|---|---|---|
| Shape | Decoder only | One task, predicting the next token, covers both stages. Raw code doesn't come in input/output pairs, so an encoder-decoder has nothing to pair it with. |
| Size | 12 layers × 768 | GPT-2 small is a known-good shape. It trains in hours on one consumer GPU and fine-tunes inside 18 GB on a Mac. |
| Tokenizer | Own BPE, 16,384 tokens | Trained on Python, with indentation as its own token: 3.9 characters per token. GPT-2's 50k vocabulary would make the embedding 38.6M parameters, 40% of the model. |
| Positions | RoPE | Rotates queries and keys by their position, so attention sees relative distance. No learned table, and it can be stretched later for longer context. |
| Normalisation | RMSNorm, before each sub-layer | Cheaper than LayerNorm (no mean, no bias) with the same quality. Norm-first keeps training stable without tuning. |
| MLP | GELU, 4× wide | Kept from GPT-2. SwiGLU is a possible upgrade. |
| Embeddings | Tied | Input and output share one matrix, saving 12.6M parameters. |
| Context | 1,024 tokens | Most Python functions fit. Attention cost grows with the square of the length. |
| Data size | 2B tokens | Chinchilla's rule of about 20 tokens per parameter: 20 × 97.5M ≈ 1.95B. |
| Optimiser | AdamW, LR 6e-4, warmup + cosine | The GPT-3 small recipe. Weight decay 0.1, β₂ 0.95, gradients clipped at 1.0. |
| Batch | 524,288 tokens per step | GPT-3 small used about 0.5M. Large batches give smooth gradients and keep the GPU busy. |
| Speed | bf16 + `torch.compile` | 234,000 tokens/s on an RTX 5090, 77× the MacBook. |
| Fine-tuning | Learn answers only, LR 1e-4 | The model only learns from answer tokens. A learning rate 6× lower than pre-training keeps what it already knows. |

---

## Pre-training

**Data.** 2B tokens from [codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train), a deduplicated set of Python files from GitHub. Every file becomes its tokens plus an end marker, and all of them go into one 4 GB file. Training reads random 1,024-token windows from it.

**Two runs.** First, a 100M-token test run on the MacBook: 9 h 24 min. At that speed the full 2B tokens would have taken about 190 hours, so I rented a GPU for the real run.

![Pre-training loss](https://raw.githubusercontent.com/tmgbedu/gpt-2/main/docs/blog/images/pretrain-loss.png)

The 2B run ends at a validation loss of **1.501**, and the MacBook run stops at 2.125. The MacBook curve falls faster per token because its batch is 16× smaller, so it takes 16× more steps for the same data.

A loss of 1.50 means the model is unsure among about e^1.5 ≈ 4.5 likely next tokens. A random guess over 16,384 tokens would score ln(16,384) ≈ 9.7.

The pretrained model writes Python that looks real but is often wrong:

```python
def is_even(n):
    """Return True if n is even."""
    if n > 0:
        return False        # right shape, wrong logic
```

### Renting the GPU

I rented one RTX 5090 on Vast.ai, a marketplace of GPUs rented by the hour, at $0.584/h.

```bash
vastai search offers 'gpu_name=RTX_5090 num_gpus=1' -o 'dph'      # cheapest first
vastai create instance <OFFER_ID> --image vastai/base-image:cuda-13.2.1-auto --disk 50 --ssh
git archive HEAD | ssh -p $PORT root@$HOST 'mkdir -p keera && tar -x -C keera'   # only tracked files
bash scripts/gpu.sh setup && bash scripts/gpu.sh bench && bash scripts/gpu.sh train
vastai destroy instance <ID>                                    # a stopped instance still bills
```

- **Benchmark before you train.** 30 steps give tokens/s, and the cost is tokens ÷ tok/s ÷ 3600 × $/h. For this run: 2e9 ÷ 236,000 ÷ 3600 × $0.584 ≈ **$1.40**.
- **Pick On-Demand.** Two cheaper interruptible instances were outbid and paused before training started.
- **Copy with `git archive`**, so no keys or caches leave your machine.

---

## Post-training: teaching it to answer

A pretrained model only continues text. To make it answer requests, I fine-tuned it on [self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k): 50k Python instructions with answers whose code was run and passed its own tests.

Each example is wrapped in a chat template. The model reads the whole thing, but its loss only counts the answer:

```
<|user|> Write a function that reverses a string. <|end|> <|assistant|>     ← read, not learned
Here's how … def reverse(s): return s[::-1] … <|end|>                     ← learned
```

The 49,549 examples are packed into 16,121 rows of 1,025 tokens, 96% full.

![Fine-tuning loss](https://raw.githubusercontent.com/tmgbedu/gpt-2/main/docs/blog/images/sft-loss.png)

Three epochs on the MacBook GPU took 4 h 40 min. Validation loss goes flat at **0.81** early in epoch 3, while train loss keeps falling to 0.49. In that epoch the model was memorising, so two epochs would have been enough.

The model learned the answer *format* fast: by step 100 it gave an intro, one code block, a short explanation, and then stopped. The logic improved less. On ten of my own prompts it got 1 right, 3 nearly right and 6 wrong. Here's the one it got right:

```python
# "Write a Python function read_lines(path) that returns a list of the lines in the file at path."
def read_lines(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    return lines
```

---

## Results, compared

**HumanEval** is 164 Python problems with hidden unit tests. Pass@1 is the share it solves on the first try. Every answer runs in its own sandboxed process. As a check, all 164 official solutions pass my scorer and empty answers all fail.

![HumanEval comparison](https://raw.githubusercontent.com/tmgbedu/gpt-2/main/docs/blog/images/humaneval.png)

| Model | Size | Trained on | HumanEval pass@1 |
|---|---:|---|---:|
| GPT-Neo 125M | 125M | general web text | 0.8% |
| **Keera, pretrained** | **97.5M** | **2B tokens of Python** | **1.8%** |
| codeparrot-small | 110M | same dataset | 3.8% |
| codeparrot | 1.5B | same dataset | 4.0% |
| GPT-Neo 1.3B | 1.3B | general web text | 4.8% |
| **🏁 Keera, fine-tuned** | **97.5M** | **+ 50k instructions** | **5.5%** |
| Codex 85M | 85M | GitHub code | 8.2% |
| GPT-J 6B | 6B | general web text | 11.6% |
| CodeGen-Mono 350M | 350M | many languages, then Python | 12.8% |
| Codex 12B | 12B | GitHub code | 28.8% |
| phi-1-small | 350M | "textbook-quality" data | 45% |

> **Fine-tuning tripled the score, from 3 to 9 solved problems.** That puts a 97.5M model above GPT-Neo 1.3B, and above codeparrot 1.5B, which is 15× larger and trained on the same data.

phi-1-small shows how far there is to go: at 350M parameters with a similar architecture, it scores 45% because of carefully chosen data.

*Other models' numbers are from their papers or model cards. I haven't checked the fine-tuning data for overlap with HumanEval yet, so 5.5% may be a little high.*

### What it cost

| Step | Where | Time | Cost |
|---|---|---:|---:|
| Tokenizer + data | MacBook | minutes | – |
| 100M-token test run | MacBook (M3 Pro) | 9 h 24 min | – |
| 2B-token pre-training | RTX 5090, Vast.ai | 2 h 23 min | **$1.40** |
| Fine-tuning | MacBook | 4 h 40 min | – |
| HumanEval | MacBook | ~20 min | – |

---

## Lessons

- **Lower loss doesn't mean more skill.** Validation loss went from 2.13 to 1.50 with 20× more data, and the model still solved 2 of my 9 test tasks. A few hours of fine-tuning tripled HumanEval.
- **Data matters more than size.** codeparrot 1.5B on this data scores below phi-1-small at 350M.
- **Two epochs of fine-tuning are enough.** The third only memorised.
- **On a Mac, cap GPU memory** (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6`). Otherwise macOS swaps and the GPU stalls.
- **Test on a tiny model first.** A tiny CPU model caught most bugs before I spent any money.

## Next

- **Better data:** educational Python in the style of phi-1.
- **Honest numbers:** a contamination check, plus MBPP and pass@10.
- **Fill-in-the-middle** training for editor autocomplete. The special tokens already exist.
- **Scale up:** 350M, then 1B (about 20B tokens, roughly $150–250 of GPU time), with grouped-query attention.
- **Longer context:** stretch RoPE from 1,024 to 4,096 tokens.
- **Preference tuning** (DPO) after fine-tuning.

---

## How this was built

I used AI throughout. **Claude Opus 5.5** (in Claude Code) wrote most of the code and tests, and **Codex** reviewed and validated it. I made the decisions, ran the experiments, rented the GPU and checked the results. Every number here comes from my training logs, summarised in the repo's [training log](https://github.com/tmgbedu/gpt-2/blob/main/docs/training-log.md).
