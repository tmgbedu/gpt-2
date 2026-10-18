---
title: Training a 97.5M Python coding model from scratch for $1.40 in GPU time
published: false
cover_image: https://raw.githubusercontent.com/tmgbedu/gpt-2/main/docs/blog/images/cover.png
description: Building a 97.5M-parameter Python coding model, with about $1.40 of pre-training GPU time and fine-tuning on a MacBook.
tags: machinelearning, python, llm, deeplearning
---

*Bedram Tamang · October 2026 · code: [github.com/tmgbedu/gpt-2](https://github.com/tmgbedu/gpt-2)*

I wanted to really understand how language models are built, and the best way I know to understand something is to build it. So I built a small one myself, from nothing: my own tokenizer, my own model, my own training loop, fine-tuning and benchmark. I kept it to Python code only, and kept it small enough that I could afford to train it.

The main pre-training run used about $1.40 of rented GPU time.

> ### The result
> Keera solved **9 of 164 HumanEval problems (5.5%)** using greedy decoding, after solving **3 (1.8%)** before fine-tuning.
> Pre-training the **97.5M-parameter** model took **2 h 23 min** on a rented RTX 5090, costing approximately **$1.40** in training-time rental charges.
>
> These are preliminary results: I haven't checked the training data for overlap with HumanEval. The before/after runs also used different prompt formats and token limits, so the increase doesn't isolate the effect of fine-tuning.

| | |
|---|---|
| Parameters | 97.5M |
| Pre-training | 2B tokens of Python, 2 h 23 min on one RTX 5090 |
| Fine-tuning | 50k instruction examples, 4 h 40 min on a MacBook (M3 Pro) |
| HumanEval pass@1 | 1.8% → **5.5%** |

5.5% is not a useful coding assistant. But it's a real number on a real benchmark, from a model I understand line by line, and getting there taught me more than anything I've read.

---

## The architecture

![Keera architecture](https://raw.githubusercontent.com/tmgbedu/gpt-2/main/docs/blog/images/architecture.png)

I called the model Keera. It's shaped like GPT-2 small: a stack of 12 identical blocks that reads a sequence of tokens and guesses the next one. Every block does the same two things. First the tokens look back at each other (attention), then each token gets processed on its own (the MLP). After each step, the block adds its input back on top, so information can skip straight through.

```
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

That's really the whole model. Most of the 97.5M parameters sit inside those 12 blocks:

| Part | Parameters | Share |
|---|---:|---:|
| 12 blocks (attention 4·768² + MLP 8·768² each) | 84.9M | 87% |
| Embedding 16,384 × 768 (shared with the output layer) | 12.6M | 13% |
| **Total** | **97.5M** | |

---

## Why I made these choices

My rule was simple: copy what's known to work at this size, and only change something when there's a newer part that's cheaper or better.

**Decoder only.** Predicting the next token gives me one simple objective for both raw-code pre-training and instruction fine-tuning.

**GPT-2 small's size (12 layers, 768 wide).** It's a well-tested shape. It trains in a couple of hours on one consumer GPU, and it's small enough to fine-tune inside the 18 GB of memory on my Mac.

**My own tokenizer, with 16,384 tokens.** I trained it on Python, keeping leading indentation separate from the code that follows it. It averages 3.9 characters per token. GPT-2's roughly 50k-token vocabulary would have made the embedding alone 38.6M parameters, about 31% of a larger 123.5M-parameter model. A smaller vocabulary left more of my parameter budget for the transformer blocks.

**RoPE for positions.** It rotates queries and keys to encode relative positions without adding learned parameters. Extending the context later will still need testing and possibly more training.

**RMSNorm, before each sub-layer.** It normalises activations without subtracting their mean or adding a learned bias. This setup trained stably in my runs.

**GELU MLP, 4× wide.** Straight from GPT-2. SwiGLU might be a little better; that's for a later run.

**Tied embeddings.** The input and output layers share one matrix. That saves 12.6M parameters, and works fine at this size.

**1,024 tokens of context.** Most Python functions fit, and attention gets expensive fast, since its cost grows with the square of the length.

**2B tokens of training data.** The Chinchilla rule of thumb is about 20 tokens per parameter: 20 × 97.5M ≈ 1.95B.

**The training recipe.** AdamW at a learning rate of 6e-4, warmed up and then decayed on a cosine curve, with 524,288 tokens per step. With bf16 and `torch.compile` the RTX 5090 got through 234,000 tokens a second, about 77 times my MacBook's throughput under its training settings.

---

## Pre-training

For data I used [codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train), a deduplicated pile of Python files from GitHub. I tokenized 2B tokens of it into one 4 GB file, with an end-of-file marker between files. Training just reads random 1,024-token windows out of that file.

I started on my MacBook with a 100M-token test run. It worked, but it took 9 h 24 min. At that speed, the full 2B tokens would have taken about 190 hours, so for the real run I rented a GPU.

![Pre-training loss](https://raw.githubusercontent.com/tmgbedu/gpt-2/main/docs/blog/images/pretrain-loss.png)

The big run finished at a validation loss of **1.501**. The MacBook run had stopped at 2.125. Its curve drops faster at the start. Its batches were 16 times smaller, giving it 16 times more optimiser steps over the same number of tokens; the schedules also differed, so this isn't a controlled batch-size comparison.

What does 1.50 actually mean? Exponentiating the loss gives a **perplexity** of e^1.5 ≈ 4.5. You can think of that as an effective average branching factor, rather than a literal count of likely tokens at every step. A model assigning equal probability to all 16,384 tokens would have a loss of ln(16,384) ≈ 9.7.

Even so, the pretrained model writes code that looks right and often isn't:

```python
def is_even(n):
    """Return True if n is even."""
    if n > 0:
        return False        # right shape, wrong logic
```

### Renting the GPU

I used Vast.ai, a marketplace where people rent out GPUs by the hour. My RTX 5090 rental was $0.584 an hour. This is the workflow I used, with placeholders for the offer, instance and SSH details supplied by Vast.ai.

From my laptop, I created the instance and copied the tracked code:

```bash
vastai search offers 'gpu_name=RTX_5090 num_gpus=1' -o 'dph'      # cheapest first
vastai create instance <OFFER_ID> --image vastai/base-image:cuda-13.2.1-auto --disk 50 --ssh
git archive HEAD | ssh -p "$PORT" "root@$HOST" 'mkdir -p keera && tar -x -C keera'
ssh -p "$PORT" "root@$HOST"
```

Then, on the rented machine:

```bash
cd keera
bash scripts/gpu.sh setup
bash scripts/gpu.sh bench
bash scripts/gpu.sh data
bash scripts/gpu.sh train
bash scripts/gpu.sh status
```

`train` launches in the background. After the run finished, I downloaded the checkpoint from my laptop and verified it loaded before destroying the instance:

```bash
mkdir -p checkpoints/run-2b
scp -P "$PORT" "root@$HOST:keera/checkpoints/run-2b/checkpoint.pt" checkpoints/run-2b/checkpoint.pt
# After verifying the downloaded checkpoint:
vastai destroy instance <ID>
```

A few things I learned the hard way:

- **Benchmark before you train.** Thirty steps tell you the tokens per second, and from there the cost is simple: tokens ÷ tok/s ÷ 3600 × price per hour. For me that was 2e9 ÷ 236,000 ÷ 3600 × $0.584 ≈ **$1.40**.
- **Pay for On-Demand.** I tried two cheaper interruptible machines first, and both got outbid and paused before training even started.
- **Copy your code with `git archive`.** It excludes untracked files and caches. Check tracked files too: anything committed, including a secret, will be included.
- **Destroy the instance when you're done.** A stopped instance keeps billing for its disk.

---

## Post-training: teaching it to answer

A pretrained model only knows how to continue text. Ask it a question and it might write more questions. To turn it into something that answers, I fine-tuned it on [self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k), 50k Python instructions paired with answers. What I liked about this dataset is that every answer's code was actually run and passed its own tests.

Each example gets wrapped in a simple chat format. The model reads the whole thing, but it's only trained on the answer part, since there's no point teaching it to write the user's questions:

```
<|user|> Write a function that reverses a string. <|end|> <|assistant|>     ← read, not learned
Here's how … def reverse(s): return s[::-1] … <|end|>                     ← learned
```

After holding out validation data and dropping examples that exceeded the row length, I packed 49,549 training examples into 16,121 rows of 1,025 tokens, about 96% full. The trainer samples rows with replacement; I ran roughly three epochs' worth of sampled rows on my MacBook. It took 4 h 40 min.

![Fine-tuning loss](https://raw.githubusercontent.com/tmgbedu/gpt-2/main/docs/blog/images/sft-loss.png)

Validation loss barely improved during the third epoch's worth of training, staying around **0.81**, while training loss continued falling to about 0.47. That suggests diminishing returns and possible overfitting. Next time I'll compare earlier checkpoints before deciding how long to fine-tune; I haven't measured whether two epochs would give the same HumanEval score.

The model picked up the answer *format* almost immediately. By step 100 it was already writing a short intro, one code block, a sentence of explanation, and then stopping. Getting the logic right came much more slowly. I tried it on ten prompts of my own: 1 right, 3 nearly right, 6 wrong. This is the one it got right:

```python
# "Write a Python function read_lines(path) that returns a list of the lines in the file at path."
def read_lines(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    return lines
```

---

## How it compares

I used **HumanEval**: 164 Python problems with unit tests withheld from the model. For my runs, I generated one answer per problem using greedy decoding and report the fraction that passed as pass@1. Each answer runs in a separate process with a temporary working directory and a 10-second timeout. Before trusting my scorer, I checked it both ways: all 164 official solutions pass, and empty answers all fail.

The pretrained model continued the function prompt with up to 300 new tokens. The fine-tuned model received a chat instruction to complete the function, with up to 512 new tokens; I extracted its first code block for testing. The score rose from **3/164 to 9/164**, but the changed prompt and token budget mean this isn't a controlled measurement of fine-tuning alone.

The completed evaluation's console log records 9/164. The saved fine-tuned per-problem results file is currently incomplete and contains NUL bytes, so I need to restore it or rerun the evaluation before treating the result as fully reproducible. I also haven't checked the pre-training or fine-tuning data for overlap with HumanEval; the effect of any contamination is unknown.

**Published results for context; evaluation settings differ.** The historical baselines below include sampled pass@1 estimates, whereas Keera's results use greedy decoding. The chart and table aren't a ranking under a shared evaluation setup.

![HumanEval comparison](https://raw.githubusercontent.com/tmgbedu/gpt-2/main/docs/blog/images/humaneval.png)

| Model | Size | Trained on | HumanEval pass@1 |
|---|---:|---|---:|
| GPT-Neo 125M | 125M | general web text | 0.8% |
| **Keera, pretrained** | **97.5M** | **2B tokens of Python** | **1.8%** |
| codeparrot-small | 110M | same dataset | 3.8% |
| codeparrot | 1.5B | same dataset | 4.0% |
| GPT-Neo 1.3B | 1.3B | general web text | 4.8% |
| **Keera, fine-tuned (preliminary)** | **97.5M** | **+ 50k instructions** | **5.5%** |
| Codex 85M | 85M | GitHub code | 8.2% |
| GPT-J 6B | 6B | general web text | 11.6% |
| CodeGen-Mono 350M | 350M | many languages, then Python | 12.8% |
| Codex 12B | 12B | GitHub code | 28.8% |
| phi-1-small | 350M | "textbook-quality" data | 45% |

Sources: [CodeParrot's evaluation and training details](https://github.com/huggingface/transformers-research-projects/blob/main/codeparrot/README.md#evaluation), the [Codex paper](https://arxiv.org/abs/2107.03374) for Codex, GPT-Neo and GPT-J, the [CodeGen paper](https://arxiv.org/abs/2203.13474), and the [phi-1 paper](https://arxiv.org/abs/2306.11644). CodeParrot used the same pre-training dataset family, but trained for 25–30B tokens, compared with my 2B; the tokenizers also differ.

phi-1-small's reported 45% makes me want to try better data. Its pipeline used filtered web data, synthetic textbooks and coding exercises. It also has more parameters and a different training budget, so this comparison doesn't tell me how much of the gap comes from data alone.

### What it cost

| Step | Where | Time | Cost |
|---|---|---:|---:|
| Tokenizer + data | MacBook | minutes | – |
| 100M-token test run | MacBook (M3 Pro) | 9 h 24 min | – |
| 2B-token pre-training | RTX 5090, Vast.ai | 2 h 23 min | **$1.40** |
| Fine-tuning | MacBook | 4 h 40 min | – |
| HumanEval | MacBook | ~20 min | – |

The approximately **$1.40** covers the main pre-training run's duration at the quoted hourly rate. It isn't a total project bill: setup, data preparation, storage, transfers, earlier rental attempts, laptop electricity and hardware costs aren't included. A dash means I haven't assigned a cost to that step.

---

## What I learned

- **Loss and task accuracy measure different things.** Twenty times more training tokens took the loss from 2.13 to 1.50, but both models solved only 2 of my 9 informal test tasks, with different successes. After fine-tuning and changing the evaluation prompt, HumanEval rose from 3 solved problems to 9.
- **Data quality is worth testing.** phi-1-small's results make educational and synthetic Python data a promising next experiment.
- **Watch the validation curve during fine-tuning.** My third epoch's worth of training brought little improvement in validation loss. Earlier checkpoints deserve a benchmark run too.
- **On my Mac, a GPU memory cap helped.** Setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6` (with the low watermark at 0.5) stopped the swapping and GPU stalls I saw on my 18 GB M3 Pro. Other machines may need different limits.
- **Test on a tiny model first.** A tiny CPU-sized version of the model caught most of my bugs before I spent a cent.

## What's next

- **Better data.** Educational Python, the lesson from phi-1.
- **More reproducible numbers.** Restore the complete HumanEval output, check for contamination, compare checkpoints under matched evaluation settings, then add MBPP and pass@10.
- **Fill-in-the-middle training**, so it can autocomplete inside an editor. The special tokens are already in the tokenizer.
- **Going bigger.** 350M next, then perhaps 1B. At 20 training tokens per parameter, a 1B model would call for about 20B tokens. I'll benchmark it before estimating the rental cost.
- **Longer context.** Stretching RoPE from 1,024 to 4,096 tokens.
- **Preference tuning (DPO)** on top of fine-tuning.

---

## How this was built

I used AI throughout. **Claude Opus 5.5** (in Claude Code) wrote most of the code and tests, and **Codex** helped review the implementation and check the results. I made the decisions, ran the experiments, rented the GPU and checked the results. My run statistics are summarised in the repo's [training log](https://github.com/tmgbedu/gpt-2/blob/main/docs/training-log.md); the other models' scores come from the sources linked above.
