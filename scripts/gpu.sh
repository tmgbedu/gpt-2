#!/usr/bin/env bash
# Run Keera on a rented NVIDIA machine (Linux). From the repo root:
#
#   bash scripts/gpu.sh setup   # install uv + dependencies, run the test suite (incl. GPU tests)
#   bash scripts/gpu.sh bench   # ~30 steps on every GPU: prints tokens/sec to price a full run
#   bash scripts/gpu.sh data    # build the 2B-token dataset (data/pretrain-2b, ~4 GB)
#   bash scripts/gpu.sh train   # the 2B-token run, detached: survives a dropped SSH connection
#   bash scripts/gpu.sh status  # last log lines of the run
#
# Rerunning `train` resumes from the last checkpoint. Settings: NGPU (default: all GPUs),
# MICRO_BATCH (default 16; lower it if a GPU runs out of memory). The batch stays
# 524,288 tokens per step for any GPU count, so results don't depend on the machine.
set -euo pipefail

NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l || true)}
MICRO_BATCH=${MICRO_BATCH:-16}
GRAD_ACCUM=$((512 / MICRO_BATCH))  # 512 sequences x 1,024 tokens per optimizer step
RUN=checkpoints/run-2b
export PATH="$HOME/.local/bin:$PATH"

launch() {  # one training process per GPU
  uv run torchrun --standalone --nproc-per-node "$NGPU" -m trainer.train \
    --device cuda --precision bf16 --compile \
    --micro-batch "$MICRO_BATCH" --grad-accum "$GRAD_ACCUM" "$@"
}

case "${1:-}" in
  setup)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync
    nvidia-smi --query-gpu=name,memory.total --format=csv
    uv run python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPUs', torch.cuda.device_count())"
    uv run pytest -q
    ;;
  bench)
    [ -f data/smoke/train.bin ] || uv run python -m data.prepare --train-tokens 10000000 --out-dir data/smoke
    rm -rf checkpoints/bench
    launch --data-dir data/smoke --out-dir checkpoints/bench --max-steps 30 --warmup-steps 5 \
      --eval-interval 1000 --eval-batches 4 --checkpoint-interval 1000 --log-interval 5
    echo "Ignore the first steps (compiling). cost = tokens / tok_per_s / 3600 * price_per_hour"
    ;;
  data)
    uv run python -m data.prepare --train-tokens 2000000000 --val-tokens 5000000 \
      --out-dir data/pretrain-2b 2>&1 | tee data/pretrain-2b.log
    ;;
  train)
    mkdir -p "$RUN"
    # 3,815 steps x 524,288 tokens = 2.0B tokens (~20 per parameter).
    nohup bash "$0" _train >>"$RUN/console.log" 2>&1 &
    echo "started; follow with: tail -f $RUN/console.log"
    ;;
  _train)
    launch --data-dir data/pretrain-2b --out-dir "$RUN" --max-steps 3815 --warmup-steps 200 \
      --eval-interval 250 --eval-batches 40 --checkpoint-interval 250 --keep-every 500 --log-interval 10
    ;;
  status)
    tail -n 5 "$RUN/console.log"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
    ;;
  *)
    sed -n '2,13p' "$0"; exit 1
    ;;
esac
