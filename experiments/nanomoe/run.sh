#!/bin/sh
# nanoMoE on Shakespeare (characters) on CPU, twice from the same seed: once with its balance
# loss (`use_aux_loss`) and once without. Everything else is nanoMoE's own MoE configuration
# (config/train_nano_moe.py: 8 experts, top-2, a router in every second block, router z-loss,
# Switch Transformer init, full-precision router), on a model small enough for a laptop CPU.
#
# Run from the repository's root:   sh experiments/nanomoe/run.sh [steps]
#
# It needs experiments/nanomoe/work: a copy of nanoMoE at 661fbcf (`git archive HEAD | tar -x`),
# with data/shakespeare_char prepared by its own prepare.py and tensorquill.patch applied to
# train.py. The clone itself is left as it is. Captures go to experiments/nanomoe/out.
set -eu
ROOT=$(pwd)
STEPS=${1:-2000}
WORK=$ROOT/experiments/nanomoe/work
OUT=$ROOT/experiments/nanomoe/out
PY=$ROOT/.venv/bin/python
export PYTHONPATH=$ROOT/sdk/src
mkdir -p "$OUT"
{
  echo "sdk commit: $(git rev-parse HEAD)"
  echo "sdk/src differs from e531c36: $(git diff --quiet e531c36 -- sdk/src && echo no || echo yes)"
  echo "sdk/src uncommitted changes: $(git diff --quiet -- sdk/src && echo no || echo yes)"
  echo "nanoMoE commit: 661fbcf, with experiments/nanomoe/tensorquill.patch"
  echo "steps: $STEPS"
  echo "started: $(date)"
} > "$OUT/provenance.txt"
cd "$WORK"
for arm in balance-on balance-off; do
  aux=True; [ "$arm" = balance-off ] && aux=False
  rm -rf tensorquill-out
  /usr/bin/time -p "$PY" train.py config/train_shakespeare_char.py \
    --device=cpu --compile=False --dtype=float32 --wandb_log=False \
    --wandb_project=nano-moe --wandb_run_name="$arm" --out_dir="out-$arm" \
    --n_exp=8 --top_k=2 --stride=2 --use_aux_loss=$aux --aux_loss_weight=0.01 \
    --use_router_z_loss=True --router_z_loss_weight=0.001 \
    --use_switch_tfm_init=True --router_use_full_prec=True \
    --max_iters="$STEPS" --lr_decay_iters="$STEPS" --eval_interval=500 --eval_iters=20 --log_interval=100 \
    --batch_size=16 --block_size=128 --n_layer=6 --n_head=6 --n_embd=192 --dropout=0.0 \
    > "$OUT/$arm.log" 2>&1
  rm -rf "$OUT/$arm"
  mv tensorquill-out "$OUT/$arm"
done
cd "$ROOT"
echo "finished: $(date)" >> "$OUT/provenance.txt"
echo "sdk/src differs from e531c36 at the end: $(git diff --quiet e531c36 -- sdk/src && echo no || echo yes)" >> "$OUT/provenance.txt"
