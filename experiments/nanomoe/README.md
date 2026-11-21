# nanoMoE on Shakespeare, balance loss on and off

Run on Nov 20 to 21, 2026, on a Mac's CPU (macOS, Python 3.12, PyTorch 2.15), with the SDK at `e531c36` and the Python detectors at their default thresholds. One run of each arm, one seed. The two runs shared the machine with three lane sessions, so their timings are not to be read.

## What was run

[nanoMoE](https://github.com/wolfecameron/nanoMoE) at `661fbcf`, its character-level Shakespeare job, trained from scratch for 2,000 steps, twice from the same seed:

- **balance-on:** nanoMoE's own MoE settings (`config/train_nano_moe.py`): 8 experts, top-2, a router in every second block, the balance loss (`use_aux_loss`, weight 0.01), the router z-loss, Switch Transformer init, a full-precision router.
- **balance-off:** the same with `use_aux_loss=False`. The router z-loss stays on.

Both on a model small enough for a laptop CPU: 6 layers, 6 heads, width 192, batch 16, block 128 (8.87 million parameters, 2,048 tokens a step). `run.sh` has every flag.

The clone is left as it is. `run.sh` works in `work/`, a copy made with `git archive`, with `tensorquill.patch` applied to `train.py` (three additions: `tensorquill.init` before the model is compiled, `session.log` of the loss before the optimizer step, `session.close` at the end). The data is one download, of the file nanoMoE's own `data/shakespeare_char/prepare.py` names (`input.txt`, 1.1 MB, fetched with `curl`); `prepare.py` then wrote the training and validation files from it. `work/` and `out/` are not in the repository.

## What the routers' counts are

TensorQuill watches each router's inner linear layer (`router.w_g`), whose output is the logits, and counts the top-2 of them: the assignments the router asked for. nanoMoE then drops assignments over an expert's capacity. `check_router_hook.py` sets one against the other on one step at initialisation:

| Router | Asked for (TensorQuill) | Kept (nanoMoE's `used_capacity`) | Dropped at capacity |
|---|---|---|---|
| `transformer.h.0` | 4,096 | 3,811 | 285 |
| `transformer.h.2` | 4,096 | 3,775 | 321 |
| `transformer.h.4` | 4,096 | 3,112 | 984 |

On every expert, the kept count equals the asked count cut at the capacity of 640. So the recorded load is the router's choice, not what each expert received.

## What the two runs showed

| | balance-on | balance-off |
|---|---|---|
| Loss at step 2,000, train / validation (20 batches each) | 1.335 / 1.599 | 1.283 / 1.550 |
| `moe_imbalance` | step 0: router `h.4`, expert 1 at 36.2% | step 0, the same |
| `activation_drift` | step 79: `transformer.h.5`, `std` 3.04 times its early level | step 81: the same module, 3.04 times |
| `starved_experts` | checked, none | from step 130: router `h.2`, experts 2 and 5 at 1.17% and 1.19% of assignments over 50 steps (an even share is 12.5%) |
| Run state | unhealthy | unhealthy |

`compare` on the two (in `out/compare.json`):

- **Setup:** it names `use_aux_loss` as the difference, beside the run name and output folder.
- **Incidents:** two in both runs, and `starved_experts` in balance-off only, where balance-on checked the kind and found none.
- **Where the runs part:** no answer. The two do not agree in their first 30 steps, since the balance loss changes training from the first step.

So the one incident that separates the arms is the one the missing balance loss would be expected to cause, on a router this run names. The run without the balance loss also ends at a lower loss at 2,000 steps.

## The two incidents on the healthy arm

Both are on the arm with nanoMoE's own settings, so both are false alarms on a healthy run trained from scratch.

- **`moe_imbalance` at step 0** is the router at its random initialisation, before any update.
- **`activation_drift` at step 79.** The rule compares a module's recent `std` with its first 50 records. The ratio is 3.04 against a factor of 3 where it first crosses, and it goes on rising: `drift_measure.py` runs the rule's own arithmetic with the factor left open.

| Factor | balance-on: first step at or over it | balance-off |
|---|---|---|
| 2 | 59 | 59 |
| 3 (the default) | 79 | 81 |
| 4 | 102 | 104 |
| 5 | 145 | 156 |
| 8 | never | never |

The largest ratio of any module is 7.92 (balance-on, `transformer.h.0`, step 693) and 7.73 (balance-off), and every one of the six modules reaches at least 5.9. A model trained from scratch grows its activations several times over its first 50 steps as it learns. Of the factors tried, 2 to 5 all fire on both runs and 8 does not; a factor above the largest ratio, 7.92, would be quiet on these two runs, which says nothing of another healthy run. On the six recorded fine-tunes (v3 plan, Nov 20, 2026) the healthy ratios stay under 1.4 and the seeded ones are near 100. Nothing was changed: these are measurements only.

## Not shown

- One seed, one model size, 2,000 steps, CPU. No second seed says how much of the difference between the arms is the balance loss.
- nanoMoE's own configuration (12 layers, OpenWebText, two GPUs) was not run.
- Whether this run counts toward the plan's "real training job" result is the author's decision.

## Run again on the SDK after wave AD (Nov 21, 2026)

Both arms trained again with `run.sh` unchanged, from a worktree of `265311c` (wave AD merged): its `sdk/src` on `PYTHONPATH`, the same work copy (`train.py` checked equal to `train.py.orig` with `tensorquill.patch` applied, the same prepared data), the same seed and flags, 2,000 steps each on the Mac's CPU, beside three other lane sessions. The Rust detectors are a release build of that worktree's `server/` (`cargo build --offline --release`), named by `TENSORQUILL_SERVER_BIN`. The first run's records were left where they are and read only. Wave AD changed `activation_drift` and `moe_imbalance`, added a score beside each ratio, four measurement blocks and two keys a capture writes once. Until this run, the arms had only been recomputed from captures that the SDK at `e531c36` wrote.

```sh
cd server && cargo build --offline --release && cd ..
sh experiments/nanomoe/run.sh 2000
export TENSORQUILL_SERVER_BIN=$PWD/server/target/release/tensorquill-server PYTHONPATH=$PWD/sdk/src
.venv/bin/python experiments/nanomoe/fresh_compare.py --fresh experiments/nanomoe/out \
    --old <the first run's out/> --server "$TENSORQUILL_SERVER_BIN" --write experiments/nanomoe/out/ae3-evidence
.venv/bin/python -m tensorquill compare experiments/nanomoe/out/balance-on/<run>.rank0.jsonl \
    experiments/nanomoe/out/balance-off/<run>.rank0.jsonl -o experiments/nanomoe/out/compare.json
.venv/bin/python experiments/nanomoe/drift_measure.py balance-on=<capture> balance-off=<capture>
```

`fresh_compare.py` builds each capture's evidence with `build_evidence(..., detectors="python")` and `detectors="rust"`, compares the two key by key, and sets the fresh records against the first run's, step by step.

**Training is the same.** nanoMoE's own log gives the same loss at every logged step in both arms: step 2,000 train / validation 1.3353 / 1.5994 with the balance loss and 1.2832 / 1.5498 without it. Only the timings differ. Every one of the 2,001 records of each arm equals the first run's on every key the two share, exactly, apart from the run ID, the attempt and the timestamp.

**The two engines agree.** On both fresh captures, and on both first-run captures, the Python and Rust evidence are equal on `incidents`, `not_checked`, `kind_checks`, `run_state`, `moe_source` and the four measurement blocks, and on every other key but `detectors`. No difference was found.

**What the detectors give**, the same on the fresh captures as recomputed from the first run's at the merge:

| | balance-on | balance-off |
|---|---|---|
| Run state | healthy, no incident | unhealthy: two incidents |
| `moe_imbalance` | checked, none | step 7: router `h.2`, expert 3 overloaded, diversity 0.666 |
| `starved_experts` | checked, none | from step 130: router `h.2`, experts 2 (from step 130, 1.17% of assignments over 50 steps) and 5 (from step 187, 1.19%) |
| `activation_drift` | checked on 2,001 records (110 needed), none | the same |
| `not_checked` | none | none |

With the balance loss, the evidence of the fresh capture differs from the first run's only in `moe_source` and the run's identity (`run_id`, `run`, `run_end`). The same holds without it, in both engines.

**What the fresh records carry that the first run's could not.**

- `moe_source`, written once at step 0 for each of the three routers (`transformer.h.0`, `h.2`, `h.4` `.mlp.router.w_g`): `form` "scores", `point` "requested". Both are right for how nanoMoE is watched here. The hooked module is the router's inner linear layer, whose output is the logits, and TensorQuill takes their top-2 itself, so the counts are the assignments asked for, before nanoMoE drops those over an expert's capacity. `check_router_hook.py`, run again on this SDK, prints the same as before: 285, 321 and 984 assignments dropped at capacity on the three routers at initialisation. So "dispatched" would have been wrong.
- `layers_not_read`: none. All six blocks' outputs were read on every step.
- The measurement blocks:
  - `gradient_trend`, on `clip_grad_norm` over 2,001 points: growth 0.983 with the balance loss, 0.986 without; −0.0013 and −0.0010 per 100 steps.
  - `weight_jumps`: 40 weight checks, 42 parameters. The largest change of a norm between two checks is 1.210 with the balance loss (`transformer.wpe.weight`, steps 199 to 249) and 1.213 without it (the same tensor, 249 to 299). Next comes `h.0.attn.c_proj.weight` near 1.20 between 99 and 149 in both.
  - `attention_trends` and `update_trends` are absent in both arms and both engines. The patch runs no probe and does not ask for `updates=True`, so the records hold nothing for them to read: `update_spike` reads `not_configured` for the same reason.
- The score: no incident on either arm has a ratio, so no evidence holds a `score`.

**`compare`** on the two fresh arms:

- **Setup:** `use_aux_loss`, beside the run name and output folder.
- **What was checked:** no kind differs.
- **Incidents:** none in both runs. Two are in balance-off only, `moe_imbalance` from step 7 and `starved_experts` from step 130, on router `h.2`, and for each the arm with the balance loss checked the kind and found none.
- **Where the runs part:** no answer, as before. The two do not agree in their first 30 common steps (on 66 series the typical early difference is 0.23 times a series' own step-to-step movement, and 27 of them agree).

Against the first run's `compare.json`, the two incidents that sat in both arms (`moe_imbalance` at step 0 and `activation_drift` at 79 and 81) are gone. What separates the arms is now the whole of what either one reports.

`drift_measure.py` still runs on this SDK. Its "detector today" line calls the current rule and reports nothing on either arm. Its factor table encodes the rule before wave AD (a baseline of a module's first 50 records), and gives the same numbers as the table above, since the records are the same.

Not run: a second seed; another model size or nanoMoE's own configuration; a GPU or Linux. `--detectors rust` through the command line was not used (the Rust engine was called through `build_evidence`), and neither was a live server receiving the run. The records and evidence are not in the repository.
