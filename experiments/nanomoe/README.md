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

The largest ratio of any module is 7.92 (balance-on, `transformer.h.0`, step 693) and 7.73 (balance-off), and every one of the six modules reaches at least 5.9. A model trained from scratch grows its activations several times over its first 50 steps as it learns, so no factor under 8 is quiet on these two runs. On the six recorded fine-tunes (v3 plan, Nov 20, 2026) the healthy ratios stay under 1.4 and the seeded ones are near 100. Nothing was changed: these are measurements only.

## Not shown

- One seed, one model size, 2,000 steps, CPU. No second seed says how much of the difference between the arms is the balance loss.
- nanoMoE's own configuration (12 layers, OpenWebText, two GPUs) was not run.
- Whether this run counts toward the plan's "real training job" result is the author's decision.
