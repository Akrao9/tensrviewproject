# TrainCheck's 20 silent errors against TensorQuill, on CPU

Lane T7, Nov 18, 2026: the TrainCheck part of row M1 in the v3 plan's build list ("the
detectors against ... TrainCheck's 20 reproduced bugs, to state how many each catches").
TrainCheck (OSDI '25, arXiv 2506.14813, Table 3 on p. 11) reproduces 20 real silent
training errors, B1 to B20. Each one has a row in [RESULTS.md](RESULTS.md), which
`run_all.py` writes from `results.json`. Nothing in TensorQuill was changed. A gap found here
is written down as a gap, not fixed.

## First: only 2 of the 20 reproduction scripts are on this machine

The TrainCheck clone (`../TrainCheck` from the repository root, at `7a62cbb`) holds
something for three of the 20, and the third is not a script:

- **B1's script** is in the tree, `docs/assets/code/84911.py` (the 5-minute tutorial's).
- **B8's script** is in the history only: `example_pipelines/115607_bug.py`, deleted in `471f57e`.
- **B3** has only a config in the history (`example_pipelines/DS-1801/config.json`, deleted in `7de7430`), which points at an external Megatron-DeepSpeed checkout.

The other reproductions are in a separate repository, `OrderLab/TrainCheck-Evaluation-Workloads`
(the AE guide, `docs/ae-eval-s5.1-silent-issue-detection.md`). It was not fetched: that is a
download, and the user's call. The history was searched through every path any commit
touched, `bugs.md` (deleted in `dfd8d84`) included. Neither was the issues' own text read:
every re-creation below is built from Table 3's one-line synopsis, TrainCheck's `bugs.md` and
blog post, and what is in PyTorch 2.15.

So each row falls in one of these levels, which the table keeps apart and which must not be
added together:

| Level | Meaning | Rows |
|---|---|---|
| shipped | TrainCheck's own script, as kept in the clone | B8 |
| swapped | TrainCheck's script with only the model and the data replaced (downloads not installed) | B1 |
| re-created | written here from the synopsis | B4, B9, B10, B14, B19 |
| simulated | the mechanism, or only the effect, put in by hand because the bug's own trigger cannot run here | B3 (mechanism), B6 (the old worker seeding), B2, B17 and B18 (one shared run of the effect only, as TrainCheck itself reuses the PT-104336 trace for these three) |
| not run | the bug needs a package, a download, an old library version or a GPU that is not here | B5, B7, B11, B12, B13, B15, B16, B20 |

Every script measures the failure with plain PyTorch, apart from TensorQuill: weights that
did not change, replicas that differ, repeated augmentations, a wrong sigma. So "nothing
fired" can be told apart from "the bug did not happen". Each also runs a healthy variant to
count false incidents. The healthy variants are written here: TrainCheck's `mnist.py` is its
reference pipeline, not a fixed copy of B1.

## How it was run

- **Python and packages.** The project's `.venv`: Python 3.12, PyTorch 2.15.0, CPU. The `tensorquill` package is this worktree's `sdk/src`, at `3f93636`.
- **TensorQuill's start.** `tensorquill.init(model, capture="full", updates=True, run_id=..., sink=JsonlSink(out/...))`, plus `replica_group=` (and `replica_params=` under tensor parallelism) on the two-process runs.
- **Data and checkpoints.** `session.watch_loader` on every loader, `session.log({"train/loss": ...})` before each optimizer step, and `session.checkpoint` on B14's save and load.
- **Evidence.** Built with `tensorquill.build_evidence` for each rank's file, plus the cross-rank `detect_duplicate_batches` on the two-process runs.
- **Multi-process runs.** Two CPU processes with gloo, meeting through a file rather than a port.
- **Settings.** The defaults first; then any change of `weights=`, `replica_every=` or run length, named after what it changes.
- **Not run.** Nothing was timed: six other sessions shared the machine. Each configuration ran once, with fixed seeds.

```bash
cd experiments/traincheck-bugs
../../.venv/bin/python run_all.py          # every configuration, buggy and healthy
../../.venv/bin/python run_all.py --report # rewrite RESULTS.md only
```

`out/` (records, evidence, logs) is not committed; `results.json` holds every run's summary.

## What the results show

Counts are of rows, and the levels are kept apart. B2, B17 and B18 share one run.

**Caught: a kind fired on the buggy run and not on the healthy one, and both runs completed with valid observations.**

- **By level.** None with TrainCheck's script as it is in the clone. B1 with its script, model and data swapped. B9 and B10 re-created. B3 simulated. Plus the simulated hardware effect.
- **B9 and B10 are one measurement.** They are the same training with the gradient sync lost two ways: the replicas end the same distance apart (0.1107).

| Bug | Level | Kind @ step | Note |
|---|---|---|---|
| B10 X-DDP | re-created | `replica_drift` @ 49 (both ranks); @ 0 with `replica_every=1` (its first update) | The first cross-rank check. TrainCheck reports within one iteration. |
| B9 PT-104336 | re-created | `replica_drift` @ 49 (both ranks) | The trigger is `overwrite_module_params_on_conversion`: on 2.15 it makes `.to()` put new `Parameter`s in place, so DDP's hooks stay on the old ones, as the synopsis describes. The issue's own trigger was not read. |
| B3 DS-1801 | simulated mechanism | `replica_drift` @ 49 (both ranks); @ 1 with `replica_every=1` | LayerNorm weights end 2e-3 apart; the two runs' final losses are 0.6791 and 0.6794. |
| B1 PT-84911 | swapped | `stalled_weights` @ 19 with `weights=5`; @ 199 over 200 steps at the default | Caught in 2 of its 4 configurations, under two conditions. First, enough weight checks: at TrainCheck's own 22 steps with the defaults none falls in the run (`kind_checks` says `not_configured`, 0 of 3 records). Second, `init` called before the freezing loops: called after them, it records nothing (see the gaps). The incident names the 12 BatchNorm parameters that get gradients and says no optimizer holds them: the root cause. |
| B2, B17, B18 | simulated effect only | `replica_drift` @ 99 | Rank 1 loses a tenth of one weight at step 75; caught at the next check. Not a run of any of the three bugs. |

**Not caught, though the failure showed (4)**

| Bug | Level | What happened | Why TensorQuill does not see it |
|---|---|---|---|
| B4 SO-60335387 | re-created on TrainCheck's `mnist.py` | The gradient norm climbs from 5.2 to 38 over 102 steps (360 times one batch's at the end); the model stays at chance (10% against 100% healthy). | `grad_spike` compares a step with the recent median, which climbs with it. `update_spike` does not read Adadelta (`not_checked`). The one incident, `activation_drift`, fires on the healthy run too. |
| B6 SO-67180955 | simulated (pre-1.9 seeding) | 75% of the augmentations repeat across the four workers. | Batches are hashed from their sample indices, which stay distinct; `duplicate_batches` compares ranks, not workers. The review counted B6 as within reach; in this simulation, which is this lane's reading of the synopsis, it is not caught. |
| B14 TF-29903 | re-created | Two LSTM tensors are missing from the checkpoint; after the resume they are fresh (changed by up to 0.41); the final loss is 1.76 against 1.35 healthy. | `resume_incomplete` checks which components a checkpoint holds, not whether the model's tensors are all in it. The manifest records the model's tensor count and a hash of its keys, and no rule compares them with the model. With `--checkpoint minimal` (no RNG state, no data position) `resume_incomplete` fires on both variants, as an observation. |
| B19 PT-51800 | re-created | The first evaluation divides each weight by -0.038 and -0.020 times its true sigma; validation loss 1517 against 3.07. | It happens in an evaluation, outside the training steps TensorQuill records. |

**Ran, but tested nothing (1).** B8 (TrainCheck's script) runs on PyTorch 2.15 once `save_config=False` is dropped, but the bug does not happen: the compiled loop's Adam step counter equals the eager one. The script builds its model inside the compiled function, so TensorQuill is started there and Dynamo traces `tensorquill.init`. Until Nov 19, 2026 that raised an `AssertionError` on `mark_static_address` into the script; `init` now steps out of the trace itself (lane U1), and the script records all six steps with or without `init` and `session.log` wrapped in `torch.compiler.disable`.

**Not run (8).** B5, B7, B11, B12, B13, B15, B16 and B20 need a package, a download, an old library version or a GPU that is not here; each row of `RESULTS.md` says which. Judged from the synopses and TensorQuill's code, not tested, seven of them are also out of its reach as built, whatever the setup: lengths (B5), dtypes (B7), batch shapes after a processor (B11), evaluation (B12), model loading (B13), the planned step count (B15; TrainCheck misses it too) and which parameters are meant to be frozen (B20). B16's hang would need the server's watchdog or NCCL Flight Recorder dumps.

**False incidents.** `activation_drift` needs 60 records, so it judges 12 of the 17 healthy configurations: B1's three 22-step runs and B8's two 6-step runs are too short.
- **Anomalies.** `activation_drift` fired on 11 of the 12, plus one `activation_spike` (B4).
  - The 12 are only seven distinct trainings: B9, B10, B10 every-step and the hardware run are one healthy training, and B3's two and B14's two are one each.
  - So it fired on 6 of 7, B1's 200-step run the only clean one.
  - It also fired on 10 of the 12 buggy runs long enough to judge.
- **Observations.** One `resume_incomplete`, on B14's healthy run with the minimal checkpoint, which saves no RNG state or data position.
- **Invariants.** None fired on a healthy run.

Because `activation_drift` fires on almost every small run that learns, it never counts as a catch here.

**Against the review's reading (D7: up to seven within reach, B1 and B3 doubtful).** By level, never summed:
- Caught: B1 (swapped; with enough weight checks and `init` before the freezing), B9 and B10 (re-created; one measurement), B3 (simulated).
- Not caught: B6 (simulated).
- Not testable: B8 on 2.15.
- Not run: B16.

**Gaps found in TensorQuill (not fixed here)**
- **Where `init` is called decides B1.** `init` tracks the parameters that need gradients at that moment.
  - Called just before the training loop, after the freezing loops, it treats the optimizer, which holds only frozen parameters, as another job's. It records no step at all (an empty file, run state `unknown`), and its one warning names only DeepSpeed ZeRO and Megatron as causes. **Fixed since (Nov 19, 2026, wave X, lane X1):** an optimizer that holds only frozen parameters of the model now ends steps, and so does a model whose every parameter is frozen at `init`, so this placement records steps and B1's case is caught as `stalled_weights` (`sdk/tests/test_lane_x1_capture.py`). The four configurations of this page were not run again after the fix, so the table above still shows the result measured before it.
  - Called right after `model.to(device)`, which in `84911.py` comes before the freezing, it catches the bug.
  - The guide ("after the model is on its device and before `torch.compile`") allows both.
- **`init` raises when Dynamo traces it** (B8).
- **`resume_incomplete` does not compare a checkpoint's model tensors with the model** (B14).
  - **Fixed since (Nov 21, 2026, wave AC):** at `session.checkpoint(state, event="load")` the checkpoint's tensor names are compared with the model's (`CHECKPOINT_TENSOR_CONTRACT.md`). `b14_tf29903.py` was run again in its two default configurations on the merged tree: the buggy run raises `resume_incomplete` at step 100 with the finding `missing_tensors` naming `lstm.bias_hh_l0` and `lstm.weight_hh_l0` (4 of the model's 6 tensors matched), from the Python detectors and from the server alike, and the healthy run raises none (6 of 6 matched). The two `--checkpoint minimal` configurations were not run again. `RESULTS.md`, `results.json` and the B14 row of the table above were not rewritten, so they still show the result measured before the fix.
- **The batch hash is over indices**, so identical content from different indices is not seen (B6).
- **`grad_spike` does not see a steady climb** of the gradient norm (B4).
- **`replica_drift` at its default interval** sees a desync up to 49 steps after it starts. TrainCheck's paper reports one iteration. With `replica_every=1` it is step 0 (B10) or 1 (B3), at a cost not measured here.

## What they do not show

- **The bugs themselves.** Only B1 and B8 are TrainCheck's code. B8's bug does not happen on PyTorch 2.15. The re-created and simulated runs are this lane's reading of a one-line synopsis, so a re-creation may miss what made the real bug hard. B9's trigger is one that takes the same path on 2.15, not necessarily the issue's. B3 is the mechanism without DeepSpeed's code.
- **A count of "N of 20 caught".** The levels above measure different things, and eight rows were not run.
- **Real hardware and models.** Nothing ran on a GPU, NCCL, Linux, a real model or real data: small MLPs, CNNs and an LSTM on seeded synthetic data.
- **Timing.** Detection steps depend on the check intervals (`weights=50`, `replica_every=50` by default), which the configurations vary.
- **A head-to-head with TrainCheck.** TrainCheck is not installed in the venv and was not run, so its numbers (detects 18 of 20, each within one iteration) are the paper's, on its own reproductions and versions.
- **Repeat runs.** One seed per configuration, so nothing says how often a kind would fire on other seeds.
