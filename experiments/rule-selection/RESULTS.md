# Rule selection for wave AD: what was measured

Nov 21, 2026, on the branch `contracts/wave-ad` at the SDK of `eb5a71b`. Measurement only: nothing in the SDK was changed. Every input is on `allowlist.json`, which was committed before the first run (`a873c23`). Gate 1's six held-out cases were not read.

To run it again: `python inventory.py`, `python measure_drift.py`, `python measure_imbalance.py`, from this folder with the project's `.venv`. The two toy captures and the TrainCheck re-creations are generated, not stored (`generated/` and `experiments/traincheck-bugs/out/` are ignored).

## The inputs

| Input | Records | For `activation_drift` | For `moe_imbalance` |
|---|---|---|---|
| Granite 3.0 1B MoE fine-tune, three capture modes with layers, one routers-only (GPU) | 650 each | Healthy | Seeded: one router collapsed at step 450 (`install_collapse` in `moe_dogfood.py`) |
| SmolLM2 fine-tune, and two `Trainer` runs (GPU) | 400 each | Seeded: an activation fault at step 250 (253 in one) | No routers |
| nanoMoE from scratch, balance loss on | 2,001 | Healthy | Healthy |
| nanoMoE from scratch, balance loss off | 2,001 | Healthy for this rule | The imbalance the missing loss causes |
| TrainCheck re-creations, 46 record sets from 9 scripts, healthy and buggy | 60 to 200 | Healthy runs are healthy. No buggy run plants an activation fault | No routers |
| Seeded-failure toy with nothing planted (CPU) | 350 | Healthy | No routers |
| `toy_moe.py` (CPU) | 300 | No layers | Seeded: the router collapses at step 200 |
| Gate 1's visible practice fixture | 4 | No layers | Seeded: an overload at step 1,850, in a capture of four sampled steps |
| Two hand-written record sets of the parity suite | 300, 30 | One planted drift, written to fit today's rule | Planted overloads and dead experts |

28 healthy record sets hold layers. 22 of them are long enough for every candidate below; the other six have no layers past 60 records or fewer than 110 records.

## `activation_drift`

Today's rule compares a module's last 10 records with its first 50, and fires at 3 times. A model trained from scratch grows its activations for as long as it trains, so that ratio has no ceiling.

| Candidate | What it compares | Healthy sets that fire (of 28) | Largest healthy ratio | The three GPU faults |
|---|---|---|---|---|
| Today, factor 3 | Last 10 against the first 50 | 18, from 12 runs | 7.92 (nanoMoE, step 693) | All fire at step 254 |
| Factor 5 | The same | 12 | 7.92 | All at 254 |
| Factor 8 | The same | 0 | 7.92 | All at 254 |
| Factor 10 | The same | 0 | 7.92 | All at 254 |
| Trailing, factor 2 | Last 10 against the 50 before them | 11 | 4.33 (B04, step 59) | All at 254 |
| Trailing, factor 3 | The same | 1 (B04, at step 59) | 4.33 | All at 254 |
| Trailing, factor 5 | The same | 0 | 4.33 | All at 254 |
| **Trailing, start left out, factor 3** | Last 10 against the 50 before them, and a module's first 50 records are never the baseline | **0** | **1.68** (B14, step 143) | **All at 254** |
| Trailing, start left out, factor 2 | The same | 0 | 1.68 | All at 254 |
| Settled baseline (three tolerances) | A baseline frozen once the module stops changing | 2 to 7 | 5.76 | All at 254 |

What the table says:

- **Raising the factor works only on these runs.** Factors 8 and 10 are quiet because the largest healthy ratio here happens to be 7.92. That ratio grows with the length of training, so a longer healthy run would cross any fixed factor.
- **A trailing baseline is bounded by how fast training changes, not by how long it runs.** Its false alarm, and every one of its largest healthy ratios, is at step 59: the first step it can judge, when its baseline is the run's first 50 records.
- **Leaving the start out removes that.** With a module's first 50 records never used as a baseline, the largest healthy ratio in 22 record sets is 1.68, against a factor of 3, and against about 100 on the seeded faults.
- **The settled baseline is fragile.** A module that pauses and then grows again fires it (B14 healthy at step 133).

Each seeded fault, by the recommended candidate:

| Fault | Result |
|---|---|
| SmolLM2, block 8, step 250 | Fires at step 254, as today. Ratio about 100 |
| `Trainer`, step 250 | Fires at 254, as today |
| `Trainer`, step 253 | Fires at 254, as today |
| The hand-written planted drift | Fires at step 185, as today. Its ratio and its count of affected steps change |
| TrainCheck's buggy runs, nanoMoE without the balance loss | No activation fault is planted: the signal was never in the capture |

What the recommended candidate gives up:

- **A slow drift.** A level that changes by less than 3 times within 60 records never fires, however far it goes in the end.
- **The first 110 records of a module** are not judged for drift (60 today). `activation_spike` still judges from the 11th record.
- **After a shift, the incident lasts about 30 records** and then the new level is the baseline. The incident is still dated at its first step.

## `moe_imbalance`

nanoMoE with its own settings raises it at step 0: one expert of router `h.4` has 36.2% of the assignments against a line of 35%, at random initialisation. It is flagged on steps 0 and 1 and never again in 2,001 steps.

| Candidate | nanoMoE, balance on (healthy) | nanoMoE, balance off | Granite collapse at 450 | `toy_moe` collapse at 200 | Practice fixture, overload at 1,850 |
|---|---|---|---|---|---|
| Today | Fires at step 0 | Step 0, on `h.4`: the same random start | 450 | 200 | 1,850 |
| Not judged at step 0 | **Fires at step 1** | Step 1 | 450 | 200 | 1,850 |
| Two flagged samples in a row | Fires (steps 0 and 1) | Fires | 450, seen at 451 | 200, seen at 201 | 1,850, seen at 1,900 |
| Three in a row | Quiet | Step 38, on `h.2` | 450, seen at 452 | 200, seen at 202 | **Missed**: only two samples are left |
| Five in a row | Quiet | Step 59 | 450, seen at 454 | 200, seen at 204 | **Missed** |
| **Starting state, up to 10 samples** | **Quiet** | **Step 7, on `h.2`** | **450** | **200** | **1,850** |
| Starting state, up to 5 or 50 samples | The same | The same | The same | The same | The same |

"Starting state": a router that is flagged from its first sample at step 0 is in the state it was initialised in. That leading run of flagged samples is not reported while it is shorter than the limit, and is reported, dated at its first sample, once it reaches the limit.

What the table says:

- **"Not before the first update" only moves the incident to step 1.** One update does not undo a random start.
- **A sustain rule misses sparse captures.** The practice fixture has four sampled steps, and its overload is on the last two.
- **The starting-state rule separates the two nanoMoE arms**, which today's rule does not: quiet with the balance loss, and step 7 on router `h.2` without it, where one expert then stays overloaded for 1,726 of 2,001 steps.
- **Every seeded collapse is reported at the same step and router as today.**
- The limit makes no difference on these inputs between 5 and 50. 10 is the history the top-k rule already waits for (`OVERLOAD_MIN_HISTORY`).

What it gives up: a router that is collapsed from its first step is reported 10 samples later than today, dated at its first sample.

## Not shown

- One seed of each from-scratch run. No large model trained from scratch, and no run longer than 2,001 steps.
- No healthy run here has a learning-rate restart or a resume, either of which could move activations within 60 records.
- The three GPU faults are all step changes of about 100 times. No recorded fault is small or gradual.
- The spike, diversity, starvation and stall thresholds were not re-measured in this pass.
