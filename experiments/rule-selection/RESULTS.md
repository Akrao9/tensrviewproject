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

How far this goes:

- **The recommended candidate was shaped on these runs.** The plain trailing rule was tried first; leaving a module's start out of the baseline was added after it fired on B04's healthy run at step 59. So this table is a fit, not a test.
- **B04's healthy run is not judged by it.** That run has 102 records and the candidate needs 110. It is quiet there for want of history.
- **The 22 judged record sets are about ten distinct workloads:** Granite (three capture modes of one job), the two nanoMoE arms (one model, one seed), the seeded-failure toy, B1, B3, B6, B14, B19, and one model shared by B9, B10 and the hardware simulation.

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
- The spike, diversity, starvation and stall thresholds were not re-measured in this pass, though the plan said they would be read on the same runs.
- Of the sibling repositories, only LLM-Training-Debugger was read for how it scores against a baseline.
- `trainer-check-block` is listed with its fault at step 253; its activation spike shows at step 250, as in the other `Trainer` run. The table's "fires at 254" holds either way.

## A robust local trend, for Q1 and B4 (the author's request, Nov 21, 2026)

`measure_trend.py`. The candidate is `robust_local_trend` of the sibling repository LLM-Training-Debugger (`src/training_debugger/detectors/base.py`), written again here from its description: a line through the median of a window's first third and the median of its last third, each placed at the median of its own step numbers. It uses the actual step numbers, so probes 1,000 steps apart and records one step apart are both read rightly. Under six points it gives no trend. Nothing is judged and no threshold is set.

Recorded per series: `growth` (last-third median over first-third median), the slope as a share of the first anchor per 100 steps, and the longest stretch of steps over which the trailing window's trend stayed up.

**B4, the total gradient norm.**

| Run | First third to last third | Growth | Longest rise |
|---|---|---|---|
| Granite, healthy | 15.19 to 15.40 | 1.01 | 221 steps |
| nanoMoE, both arms | 0.77 to 0.75 | 0.98 | 235 and 252 steps |
| Seeded-failure toy, healthy | 2.31 to 2.29 | 0.99 | 151 steps |
| B1 long, healthy / buggy | 0.32 to 0.58 / 0.69 to 3.56 | 1.82 / 5.20 | 180 / 194 steps |
| B9, healthy / buggy | 1.54 to 1.69 / 2.02 to 2.46 | 1.09 / 1.21 | 190 / 192 steps |
| B14, healthy / buggy | 0.33 to 0.77 / 0.33 to 0.46 | 2.32 / 1.39 | 174 / 174 steps |
| **B4, healthy / buggy** | 1.76 to 1.81 / **38.45 to 38.45** | 1.03 / **1.00** | 81 / 16 steps |
| The three GPU activation faults | 2.0 to 7.8, 10.1 and 22.3 | 3.45, 5.14, 11.35 | 127 to 130 steps |

- **It does not show B4's re-creation.** Without `zero_grad()` the gradient norm goes 5.2, 32.1, 38.4 in three steps and then stays at 38.45 for a hundred. The climb is over before any window holds six points, and the plateau is flat. What marks that run is its level, 360 times the healthy run's, and one run alone has no healthy level to compare with.
- **It does show a sustained rise where there is one** (B1's buggy long run: 5.2 times, rising for 194 of 200 steps).
- **A healthy run can rise more than a buggy one** (B14: 2.32 healthy, 1.39 buggy; B1 healthy 1.82). So the number is a description of the run and not evidence of a fault, which is why it carries no threshold.

**Q1, an attention module's largest logit.**

- **No capture on the allowlist can show it.** The three captures with attention probes have four probes each (steps 99, 199, 299, 399), under the six a trend needs.
- On the review's own series, written by hand (a logit from 10 rising by a fixed share at each probe, 1,000 steps apart): at 25% a probe, where `attention_logit_growth` says nothing, the trend reads growth 3.81 after 8 probes and 108 after 31; at 5% a probe, 1.34 and 2.79; flat reads 1.00. So it measures what the moving baseline cannot. That is arithmetic on a made-up series, not a finding about a real run.

## S6, the median-and-MAD score, on these captures

`measure_score.py`. Each record's total gradient norm, and each module's `abs_max` and `std`, scored against the 50 records before it: `(value - median) / (1.4826 * MAD)`, beside the ratio `value / median`. Also the same on residuals from the window's robust local trend.

| Series | Largest healthy ratio | Largest healthy score | At the seeded faults: ratio | Score |
|---|---|---|---|---|
| Total gradient norm | 4.91 (Granite, step 136) | 75.9 (the same step) | 4.3, 5.6, 25.8 | 31, 54, 360 |
| A module's `abs_max` | 7.81 (B4 healthy, step 31) | 11.9 | about 780 | about 4,100 |
| A module's `std` | 7.37 (B4 healthy, step 31) | 24.3 (nanoMoE, step 11) | about 100 | 2,500 to 2,800 |

- **For activations the score separates by a wide margin**: at most 24 on 28 healthy record sets, thousands on the faults. The ratio separates them too.
- **For the total gradient norm neither number separates.** A healthy Granite step scores 76, above two of the three faults (31 and 54), and its ratio of 4.91 is above one fault's 4.3. The sibling's default line of 6 would fire on every healthy run here: their largest scores run from 4 to 76.
- **Detrending changes little on these runs.** Largest healthy scores of 75.5, 8.8 and 13.7 against 75.9, 11.9 and 24.3. Its one large effect is at the very start of a from-scratch run (nanoMoE's `std` at step 11: 24.3 to 9.6).
- **A zero MAD did not happen in any real capture**: 0 of about 144,000 scored windows. It happens in the hand-written record set (290 of 580 windows of one statistic), where values repeat exactly.
- **Asking for 10 or 20 earlier values** gives the same largest healthy scores, except that 20 leaves out nanoMoE's step 11 (24.3 becomes 19.6).

So the score is worth recording beside the ratio, and these captures give no line to draw on it.

## Validation: one fresh run, not used to choose either rule

`validate.py`, Nov 21, 2026. nanoMoE from scratch in another shape than the allowlist's: 4 layers, width 128, 4 experts at top-1 with a router in every block, dropout 0.1, batch 12, block 96, 900 steps (901 records). Run once, after both rules were proposed, against the rules as the contract's generator states them.

**`activation_drift`.** 4 modules have layer statistics. All 4 reached 110 records (901 each) and all 4 were judged, from step 109, on 792 records each. None fired: the largest ratio was 1.44 (`transformer.h.0`, step 122). Today's rule fires on this run at step 114 and reaches 5.28. So the proposed rule was exercised here, not skipped for want of history.

**`moe_imbalance`.** This run does not test the starting-state rule, because its routing collapsed for real. All four routers are flagged on most of their 901 samples:

| Router | Samples flagged | With a dead expert | Median share of the busiest expert | Flagged from step 0 for |
|---|---|---|---|---|
| `h.0` | 791 | 2 | 57% | 0 samples |
| `h.1` | 848 | 231 | 81% | 0 samples |
| `h.2` | 812 | 330 | 91% | 25 samples |
| `h.3` | 889 | 714 | 97% | 0 samples |

With four experts at top-1 an even share is 25%. One expert takes most of every router's tokens for most of the run, though nanoMoE's balance loss was on at its own weight of 0.01. Today's rule reports step 0 on `h.2`. The proposed rule reports step 5 on `h.3`, the first incident that can be reported; `h.2`, collapsed since step 0, becomes reportable at its tenth sample. Both are right to report. What the run cannot say is whether the rule stays quiet on a healthy router of another shape: for that it would need a run whose routing did not collapse.

**The trend.** The gradient norm's growth is 1.14 over the run.
