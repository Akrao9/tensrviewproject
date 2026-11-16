# TensorQuill — Focused Plan

Nov 16, 2026 · @Ak B

TensorQuill does one job: it wakes a team's coding agent with layer-level evidence when a multi-GPU PyTorch run goes wrong. Work that does not serve that job is frozen until the five results under Prove are in. The v3 plan of Nov 13, 2026 stays as the decision log.

## Product

- **Who it is for:** teams training mixture-of-experts and other large models on several GPUs, where the loss curve looks healthy while the run fails.
- **The pitch:** add one line to a PyTorch script. When the run fails, your agent is told the step, the layer and the kind of failure, with the numbers behind it.
- **What it captures:** per-layer activations, attention, MoE routing, gradients and weights, and weight drift across GPUs, without breaking `torch.compile`.
- **What it costs:** router-only capture, the MoE default, measured +0.70% per step on an RTX PRO 6000. Full capture is opt-in at about +3% to +4%.
- **What it hands over:** `evidence.json`, a list of incidents with step, layer or parameter, kind and the scalars logged at that step.
- **What it leaves to others:** scalars and run metadata stay in MLflow or W&B, the team's own agent does the diagnosis, and stock Rerun 0.43.2 is the viewer.

## What sets it apart

Per-layer debuggers now exist, so "first" is not the claim. The claim is multi-GPU failures, measured cost under `torch.compile`, and evidence any agent can read.

| Alternative | What it gives | What it lacks or leaves open |
| --- | --- | --- |
| MLflow and W&B, with their MCP servers | Scalar curves and whatever the user logged, open to an agent's queries | No capture of layer internals and no failure detectors |
| CoreWeave ARIA | An agent inside W&B that analyses runs and drafts changes | Works on what W&B logged; tied to W&B and CoreWeave |
| Trainy Pluto | MCP tools that read logged panels, with a summary of each curve | Displays what the user logs; no capture SDK or detectors found |
| [NeuralDBG](https://github.com/LambdaSection/NeuralDBG) 1.5.0, MIT | Gradient and activation hooks on every leaf module, with ranked hypotheses and a JSON export. An MCP server with three tools: diagnose, benchmark and explain | Multi-GPU: it only warns when the model is wrapped in DDP or DataParallel. Compile: its hooks are wrapped in `torch._dynamo.disable`, and its tests allow a 20× slowdown. Its diagnose tool runs model code passed to it, so it does not read a live run. The dead-expert check is an empty stub. Detection rates come from bugs injected into small synthetic models. The README says the engine is proprietary, but an MIT engine is bundled |
| [MoEWatch](https://forum.fossunited.org/t/moewatch-v0-1-open-source-diagnostics-for-moe-routing-failure/9532) 0.1.0, Apache 2.0 | Dead and cold experts, routing entropy and load imbalance on Hugging Face MoE models, with WARN and ERROR alerts and JSON output | Counts only each token's top-1 expert, so a top-k router is under-counted. Copies the router output to the CPU on sampled steps. Nothing for `torch.compile` or more than one process. No trigger or webhook |
| [LLM-Training-Debugger](https://github.com/YanKaFei/LLM-Training-Debugger) 0.1.7, Apache 2.0 | Incidents in Parquet from loss, gradients, per-module activations and throughput, each with its evidence and what could not be concluded. `compare` finds the first step where two runs diverge, and `scan` finds the outlier in a batch of runs | Its limits list says multi-rank, `torch.compile`, AMP and FSDP are untested, and every number comes from 4–6 layer models on a laptop. It reports run-level recall of 0.500 and 7.7% false incidents on healthy runs |

Four things none of the alternatives states, as read on Nov 16, 2026:

1. **Multi-GPU failures.** Weight drift between replicas, silent or stuck ranks, and resume mismatches. None of the three debuggers detects them: NeuralDBG only warns under DDP, and LLM-Training-Debugger records rank but says multi-rank is untested.
2. **Measured cost under `torch.compile`.** Zero graph breaks and recompiles are checked in CI on CPU and were measured on single-GPU runs. A 1% overhead gate applies. None of the three debuggers publishes a measured compile cost.
3. **Attention and MoE routing together.** The probe reads attention alongside FlashAttention, which never materialises the attention matrix.
4. **Layer-level evidence for any agent.** A JSON file today, and MCP tools after build piece 4, for whichever agent, tracker and cloud the team has. The trackers' MCP servers carry only what was logged, and NeuralDBG's does not read a live run.

Points 1 and 2 become public claims only after results 1 and 3 under Prove.

The tracker, ARIA and Pluto rows are carried over from the v3 competitor check of Nov 16, 2026. The three debugger rows were checked the same day against clones of their repositories: code and docs read, nothing run.

## Freeze

Six lines of work stop now. Each returns only when its condition is met.

| Work | Decision | Returns when |
| --- | --- | --- |
| New detector kinds | Stop at the 32 built | A real job shows a failure none of them catches |
| Rune viewer app | Postponed; Rerun stays the viewer | Outside users ask for views Rerun cannot give |
| Second copy of each detector | One Rust implementation for the 28 server kinds, called from Python; each Python duplicate is deleted as it is replaced. The four Python-only kinds stay as they are | Not applicable: this is a simplification, not a pause |
| Real-AWS and GCS runs, and the `s3://` reader | Postponed; the store code is built and stays | A user asks for a cloud store |
| Hosted tier, SSO, pricing | Postponed | Three outside teams run the SDK |
| RL, robotics, diffusion and multimodal packs | Not in this plan | A user brings a workload |

This page is the short front page of the plan. Decisions and measurements are still logged in the v3 plan, which stays the record.

## Build: the agent loop

Five pieces turn the evidence files into continuous monitoring by an agent. They are built in this order, and each has one done test.

1. **Say what was checked.** Each evidence file gives every kind one of three values: checked, not configured or unavailable. Only unavailable degrades the run, and the summary carries the count, such as "9 of 32 kinds checked".
   - Today `not_checked` covers four checks: attention, resume learning rate, TrainCheck and the Flight Recorder.
   - Done when a router-only capture lists the weight and activation checks as not configured and still reads `healthy`, and a failed probe reads `degraded`.
2. **Compact evidence.** The agent's file holds the incidents and a short window of steps around each, with no per-step samples.
   - Done when a 10,000-step MoE run gives a file under 1 MB (proposed line; the full file is 86 MB today).
3. **A trigger.** `--watch` takes `--on-incident <command>` and a webhook URL, and fires once per new incident. It does not fire on a change of run state, because incidents never close. Invariants fire by default, and anomalies are opt-in until thresholds are tuned.
   - Done when a seeded failure starts the agent with no human step.
4. **A read-only MCP server** over the same files, with four tools: list incidents, get one incident, get a series for one layer or parameter, and get the all-ranks status.
   - Done when a fresh agent session names the step, layer and kind for each fixture through it alone.
5. **Explain only.** The agent writes its diagnosis next to the incident. Pausing, stopping or changing a run needs a person's approval.
   - Done when the agent's instructions and tools offer no write action on the run.

TensorQuill itself still only reports. Nothing in the SDK or server blocks or stops training.

## Detector changes

No new kinds are added. Labelling the 32 kinds is done, and one change to replica drift follows the two-GPU run.

1. **Label every kind** as invariant, anomaly or observation. Done Nov 16, 2026: all 32 are labelled in Python and Rust with a parity test, as 12 invariants, 11 anomalies and 9 observations.
   - Invariants are rules that must always hold, such as replica drift, repeated steps, duplicate batches, non-finite values and collective desync.
   - Anomalies compare a signal with its own recent history, such as gradient spikes, activation spikes and drift, and memory growth.
   - Observations record something that may be intended, such as the resume checks, GPU throttling and memory imbalance.
   - Invariants and anomalies both mark a run `unhealthy`, as the code does, and `degraded` means a monitor is unavailable. False alarms are handled in the trigger (build piece 3), not in the run state.
   - Why: [TrainCheck's benchmark](https://web.eecs.umich.edu/~ryanph/slides/traincheck_osdi25_slides.pdf) held 20 real silent errors. Always-true rules caught 18; older methods, spike-and-trend monitoring among them, caught at most 3.
2. **Make replica drift name the rank.**
   - Today the step record carries only the spread per parameter. The per-rank fingerprints exist at gather time in `distributed.py` and are then discarded.
   - Change: work out at gather time which ranks sit further than the tolerance from the group's median, and record them only when drift is found. With two ranks it can only report the disagreement.
   - Why: [SCOUT](https://arxiv.org/pdf/2608.11034) attributes a fault to the rank that disagrees with a strict majority of its replicas, which takes at least three ranks.
   - Done when a three-process CPU run with one drifted rank names that rank, and a two-rank run reports drift without naming one.

The first candidates when the freeze lifts are hidden expert collapse (an expert still gets tokens but adds less and less), per-token routing entropy, slow ranks and update size per parameter.

## Prove

Five results make the claims credible, and none has been run yet. Pass lines are set here, before measuring.

| # | Result | Pass line | State today |
| --- | --- | --- | --- |
| 1 | Two-GPU run over NCCL: `replica_check.py` and the three `seeded_failures.py` commands | Every seeded case caught within one flush or check interval, on the seeded parameter. No drift reported on healthy steps at `1e-6` | Passes on CPU only |
| 2 | One of the author's real jobs | At least 1,000 steps, with a real anomaly found or zero false incidents | Not run |
| 3 | Head-to-head on the Granite run under `torch.compile`: TensorQuill, NeuralDBG and `wandb.watch` | TensorQuill router-only at 1.0% or less, with 0 graph breaks and 0 recompiles after warmup. All three tools' numbers are published, whatever they show | Not run |
| 4 | Agent benchmark on at least 8 real captures that carry loss and gradient curves | The agent names the exact step, layer or parameter, and kind in 85% of trials, and beats scalar-only data by 30 points | Not run. The Gate 1 harness on the 7 fixtures is the rehearsal, scored by its own pre-registered rules and without the Rune arm |
| 5 | Public release | On PyPI and crates.io, and 3 outside engineers run it on their own jobs | Packaging prepared, nothing uploaded |

Results 2 and 5 decide whether the product is wanted. Results 1, 3 and 4 decide what it may claim.

## Next actions

Four weeks from Nov 16, 2026, in this order. The weeks are a proposal; move them as results come in.

**Week 1: GPU runs and a real job**

- [ ] Commit this page to the repository as the plan's front page, and log today's decisions in v3
- [ ] Rent two GPUs and run `replica_check.py` over NCCL
- [ ] Run the three `seeded_failures.py` commands on a GPU
- [ ] Start TensorQuill on one real job and leave it running past 1,000 steps
- [ ] Build pieces 1 and 2: what was checked, and compact evidence

**Week 2: the agent loop**

- [ ] Build pieces 3 and 4: the trigger and the MCP server
- [ ] Write the explain-only agent instructions (piece 5)
- [ ] Rehearse the agent benchmark on the Gate 1 harness: seven fixtures, its own scoring, no Rune arm

**Week 3: the comparison**

- [ ] Run the head-to-head on Granite under `torch.compile`
- [ ] Run the agent benchmark on the real captures from week 1
- [ ] Make replica drift name the rank
- [ ] Choose how Python calls the Rust detectors, and move the first kind over

**Week 4: release**

- [ ] Claim `tensorquill` on PyPI and `tensorquill-server` on crates.io, then publish
- [ ] Post the demo: a planted failure that the agent names by step and layer
- [ ] Get three outside engineers to run it on their own jobs

## After the four weeks

Capability grows along one line, from detecting a failure to repairing it, and not sideways into a tracker. Steps 3 to 5 start only after result 5, when outside users ask for them.

1. **Detect.** Built: 32 incident kinds.
2. **Explain.** The agent loop in this plan.
3. **Act, with approval.** Pause a run, roll back to the last good checkpoint, or change the learning rate.
4. **Learn across runs.** Name the failures that recur and warn before they happen. LLM-Training-Debugger already finds the first step where two runs diverge and the outlier in a batch of runs.
5. **Widen.** Fine-tuning and RL jobs, where far more teams work.

Not planned: run-comparison dashboards, artifact and model versioning, sweeps and reports. MLflow and W&B keep those.

## Risks and open decisions

Three decisions are the author's to make, and five risks each have one answer in this plan.

**Decisions**

- **Confirm postponing the Rune viewer.** This reverses the v3 decision of Nov 14, 2026 that the app is built either way.
- **Choose how Python calls the Rust detectors.** A PyO3 extension ends the pure-Python wheel and needs a build per platform. A `detect` subcommand on `tensorquill-server` keeps the SDK pure Python, but offline use then needs the binary.
- **Price.** Per seat or per GPU-hour, decided only after result 5.

**Risks**

| Risk | What answers it |
| --- | --- |
| A debugger or tracker ships measured compile-safe capture, multi-GPU checks or its own agent first. LLM-Training-Debugger's v0.2 list names multi-node runs, Flight Recorder hangs, attention and MoE | Result 3, rerun before release; lead with multi-GPU failures and open evidence |
| Nobody outside wants it | Result 5; no hosted work before it |
| The agent reads "no incidents" as healthy | Build piece 1, before the agent benchmark |
| Thresholds tuned on Granite and SmolLM2-360M raise false alarms elsewhere | Result 2, on the author's own job; until then anomalies do not fire the trigger by default |
| Moving detectors to one Rust copy changes their answers | Keep the parity tests until each Python duplicate is deleted |
| A probe flush of about 150 ms delays a live acknowledgement; 620 ms was seen once | Logged as an open gap in v3; looked at after results 1 and 2 |

## Sources

- TensorQuill v3 — Product and Build Plan, Nov 13, 2026 (state as of Nov 16, 2026), for every TensorQuill figure and for the tracker, ARIA and Pluto rows
- A check of this page against `main` and against clones of the three debugger repositories, Nov 16, 2026: code and docs read, nothing run
- [NeuralDBG repository](https://github.com/LambdaSection/NeuralDBG)
- [MoEWatch v0.1 announcement](https://forum.fossunited.org/t/moewatch-v0-1-open-source-diagnostics-for-moe-routing-failure/9532)
- [LLM-Training-Debugger repository](https://github.com/YanKaFei/LLM-Training-Debugger)
- [TrainCheck, OSDI '25](https://www.usenix.org/conference/osdi25/presentation/jiang) for 18 of 20, and its [slides](https://web.eecs.umich.edu/~ryanph/slides/traincheck_osdi25_slides.pdf) for the baseline of 3, read Nov 16, 2026
- [SCOUT: failure localization in LLM pre-training](https://arxiv.org/pdf/2608.11034), read Nov 16, 2026
- [Motif 3: Technical Report](https://arxiv.org/pdf/2608.09119) and [Mechanism-driven monitors for LLM training instability](https://arxiv.org/html/2606.28116), for the later candidates; search excerpts only
