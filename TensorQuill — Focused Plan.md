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
| SCOUT, as lm-resiliency, BSD 3-Clause | Names the faulty rank for hangs, slow ranks and silent data corruption, by strict majority among equivalent replicas, with an observer outside the training process and replay of sampled layers | No per-layer statistics, routing, attention or gradient checks, and no evidence file for an agent. Its README says under 1% overhead; the paper says throughput overhead was not measured |
| SageMaker Debugger (smdebug), Apache 2.0 | Hooks that save tensors, and a runner for rules written against them; 21 built-in debugger rules and 11 profiler rules on SageMaker | The built-in rules run only on SageMaker. Saves raw tensors every 500 steps from one worker by default, so a one-step spike falls between saves. Nothing for torch.compile, and no rule compares workers. Last changed in 2023 |

Four things none of the alternatives states, as read on Nov 16, 2026:

1. **Multi-GPU failures.** Weight drift between replicas, silent or stuck ranks, and resume mismatches. None of the three debuggers detects them: NeuralDBG only warns under DDP, and LLM-Training-Debugger records rank but says multi-rank is untested. SCOUT, open source as lm-resiliency, does localise hangs, slow ranks and silent data corruption to a rank, with no per-layer evidence. One limit of TensorQuill's own check: a corrupted gradient reaches every replica alike, so replica drift does not see it.
2. **Measured cost under `torch.compile`.** Zero graph breaks and recompiles are checked in CI on CPU and were measured on single-GPU runs. A 1% overhead gate applies. None of the three debuggers publishes a measured compile cost.
3. **Attention and MoE routing together.** The probe reads attention alongside FlashAttention, which never materialises the attention matrix.
4. **Layer-level evidence for any agent.** A JSON file and six read-only MCP tools, built Nov 16 and 17, 2026, for whichever agent, tracker and cloud the team has. The trackers' MCP servers carry only what was logged, and NeuralDBG's does not read a live run.

Points 1 and 2 become public claims only after results 1 and 3 under Prove.

The tracker, ARIA and Pluto rows are carried over from the v3 competitor check of Nov 16, 2026. The three debugger rows were checked the same day against clones of their repositories: code and docs read, nothing run.

## Freeze

Five lines of work stop now. A sixth, new detector kinds, was lifted on Nov 17, 2026. Each returns only when its condition is met.

| Work | Decision | Returns when |
| --- | --- | --- |
| New detector kinds | Lifted by the author on Nov 17, 2026. It had stopped at the 32 built; new kinds are now added one at a time | Returned on Nov 17, 2026 by the author's decision. Its condition, a real job showing a failure none of them catches, was not met |
| Rune viewer app | Postponed, confirmed Nov 16, 2026; Rerun stays the viewer | Outside users ask for views Rerun cannot give |
| Second copy of each detector | One Rust implementation for the 28 server kinds, called from Python; each Python duplicate is deleted as it is replaced. The four Python-only kinds stay as they are. State on Nov 16, 2026: the Rust path is built and opt-in for one read, watching, the all-ranks view, the thresholds and the MoE samples. Python is still the default, and no Python copy is deleted yet. Decided Nov 17, 2026: the Python detectors stay through the release, and whether to delete them is decided after it | Not applicable: this is a simplification, not a pause |
| Real-AWS and GCS runs, and the `s3://` reader | Postponed; the store code is built and stays | A user asks for a cloud store |
| Hosted tier, SSO, pricing | Postponed | Three outside teams run the SDK |
| RL, robotics, diffusion and multimodal packs | Not in this plan | A user brings a workload |

This page is the short front page of the plan. Decisions and measurements are still logged in the v3 plan, which stays the record.

## Build: the agent loop

Five pieces turn the evidence files into continuous monitoring by an agent. They are built in this order, and each has one done test. All five were built on Nov 16, 2026. The done tests of pieces 3 and 4 are still open, because no live agent has been tried. A sixth piece, comparing runs, joined the first release on Nov 16, 2026 and was built on Nov 17, 2026.

1. **Say what was checked.** Each evidence file gives every kind one of three values: checked, not configured or unavailable. Only unavailable degrades the run, and the summary carries the count, such as "9 of 32 kinds checked".
   - Built Nov 16, 2026 as `kind_checks` in `evidence.json` and in the server's report. A router-only Granite capture reads "4 of 32 kinds checked". The done test is met in tests. The all-ranks view carries the key since Nov 17, 2026. Built Nov 17, 2026: a kind still counts as checked from its first record, and each kind also says whether its rule has enough history yet, so too few attention probes no longer degrade a run.
   - Done when a router-only capture lists the weight and activation checks as not configured and still reads `healthy`, and a failed probe reads `degraded`.
2. **Compact evidence.** The agent's file holds the incidents and a short window of steps around each, with no per-step samples.
   - Done when a 10,000-step MoE run gives a file under 1 MB (proposed line; the full file is 86 MB today).
   - Built Nov 16, 2026 as `--compact`. The done test is met: the same run gives 198 KB with a window of 5 steps, and 1 MB holds a window of up to 28.
3. **A trigger.** `--watch` takes `--on-incident <command>` and a webhook URL, and fires once per new incident. It does not fire on a change of run state, because incidents never close. Invariants fire by default, and anomalies are opt-in until thresholds are tuned.
   - Done when a seeded failure starts the agent with no human step.
   - Built Nov 16, 2026, with `--fire-anomalies` and `--fire-kind` to widen what fires. The done test is open: in tests a planted failure starts a command once, but no agent has been started by it. Built Nov 17, 2026: a stall that reaches `critical` fires by default, and a `warning` stall fires only when its kind is named. One stall episode fires once, and a stall fires again after the run stores a step beyond it or the job is started again, also when the new attempt hangs before its first stored step. A killed job still fires and reads as a lost connection, not a hang. With a trigger set, the compact evidence is always written. A local run with no server cannot detect a hang.
4. **A read-only MCP server** over the same files, with six tools: list incidents, get one incident, get a series for one layer or parameter, get the all-ranks status, list runs and compare runs.
   - Done when a fresh agent session names the step, layer and kind for each fixture through it alone.
   - Built Nov 16, 2026 as `python -m tensorquill.mcp_server`. The done test is open: a scripted client names the kind, step and layer on all seven fixtures through the four tools, but no agent session has used them.
5. **Explain only.** The agent writes its diagnosis next to the incident. Pausing, stopping or changing a run needs a person's approval.
   - Done when the agent's instructions and tools offer no write action on the run.
   - Written Nov 16, 2026 as `sdk/AGENT_INSTRUCTIONS.md`. The done test is met: the instructions and the six tools offer no write action on the run.
6. **Projects, run names and comparing runs.** A run gets a project and a name beside its ID, as in W&B. `runs` lists a project's runs, and `compare` sets two or more side by side: what differs in the setup, what was checked, the incidents, and for a paired run the first step and layer where they part. Scalar charts and sweeps stay with MLflow and W&B.
   - Done when a healthy run and a seeded run compare as one incident in the second only, on the seeded parameter, and a router-only run against a full capture reports the difference in what was checked before any incident.
   - Accepted Nov 16, 2026 for the first release; the design is in v3. Built Nov 17, 2026, reading every rank of a run. The done test is met in tests on toy runs; no real GPU capture and no job of more than two ranks has been compared.

TensorQuill itself still only reports. Nothing in the SDK or server blocks or stops training.

## Detector changes

No new kinds were added while the freeze held; it was lifted on Nov 17, 2026. Labelling the 32 kinds is done, and one change to replica drift is planned for after the release.

1. **Label every kind** as invariant, anomaly or observation. Done Nov 16, 2026: all 32 are labelled in Python and Rust with a parity test, as 12 invariants, 11 anomalies and 9 observations.
   - Invariants are rules that must always hold, such as replica drift, repeated steps, duplicate batches, non-finite values and collective desync.
   - Anomalies compare a signal with its own recent history, such as gradient spikes, activation spikes and drift, and memory growth.
   - Observations record something that may be intended, such as the resume checks, GPU throttling and memory imbalance.
   - Invariants and anomalies both mark a run `unhealthy`, as the code does, and `degraded` means a monitor is unavailable. False alarms are handled in the trigger (build piece 3), not in the run state.
   - Why: [TrainCheck's benchmark](https://web.eecs.umich.edu/~ryanph/slides/traincheck_osdi25_slides.pdf) held 20 real silent errors. TrainCheck's inferred invariants caught 18; spike, trend and anomaly detection on loss and accuracy caught 2. This supports checking invariants. It is not a measurement of TensorQuill's own 12, which have not been run on those 20.
2. **Make replica drift name the rank.**
   - Today the step record carries only the spread per parameter. The per-rank fingerprints exist at gather time in `distributed.py` and are then discarded.
   - Change: work out at gather time which ranks sit further than the tolerance from the group's median, and record them only when drift is found. With two ranks it can only report the disagreement.
   - Why: [SCOUT](https://arxiv.org/pdf/2608.11034) attributes a fault to the rank that disagrees with a strict majority of its replicas, which takes at least three ranks.
   - Done when a three-process CPU run with one drifted rank names that rank, and a two-rank run reports drift without naming one.

The freeze on new kinds was lifted on Nov 17, 2026. New kinds are built one at a time, in this order: update size per parameter, a cross-rank activation compare (the check for corruption that replica drift cannot see), per-token routing entropy, output-logit growth in the language-model head, slow ranks as an observation, and hidden expert collapse (an expert still gets tokens but adds less and less). The v3 plan holds the build list with what each needs first.

## Prove

Five results make the claims credible, and none has been run yet. Pass lines are set here, before measuring.

| # | Result | Pass line | State today |
| --- | --- | --- | --- |
| 1 | Two-GPU run over NCCL: `replica_check.py` and the three `seeded_failures.py` commands | Every seeded case caught within one flush or check interval, on the seeded parameter. No drift reported on healthy steps at `1e-6` | Passes on CPU only |
| 2 | One of the author's real jobs | At least 1,000 steps, with a real anomaly found or zero false incidents | Not run. nanochat and nanoMoE are noted as candidates; whether they count as the author's own job is not decided |
| 3 | Head-to-head on the Granite run under `torch.compile`: TensorQuill, NeuralDBG and `wandb.watch` | TensorQuill router-only at 1.0% or less, with 0 graph breaks and 0 recompiles after warmup. All three tools' numbers are published, whatever they show | Not run |
| 4 | Agent benchmark on at least 8 real captures that carry loss and gradient curves | The agent names the exact step, layer or parameter, and kind in 85% of trials, and beats scalar-only data by 30 points | Not run. The Gate 1 harness on the 7 fixtures is the rehearsal, scored by its own pre-registered rules and without the Rune arm. Since Nov 17, 2026 its gate is the evidence arm against stock Rerun with the same evidence: accuracy within 5 points, and no more tool calls at the median. Loss is now logged in the four GPU scripts, but router-only captures carry no gradient norm, so these runs need full capture |
| 5 | Public release | Both packages on PyPI, the server as wheels, and 3 outside engineers run it on their own jobs | The SDK wheel is prepared and the server's wheels are not built. Nothing uploaded. crates.io was dropped on Nov 17, 2026 |

Results 2 and 5 decide whether the product is wanted. Results 1, 3 and 4 decide what it may claim.

## Release plan

Decided Nov 17, 2026. One pure-Python SDK and one Rust server wheel, both on PyPI, built for Linux first; then proof runs; then the release. Pushing `main`, the first Linux CI run and the small fixes are done. Nothing in steps 2 to 6 is built or run yet. The v3 plan holds the full entry.

1. **Record the decisions in the plan.** Done Nov 17, 2026: the packaging decision is logged in v3; the audience and gap text, "seventeen kinds", the crates.io lines and the CI row are fixed; and the detector count is checked against the code as 28 in the server and 4 in Python. Other status rows in v3 still carry their Nov 16 state and were not re-read.
2. **Packaging.**
   - Publish `tensorquill-server` as maturin wheels for Linux x86_64, Linux arm64 and macOS arm64, plus a near-empty fallback wheel for every other platform.
   - Check that the Linux wheels are tagged `manylinux_2_17`, and build arm64 on a native runner.
   - Keep `tensorquill` pure Python. It depends on the server package by version range, with no platform markers.
   - The binary reports one protocol number, and the SDK checks it.
   - `--watch` requires the Rust binary and stops with the install hint; `--detectors python` overrides that. One-shot reads use `auto` and fall back to Python with one warning.
   - Keep the Python detectors, and do not start the deletion of the Python duplicates.
   - Document `uvx --from tensorquill tensorquill-mcp`.
   - Test a clean install from TestPyPI in a Linux container. Then upload a 0.0.x of each package with trusted publishing, which also claims the names.
   - Create the GitHub organization and transfer the repository.
3. **Proof without a GPU.**
   - Run nanoMoE on the Mac past 1,000 steps with the balance loss on and off, then `compare` the two.
   - Run a live agent on a planted failure through the MCP tools.
   - Run the watchdog once at its real default timings.
   - Send 512 to 1,024 simulated ranks into the server and record flush time and memory per rank. Label the result synthetic, on CPU.
4. **Proof on GPUs.**
   - In Colab on the RTX PRO 6000: the three seeded-failure commands, the bf16 cases, the Granite head-to-head against NeuralDBG and `wandb.watch`, and the group C cell (GPU memory, NVML and data wait).
   - On a rental: only the two-GPU NCCL run.
   - Copy each run's `.jsonl` and evidence files out of Colab before the session ends, named by GPU.
5. **Agent benchmark.** Enter the model and time limit under Amendments, then run the trial.
6. **Release.**
   - Run `gpu_check.py` in Colab by hand. No GPU job runs in CI.
   - Cut the README to a short quickstart.
   - Add a framework table (DDP, FSDP2, ZeRO, Megatron, Lightning, HF Trainer), each row labelled run on GPUs, CPU only, or not run.
   - Move the measurements to a benchmarks file.
   - Publish both packages, post the demo, and get three outside engineers to run it.

**After the release**

- Keep or delete the Python detectors based on maintenance cost. Deleting them means adding wheels for the remaining platforms and making the binary required.
- Make replica drift name the rank, and move the attention maps out of `probe_json`.
- Add a container image when a user asks.

The new detector kinds, lifted the same day, are not placed in these steps. Whether the first one is built before the release is not decided.

## Next actions

Replaced on Nov 17, 2026 by the release plan above. The four weeks from Nov 16, 2026 are kept below as the record of what was planned.

**Week 1: GPU runs and a real job**

- [x] Commit this page to the repository as the plan's front page, and log today's decisions in v3
- [ ] Rent two GPUs and run `replica_check.py` over NCCL
- [ ] Run the three `seeded_failures.py` commands on a GPU
- [ ] Start TensorQuill on one real job and leave it running past 1,000 steps
- [ ] Try nanoMoE on Shakespeare, then nanochat's small run, as trial jobs (noted Nov 16, 2026; not started)
- [x] Build pieces 1 and 2: what was checked, and compact evidence

**Week 2: the agent loop**

- [x] Build pieces 3 and 4: the trigger and the MCP server
- [x] Write the explain-only agent instructions (piece 5)
- [ ] Rehearse the agent benchmark on the Gate 1 harness: seven fixtures, its own scoring, no Rune arm
- [ ] Try the MCP server and the trigger with a live agent on a planted failure (the done tests of pieces 3 and 4)
- [ ] Say what was checked in the all-ranks view (done Nov 17, 2026)
- [ ] Build piece 6: project and run name, the `runs` listing, `compare` and its two MCP tools (done Nov 17, 2026)
- [ ] Finish the two changes decided Nov 16, 2026: each kind says whether its rule has enough history, and a `critical` stall fires the trigger (done Nov 17, 2026)

**Week 3: the comparison**

- [ ] Run the head-to-head on Granite under `torch.compile`
- [ ] Run the agent benchmark on the real captures from week 1
- [ ] Make replica drift name the rank
- [x] Finish the Rust swap (the detect subcommand is built): golden files, thresholds as an option, MoE verdicts from Rust, and the unwired paths
- [ ] Make Rust the default, then delete each Python duplicate

**Week 4: release**

- [ ] Time the 64-rank benchmark again with exact number parsing on the live path (done Nov 17, 2026: about 4 ms more detector time per flush of 64 ranks)
- [ ] Claim `tensorquill` on PyPI and `tensorquill-server` on crates.io, then publish
- [ ] Post the demo: a planted failure that the agent names by step and layer
- [ ] Get three outside engineers to run it on their own jobs

## After the four weeks

Capability grows along one line, from detecting a failure to repairing it, and not sideways into a tracker. Steps 3 to 5 start only after result 5, when outside users ask for them.

1. **Detect.** Built: 32 incident kinds.
2. **Explain.** The agent loop in this plan, built Nov 16, 2026 and not yet tried with a live agent.
3. **Act, with approval.** Pause a run, roll back to the last good checkpoint, or change the learning rate.
4. **Learn across runs.** Name the failures that recur and warn before they happen. Comparing runs moved into the first release on Nov 16, 2026 (build piece 6), and LLM-Training-Debugger already finds the first step where two runs diverge and the outlier in a batch of runs.
5. **Widen.** Fine-tuning and RL jobs, where far more teams work.

Not planned: charts of scalars across runs, a hosted page of runs, artifact and model versioning, sweeps and reports. MLflow and W&B keep those.

## Risks and open decisions

Two decisions are still the author's to make, two were settled on Nov 16, 2026, and six risks each have one answer in this plan.

**Decisions**

- **Settled: the Rune viewer is postponed.** Confirmed Nov 16, 2026. This reverses the v3 decision of Nov 14, 2026 that the app is built either way. Rerun stays the viewer until outside users ask for views it cannot give.
- **Price.** Per seat or per GPU-hour, decided only after result 5.
- **Do nanochat and nanoMoE count as result 2?** Both can meet the pass line, but they are other people's training code, and result 2 asks for one of the author's real jobs.
- **Settled: Python reaches the Rust detectors through the server binary.** `tensorquill-server` has a `detect` subcommand that Python runs as a subprocess. The SDK wheel stays pure Python. Offline detection needs the binary on PATH until it ships as per-platform wheels, which the release plan of Nov 17, 2026 schedules. A PyO3 extension was not chosen, because detection is one pass over a file and the wheel would need a build per platform.

**Risks**

| Risk | What answers it |
| --- | --- |
| A debugger or tracker ships measured compile-safe capture, multi-GPU checks or its own agent first. LLM-Training-Debugger's v0.2 list names multi-node runs, Flight Recorder hangs, attention and MoE | Result 3, rerun before release; lead with multi-GPU failures and open evidence |
| Nobody outside wants it | Result 5; no hosted work before it |
| The agent reads "no incidents" as healthy | Build piece 1, built Nov 16, 2026, with a per-kind note of whether its rule has enough history (built Nov 17, 2026); the agent benchmark has to show it works |
| Thresholds tuned on Granite and SmolLM2-360M raise false alarms elsewhere | Result 2, on the author's own job; until then anomalies do not fire the trigger by default |
| Moving detectors to one Rust copy changes their answers | Golden files hold Python's answers, and the parity tests stay until each Python duplicate is deleted |
| A probe flush of about 150 ms delays a live acknowledgement; 620 ms was seen once | Logged as an open gap in v3; looked at after results 1 and 2 |

## Sources

- TensorQuill v3 — Product and Build Plan, Nov 13, 2026 (state as of Nov 17, 2026), for every TensorQuill figure and for the tracker, ARIA and Pluto rows
- A check of this page against `main` and against clones of the three debugger repositories, Nov 16, 2026: code and docs read, nothing run
- [NeuralDBG repository](https://github.com/LambdaSection/NeuralDBG)
- [MoEWatch v0.1 announcement](https://forum.fossunited.org/t/moewatch-v0-1-open-source-diagnostics-for-moe-routing-failure/9532)
- [LLM-Training-Debugger repository](https://github.com/YanKaFei/LLM-Training-Debugger)
- [TrainCheck, OSDI '25](https://www.usenix.org/conference/osdi25/presentation/jiang) for 18 of 20, and its [slides](https://web.eecs.umich.edu/~ryanph/slides/traincheck_osdi25_slides.pdf); the paper (arXiv 2506.14813, p. 11) gives the baseline as 2 of 20, read Nov 17, 2026
- [SCOUT: failure localization in LLM pre-training](https://arxiv.org/pdf/2608.11034), read Nov 16, 2026
- [Motif 3: Technical Report](https://arxiv.org/pdf/2608.09119) and [Mechanism-Aligned Observability for Training Instability of Large Language Models](https://arxiv.org/html/2606.28116), for the later candidates; the first from search excerpts only, the second read in full on Nov 17, 2026
- A review of 22 papers and the two SageMaker Debugger repositories, Nov 17, 2026, for the SCOUT and SageMaker Debugger rows and the corrected TrainCheck figure; the corrections were checked against the cited pages
