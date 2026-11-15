# TensorQuill MoE inspector bake-off

Compare a focused Rune incident screen with the full Rerun 0.43.2 viewer. Both views show the same detector summary, status colors and incident-layer expert load table by default. The table lists each expert's load and its change from the previous sample, sorted by that change. Rerun also retains the raw-share tensor and the diversity timeline.

The two tables differ in interaction. Rune's table can be re-sorted and follows the layer clicked in the heatmap. Rerun's is a Markdown document for the incident layer only; it cannot be sorted, and it follows the time cursor, so it shows the change at every sampled step after the first. The Rust package uses the sibling Rune checkout at `c4eda05889c984fff91ccd906e3c00cdfc4cae24`.

## Fixtures and the pipeline

`generate.py` makes seven synthetic runs, and every one of them goes through the shipped SDK, not a writer of its own:

1. Step records in the shape `tensorquill.init` writes (`schema_version`, `run_id`, `rank`, `attempt`, `step`, `ts`, `layers`, `grad_norm`, `moe`, `moe_top_k`, `moe_stats`, ...) are written with `tensorquill.JsonlSink`.
2. `tensorquill.load_records` reads that file and `tensorquill.build_evidence` runs the detectors on it. The overload rule, the diversity drop, the dead and starved experts, the gradient and activation spike rules and the table text all live in the SDK, and `generate.py` has no copy of them.
3. `tensorquill.export_rerun` writes the recording and its blueprint from the evidence and the records.

`python -m tensorquill <name>.jsonl -o ... --rerun ...` runs the same steps. `test_fixtures.py` checks that it gives byte-identical evidence and the same recording contents.

Each fixture writes three files into `artifacts/<name>/`: the step records `<name>.jsonl`, the viewer-neutral evidence `<name>.evidence.json` and the Rerun recording `<name>.rrd`. The seeded fixture's evidence is also written to `artifacts/stats.json`, which the Rune inspector bakes in at build time. `artifacts/answer-key.json` holds the planted failures. Keep it, this README and `generate.py` away from trial participants and agents. The fixture names say nothing about the failure.

| Fixture | Run | Planted failure (the answer, for the scorer) |
| --- | --- | --- |
| `seeded` | MoE, 60 layers × 128 experts, top-1, 4 sampled steps | Step 1,850, layer 31: expert 42 gets 36% of tokens, experts 1–14 get none |
| `unseen-1` | MoE, 48 × 64, top-1, 8 steps | Step 1,300, layer 17: expert 53 gets 44%, experts 20–27 get none. From step 1,500 layer 9 sends 40% to expert 5, later and in an earlier layer, so it is not the first |
| `unseen-2` | Dense, 12 blocks, 400 steps | Step 310: the gradient of `blocks.7.mlp.fc2.weight` jumps 80× after two steps of build-up (rising since 308), and the loss jumps with it |
| `unseen-3` | MoE, 24 × 64, top-4, 31 steps | Step 400, layer 11: experts 9–14 are each picked by 36% of tokens. Expert 3 is picked by 70% of tokens in every layer, always (healthy). The summary and table report shares of the router's assignments, so 36% of tokens is shown as 9% |
| `unseen-4` | MoE, 32 × 128, top-1, 9 steps | Step 2,000, layer 8: 40 scattered experts get no tokens, and none is overloaded |
| `unseen-5` | Dense, 20 monitored blocks, 400 steps | Step 260: the largest activation in `blocks.13` is 100× normal for four steps. `blocks.1` carries massive activations all along (healthy) |
| `unseen-6` | MoE, 24 × 32, top-1, 7 steps | Step 700, layer 3: expert 7 gets 39% and expert 22 gets 36%. At steps 500–600 layer 9 sends 34% to expert 13, just under the 35% rule (healthy) |

MoE counts carry sampling noise and a per-layer skew, except in `seeded`, whose counts are exactly as before. Everything is seeded by name, so `.jsonl` and `.evidence.json` are byte-identical on every run and in every process (the `.rrd` files hold the same contents but differ in bytes, as each carries its start time). The MoE fixtures sample steps (every 50 to 100 steps, or every 20 for `unseen-3`), as the seeded one always did; the dense ones record every step. Router names carry the evidence's layer number (`layers.31.router` is layer 31), which a real model's zero-based names would not.

Three behaviors of the SDK's detectors show up in the fixtures, and the trial questions should allow for them:

- A top-k collapse is judged against the router's own recent window, which fills with the collapse itself. In `unseen-3` the heatmap flags experts 9–14 at step 400, and by step 520 it flags nothing, although the router's diversity stays at 0.35 against 0.75 before. Judging "does it persist at the next sample" from the heatmap alone would mislead, and the diversity timeline does not.
- In a layer with many dead experts (a quarter of the experts, or diversity below 0.5) where the 35% rule flags nothing, the SDK lowers the bar to about twice an even share, 3.1% for 64 experts. With expert 53 at 34% instead of 44%, `unseen-1` still reports it as overloaded, while `unseen-6`'s 34% decoy, in a layer with no dead experts, is not flagged. So the 35% boundary is a boundary only beside healthy neighbors.
- The detector reports the first incident of each kind. `unseen-1`'s second event is in the samples and the heatmap but not in `incidents`.

## Checks

```sh
.venv/bin/python -m pytest experiments/moe-bakeoff -q
```

`test_fixtures.py` runs the whole pipeline for all seven fixtures and reads the results back from disk. The expected answers are the planted values in `generate.py` (`Fixture.answer` and `Fixture.events`), never the detectors' output. It checks that:

- Each fixture's evidence has exactly one incident, of the planted kind, at the planted step, with the planted layer and expert IDs (or parameter or module). `unseen-2` also names the parameter that started the spike and the step it began rising.
- The flagged MoE samples are exactly the planted ones, and the decoys are not flagged. In `unseen-3` they are a subset, for the reason above.
- The records before the planted step produce no incident, and the records up to and including it produce the incident at that step.
- The CLI gives byte-identical evidence, and a recording with the same contents.
- Each `.rrd` holds exactly one recording, named after its fixture, with the fixture's steps and summary. The exporter starts a global recording on each call, so this guards against one export leaking into the next.
- The records have the keys and types of a real capture (a toy MoE with `Router`, and a toy dense model with `capture="full"`), and `moe_stats` equals the SDK's own `_router_stats`.
- `seeded` is the answer key below, and a second process generates the same bytes.

The checks failed as they should when the plant was shifted by a sample, a dead expert dropped, the hot experts' IDs moved, the spike moved a step, or a decoy pushed over 35%.

## Smoke-test results

Rune and Rerun both read the seeded fixture. Its counts, diversity values and overloaded and dead lists are identical to the fixture the earlier stand-alone generator wrote (compared sample by sample on Nov 15, and that generator reproduced the saved Nov 13 `stats.json`), so the Rune rows below carry over from Nov 13 and were not rebuilt. The Rune binary baked in the earlier file; the new `artifacts/stats.json` has the same samples plus the SDK's `incidents`, `router` and rule fields, which the inspector ignores.

| Check | Result |
| --- | --- |
| macOS arm64 / Metal | Two runs produced byte-identical PNGs and trees. A scripted tap selected dead expert 2 and displayed 0% routing share. *(Nov 13.)* |
| Linux arm64 container / Vulkan | The container ran inside an arm64 Linux VM on the same Mac. Debian Trixie lavapipe (llvmpipe, LLVM 19.1.7) produced byte-identical outputs across two runs. Bookworm's Mesa 22.3.6 failed shader validation for `SHADER_FLOAT16_IN_FLOAT32`. Linux x86-64 and a separate Linux host remain untested. *(Nov 13. The Dockerfile now takes the fixture as a build context, and that change is untested: there is no Docker on the machine that edited it.)* |
| macOS Metal vs. Linux lavapipe | The element trees matched byte for byte. At 1160×720, 14.67% of pixels differed by at least one RGB channel value, and 1.68% had a channel difference greater than 16. The mean absolute channel difference was 0.946 and the maximum was 222 on a 0–255 scale. The source of the pixel differences remains unverified. *(Nov 13.)* |
| Chrome / forced WebGL2 | The WASM-only patch compiled and presented a frame, with no browser errors or failed requests. *(Nov 13.)* |
| Rerun 0.43.2, recording made by `export_rerun` | *(Nov 15.)* `verify_rerun.py` checks all 30,720 float32 shares against count-derived float64 shares with absolute tolerance `3e-8` and zero relative tolerance. The observed maximum error was `1.43e-8`. All 240 diversity values, the status image pixels, and the detector summary matched exactly. The verifier recomputes the loads, changes and row order of the three layer 31 expert load tables from the counts (the status label is the evidence's flag), and checks the 21 table rows in Rune's element tree (the Nov 13 tree) against the step 1,850 table. It rejected a one-token move between two experts' step 1,800 counts. The heatmap colors and the summary text come from the SDK on both sides of that check, so it shows that the recording holds what the SDK computes. The rules are tested against the planted failures by `test_fixtures.py`. The verifier also passes on `unseen-1`, `unseen-3`, `unseen-4` and `unseen-6`, and the capture loads and captures each of them (`unseen-3` was inspected by eye). |
| Rerun headless capture | *(Nov 15.)* `capture_rerun.py` loaded the SDK's recording and blueprint, waited until loading finished and two consecutive frames matched, and captured step 1,850. Two back-to-back captures were byte-identical. The layout matches the Nov 13 capture, except that the summary view is now named `Training incidents`, a `MoE routing` header sits above the heatmap and the table, and the legend line adds violet for starved and gray for idle experts. |

The human and agent timing trials are pending. These runs establish rendering and fixture parity.

Per the v3 plan (decided Nov 14, 2026), TensorQuill builds its own Rune app either way, and stock Rerun stays the viewer until that app reaches parity. This prototype is arm B of the Gate 1 trial: the trial decides which interactions (a sortable table, layer selection) come first. The evidence schema and detector output are shared across the experiment's viewers.

## Run the comparison

From the repository root, with an environment that has the SDK and `rerun-sdk` (the project's `.venv`, or `pip install -e 'sdk[rerun]'` in a new one; the SDK needs PyTorch):

```sh
.venv/bin/python experiments/moe-bakeoff/generate.py
```

```sh
cd experiments/moe-bakeoff
cargo build --locked
target/debug/moe-inspector --snap artifacts/rune.png --tree artifacts/rune-tree.json \
  --theme dark --size 1160x720 --scale 1
../../.venv/bin/python verify_rerun.py --rune-tree artifacts/rune-tree.json
PATH=../../.venv/bin:$PATH python capture_rerun.py
```

`generate.py` takes an output directory, repeated `--fixture NAME`, and `--no-rerun` to skip the recordings. `verify_rerun.py [artifacts/NAME]` and `capture_rerun.py --fixture NAME` take any MoE fixture and default to `seeded`; the dense fixtures have no routing recording, so neither accepts them.

Open `target/debug/moe-inspector` for the interactive Rune view, or `rerun artifacts/<name>/<name>.rrd` for a fixture's recording. All recordings share one application ID, `tensorquill`, so open one at a time. The raw counts and statistics are in `artifacts/<name>/<name>.evidence.json`.

## Score the experiment

Use `unseen-1` to `unseen-6`: their dimensions, incident steps, layers, and expert IDs differ from the seeded fixture's, and the seeded one is the smoke test. The plan runs three arms over all six: (A) stock Rerun with this blueprint and the evidence JSON, (B) the Rune prototype and the evidence JSON, and (C) the evidence JSON alone, with three fresh agent sessions per fixture per arm, and also times the author on A against B. Counterbalance UI order across engineers. Start both viewers at the same incident sample and confirm the shared detector summary and status heatmap are visible before starting the timer. For agents, use fresh sessions with the same model, prompt, time limit, and evidence JSON. Give each condition its own viewer tools, and give it only its fixture's `.evidence.json` (and the `.rrd` for the Rerun condition). Log start and submission times and tool calls. Include a JSON-only agent condition to measure what the views add.

Use the first sampled bad step, first affected layer, and expert IDs as orientation checks. Both summaries state those answers, and both tables state the incident layer's per-expert load change from the previous sample. For the timed comparison, ask participants to verify the alert with evidence that neither default screen states, such as one expert's load change in a neighboring layer or whether the overload persists at the next sample. Pre-register the questions and balance them, since some favor Rune's layer selection and others Rerun's timeline. Score against the same counts and compare median time and tool calls at matched accuracy. The plan's gate: accuracy across the six fixtures within ±5 percentage points between arms (or higher in arm B), and arm B cuts median agent tool calls or median human time to a correct answer by at least 2.0× against arm A. A single visible fixture is a smoke test.

### Seeded fixture answer key

The last healthy sample is step 1,800. The first affected sample is step 1,850, layer 31. Expert 42 receives 36% of routed tokens. Experts 1–14 receive zero. Diversity is 0.3094112778 using `exp(H) / 128`. The detector flags strictly above 35%. The 35% boundary gives an upper bound of about 0.3225. At the actual 36% share, distributing the rest evenly across 113 experts gives a ceiling of about 0.30941129.

### Not ready for the trial

- The Rune inspector bakes in the seeded fixture (`artifacts/stats.json`), so arm B cannot yet show `unseen-1` to `unseen-6`. It needs to load an evidence file at run time, and it has no view of the dense fixtures (`unseen-2` and `unseen-5`) at all, since it shows routing only. Decide whether the dense fixtures stay in the trial.
- The six timed questions are not written, and no arm has been run.
- Not rebuilt or rerun: the Rune inspector against the new `artifacts/stats.json` (that it ignores the SDK's extra fields is untested), the Rune rows above on the other fixtures, and the Docker build with its new named context.
- No dense recording (`unseen-2`, `unseen-5`) has been opened in a viewer. They were written and read back, not looked at.

## Linux headless rendering

Run this from `experiments/moe-bakeoff` with Docker and BuildKit available, after `generate.py` has written `artifacts/seeded`. The named contexts supply Rune’s source at the tested revision and the seeded evidence. The build stage uses the pinned Bookworm Rust image. The runtime uses Trixie for its newer Mesa Vulkan driver.

```sh
docker build --build-context rune=../../../rune --build-context seeded=artifacts/seeded \
  -t tensorquill-moe-bakeoff .
mkdir -p artifacts
docker run --rm --network none -v "$PWD/artifacts:/output" tensorquill-moe-bakeoff
docker run --rm --network none -v "$PWD/artifacts:/output" tensorquill-moe-bakeoff \
  moe-inspector --snap /output/rune-linux-repeat.png --tree /output/rune-linux-tree-repeat.json \
  --theme dark --size 1160x720 --scale 1
cmp artifacts/rune-linux.png artifacts/rune-linux-repeat.png
cmp artifacts/rune-linux-tree.json artifacts/rune-linux-tree-repeat.json
```

Check the logged Vulkan adapter for llvmpipe/lavapipe. Repeat within the same container and compare PNG and tree hashes. Across macOS and Linux, compare labels, bounds and screenshot regions with explicit tolerances. The embedded fonts reduce one source of variance, while renderer and text smoothing can still differ.

## Forced WebGL2 smoke test

The pinned Rune revision fails to compile with WebGL enabled because the native instance cache requires `wgpu::Instance` to be `Send` on WASM. Apply `rune-webgl.patch` from this experiment to an isolated copy of the sibling Rune checkout. Build the inspector in the same relative layout against the patched copy.

```sh
cargo install wasm-bindgen-cli --version 0.2.129 --locked
../../.venv/bin/python generate.py --fixture seeded --no-rerun
cargo build --locked --lib --target wasm32-unknown-unknown --features webgl
mkdir -p artifacts/web
wasm-bindgen --target web --out-dir artifacts/web \
  target/wasm32-unknown-unknown/debug/moe_inspector.wasm
cp web/index.html ../../../rune/crates/rune/assets/web/*.ttf artifacts/web/
node web/smoke.mjs artifacts/web "$CHROME_BIN" artifacts/webgl.png
```

The smoke script requires `playwright-core`, available as a Node dependency or via `PLAYWRIGHT_MODULE` pointing to its `index.mjs`. It starts a loopback server and launches a fresh Chrome profile. It disables WebGPU and Rune’s CPU-first startup, then checks for a WebGL2 context, a presented frame and clean browser logs. The inspector uses Rune’s core renderer with an RGBA heatmap texture. This test does not cover `rune-fx` or Rune’s default CPU-to-GPU handover.
