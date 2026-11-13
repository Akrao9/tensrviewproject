# TensorLens MoE inspector bake-off

Compare a focused Rune incident screen with the full Rerun 0.43.2 viewer. Both views show the same detector summary, status colors and incident-layer expert load table by default. The table lists each expert's load and its change from the previous sample, sorted by that change. Rerun also retains the raw-share tensor and the diversity timeline.

The two tables differ in interaction. Rune's table can be re-sorted and follows the layer clicked in the heatmap. Rerun's is a Markdown document for the incident layer only; it cannot be sorted, and it follows the time cursor, so it shows the change at every sampled step after the first. The Rust package uses the sibling Rune checkout at `c4eda05889c984fff91ccd906e3c00cdfc4cae24`.

## Smoke-test results

| Check | Result |
| --- | --- |
| macOS arm64 / Metal | Two runs produced byte-identical PNGs and trees. A scripted tap selected dead expert 2 and displayed 0% routing share. |
| Linux arm64 container / Vulkan | The container ran inside an arm64 Linux VM on the same Mac. Debian Trixie lavapipe (llvmpipe, LLVM 19.1.7) produced byte-identical outputs across two runs. Bookworm's Mesa 22.3.6 failed shader validation for `SHADER_FLOAT16_IN_FLOAT32`. Linux x86-64 and a separate Linux host remain untested. |
| macOS Metal vs. Linux lavapipe | The element trees matched byte for byte. At 1160×720, 14.67% of pixels differed by at least one RGB channel value, and 1.68% had a channel difference greater than 16. The mean absolute channel difference was 0.946 and the maximum was 222 on a 0–255 scale. The source of the pixel differences remains unverified. |
| Chrome / forced WebGL2 | The WASM-only patch compiled and presented a frame, with no browser errors or failed requests. |
| Rerun 0.43.2 | The headless viewer captured step 1,850. `verify_rerun.py` checks all 30,720 float32 shares against count-derived float64 shares with absolute tolerance `3e-8` and zero relative tolerance. The observed maximum error was `1.43e-8`. All 240 diversity values, the status image pixels, and the detector summary matched exactly. The verifier recomputes the three layer 31 expert load tables from the counts without using `generate.py`, and checks the 21 table rows in Rune's element tree against the step 1,850 table. It rejected a one-token change to the step 1,800 baseline. `capture_rerun.py` waits until loading finishes and two consecutive frames match; two back-to-back captures were byte-identical. |

The human and agent timing trials are pending. These runs establish rendering and fixture parity.

The v3 plan starts with stock Rerun. This prototype tests whether a sortable table and layer selection improve incident verification enough to justify custom UI. If the trials show a benefit, the first implementation to evaluate is a Rust incident view inside Rerun. The evidence schema and detector output are shared across the experiment's viewers.

## Run the comparison

```sh
python3 -m venv /tmp/tensorlens-moe-venv
. /tmp/tensorlens-moe-venv/bin/activate
python3 -m pip install 'rerun-sdk==0.43.2'
python3 generate.py --rerun
cargo build --locked
target/debug/moe-inspector --snap artifacts/rune.png --tree artifacts/rune-tree.json \
  --theme dark --size 1160x720 --scale 1
python3 verify_rerun.py --rune-tree artifacts/rune-tree.json
python3 capture_rerun.py
```

Open `target/debug/moe-inspector` for the interactive Rune view, or `rerun artifacts/moe.rrd` for the training timeline. The raw counts and statistics are in `artifacts/stats.json`.

## Score the experiment

Use unseen fixtures with equivalent dimensions and different incident steps, layers, and expert IDs. Counterbalance UI order across engineers. Start both viewers at the same incident sample and confirm the shared detector summary and status heatmap are visible before starting the timer. For agents, use fresh sessions with the same model, prompt, time limit, and stats JSON. Give each condition its own viewer tools. Log start and submission times and tool calls. Include a JSON-only agent condition to measure what the views add.

Use the first sampled bad step, first affected layer, and expert IDs as orientation checks. Both summaries state those answers, and both tables state the incident layer's per-expert load change from the previous sample. For the timed comparison, ask participants to verify the alert with evidence that neither default screen states, such as one expert's load change in a neighboring layer or whether the overload persists at the next sample. Pre-register the questions and balance them, since some favor Rune's layer selection and others Rerun's timeline. Score against the same counts and compare median time and tool calls at matched accuracy. The proposed gate is at least a 2× reduction in median time to correct diagnosis with accuracy preserved. A single visible fixture is a smoke test.

### Seeded fixture answer key

The last healthy sample is step 1,800. The first affected sample is step 1,850, layer 31. Expert 42 receives 36% of routed tokens. Experts 1–14 receive zero. Diversity is 0.3094112778 using `exp(H) / 128`. The detector flags strictly above 35%. The 35% boundary gives an upper bound of about 0.3225. At the actual 36% share, distributing the rest evenly across 113 experts gives a ceiling of about 0.30941129.

## Linux headless rendering

Run this from `experiments/moe-bakeoff` with Docker and BuildKit available. The named context supplies Rune’s source at the tested revision. The build stage uses the pinned Bookworm Rust image. The runtime uses Trixie for its newer Mesa Vulkan driver.

```sh
docker build --build-context rune=../../../rune -t tensorlens-moe-bakeoff .
mkdir -p artifacts
docker run --rm --network none -v "$PWD/artifacts:/output" tensorlens-moe-bakeoff
docker run --rm --network none -v "$PWD/artifacts:/output" tensorlens-moe-bakeoff \
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
python3 generate.py
cargo build --locked --lib --target wasm32-unknown-unknown --features webgl
mkdir -p artifacts/web
wasm-bindgen --target web --out-dir artifacts/web \
  target/wasm32-unknown-unknown/debug/moe_inspector.wasm
cp web/index.html ../../../rune/crates/rune/assets/web/*.ttf artifacts/web/
node web/smoke.mjs artifacts/web "$CHROME_BIN" artifacts/webgl.png
```

The smoke script requires `playwright-core`, available as a Node dependency or via `PLAYWRIGHT_MODULE` pointing to its `index.mjs`. It starts a loopback server and launches a fresh Chrome profile. It disables WebGPU and Rune’s CPU-first startup, then checks for a WebGL2 context, a presented frame and clean browser logs. The inspector uses Rune’s core renderer with an RGBA heatmap texture. This test does not cover `rune-fx` or Rune’s default CPU-to-GPU handover.
