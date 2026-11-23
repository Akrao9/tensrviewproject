# Stage 6: Comprehension Test

**Target:** `TensorLens v3 — Product and Build Plan.md` (lines 1–397, the whole file)
**Date:** Nov 14, 2026
**Reviewer:** `egm-comprehension`, completed. It read only the plan: no other files, no web pages and no standard, as this stage requires.

**Result:** 1 Blocker, 7 Major, 12 Nits.

## What this proposes

TensorLens is a PyTorch SDK added with one call, `tensorlens.init(model)`. It records cheap per-layer statistics inside a `torch.compile`d training step:
- activation mean, std, max and zero fraction;
- gradient norms;
- MoE router expert counts;
- float64 weight fingerprints compared across GPUs.

These go into fixed GPU buffers and move to the CPU every N steps. Histograms and attention maps would come from a periodic probe pass, and full activations from a replay when a detector fires.

Detectors look for MoE routing imbalance, non-finite values, cross-GPU weight drift and a few other failures. They produce viewer-neutral JSON "evidence", plus a Rerun recording (`.rrd`) and blueprint. Engineers browse the recording in stock Rerun. The team's own coding agent reads the JSON and drives Rerun through its viewer MCP. Scalars stay in MLflow or W&B, linked by run ID.

The plan runs 12 weeks in gated phases. It ends in a free Apache/MIT release. A paid hosted monitoring tier would be decided after outside teams adopt the tool.

## How the existing system works, per this document

An SDK exists in `sdk/`. Each rank writes its records to its own JSON Lines file through a background writer thread. Arrow Flight and the Rust ingest server are not built.

Router-only capture is the default for MoE models. Full capture (`capture="full"`) is an opt-in that adds about 4%. Seven to nine Python detectors run on the JSONL files after the run. An exporter taken from an earlier "MoE bake-off" turns the results into `.rrd` files that open in stock Rerun 0.43.2.

All of this has been run on IBM Granite 1B-A400M in Colab, on one RTX PRO 6000 or one A100. Router-only capture meets the 1% overhead gate on the RTX PRO 6000 but not on the A100. The cross-GPU check has only run on two CPU processes. The probe pass, replay, ingest server, CI wiring and Lightning callback are not built.

## Verdict

The document explains *what* is being built and *why* well: the product, the gap it fills, the competitors, the cross-GPU detector and the capture rules. It explains less well *how the parts connect*:
- the architecture and roadmap diagrams do not render;
- the data path is described in ways that contradict each other;
- several central terms are never defined: "Rune", the "bake-off", the evidence and incident format, N and M.

## Findings

### 1. Blocker: the data path from capture to ingest contradicts itself, and the build steps describe the replaced design
- **Where:** lines 24, 28, 36–38, 70, 282, 286 (step 14 transport notes), 315
- **Problem:** The text describes three different paths:
  - line 70 and step 11 (line 282): a worker thread builds Arrow batches and sends them straight over Arrow Flight to ingest;
  - line 286, the "agreed design": the SDK always writes a local append-only file, and an opt-in sender reads it from a saved position;
  - line 36: the SDK writes JSON Lines "until the ingest server exists", which reads as temporary.

  Where detectors run is also inconsistent: the ingest server (line 24), Python on files afterwards (lines 37, 286), or open (line 358).
- **Why it matters:** An engineer following steps 11 and 14 would build the direct-streaming path that line 286 rejects. It is also unclear whether a free user (line 311, whose free tier lists no ingest server) needs an ingest server at all.
- **Recommendation:** Add a short current-architecture paragraph to "How it works". It should say:
  - which process runs where (training node, ingest host, user machine);
  - what file or stream connects them;
  - where detectors run in v1 and later;
  - which parts are free or self-hosted.

  Update line 70 and step 11 to the local-file-plus-sender design. Move the transport research in step 14 to a rationale note.

### 2. Major: the architecture section depends on a diagram that does not render
- **Where:** lines 22–28
- **Problem:** "[embedded content: How it works · 7 parts, 8 flows]" carries the section. The only text is two summary sentences, so the reader cannot tell what the 7 parts or 8 flows are. "Tensor summaries take their own path" (line 28) never says what that path is.
- **Recommendation:** Describe the parts and flows in prose or a list, so the section stands without the diagram.

### 3. Major: "Rune" is never defined
- **Where:** lines 220, 275–276
- **Problem:** Gate 1 compares "the Rune incident prototype" and "Rune's layer selection" against stock Rerun, and line 220 says "Rune remains an option if a separate TensorLens app becomes necessary." The reader cannot tell whether Rune is a separate viewer, an earlier custom app or a typo for Rerun. Nor can they tell how it relates to the "Incident view" row (line 40), which is a custom view inside Rerun. Without that, the Gate 1 experiment, a key decision point, cannot be understood. "Default screen" (line 275) is also undefined.
- **Recommendation:** Define Rune in one sentence where it first appears: what it is, who built it, its status, and that it is the prototype arm in Gate 1.

### 4. Major: the "MoE bake-off" is the starting point but is never explained
- **Where:** lines 32, 39, 210, 274–276, 302, 335
- **Problem:** The plan reuses "the MoE recording and blueprint from the bake-off", and the exporter, `generate.py` and the release demo all come from it. The only description is a link the reader cannot open, plus line 335 ("synthetic … verified recording read-back and headless rendering"). It never says what was compared, between which options, or what was decided.
- **Recommendation:** Add two or three sentences: what the bake-off was, what it produced, and what it decided.

### 5. Major: the evidence and "incident" format is never specified
- **Where:** lines 9, 24, 39, 91, 127, 287, 311
- **Problem:** The JSON evidence is the core product artifact, the thing the agent reads. Yet the only concrete example is the cross-GPU evidence (a rank-by-layer spread table plus the first step of divergence, line 127). "Incident" is used (line 39, "incident summary"; line 91, "incidents carry the logged values") but never defined. An engineer could not write or consume the evidence from this document.
- **Recommendation:** Define an incident and give a brief field outline of one evidence record: detector, step, layer or expert, values, attached scalars, and a reference into the recording.

### 6. Major: the detector set is inconsistent and mostly undefined
- **Where:** lines 37, 243, 287, 335
- **Problem:** The counts disagree:
  - three detectors listed (line 37);
  - "the first two" (step 15, line 287);
  - about nine marked done (line 243);
  - "seven detectors" (line 335).

  Several detectors ("weights not updating", "repeated steps", "drift and dead outputs", "starved experts") are never defined beyond the thresholds in lines 251 and 261. The definitions are left to a linked failure list the reader cannot see.
- **Recommendation:** Add a short detector table: name, signal used, firing rule, and status.

### 7. Major: N and M have no values, and N means two different things
- **Where:** lines 69–72, 82–86, 94, 122
- **Problem:**
  - Neither "every N steps" (the transfer and ring-buffer size) nor "every M steps" (the probe) has a default, so the Tier 0 size of "1.3 KB per step" cannot be turned into memory or bandwidth.
  - The cross-GPU detector says "every N steps" (line 122), but the as-built note sets 50 steps separately (line 94).
  - "Diversity exp(H)/N" (line 83) uses N for the number of experts.
- **Recommendation:** Give defaults or ranges for N and M. Use a different symbol for the expert count. Make line 122 match the `replica_every` default.

### 8. Major: the paid tier promises features with no design
- **Where:** lines 312, 46, 353
- **Problem:** "Approved fixes delivered as pull requests" and "round-the-clock monitoring, alerts" are paid features. Yet the agent loop and live fixes are postponed (line 46), and "who explains failures" is open (line 353). The reader cannot tell what would produce the fixes.
- **Recommendation:** Mark these as aspirational or post-v1, or link each to the component that would deliver it.

### Nits

| # | Where | Finding | Recommendation |
| --- | --- | --- | --- |
| 9 | 67, 71, 94, 206, 236 | Terms used before they are defined: "Tier 0" (67) before the tier table (78); "Gate 2" (206) and "the overhead gate" (71) before the roadmap; "Granite" (94) before 236; "dogfood" (236) before 266. | Add a one-line pointer at first use, or a small glossary near the top. |
| 10 | 257 | "Stage 3 measured +4.15%": "Stage 3" appears nowhere else; the plan uses phases, steps and gates. | Name the step or gate it refers to. |
| 11 | 226–228 | The roadmap diagram "predates this version" but is still embedded, and the text says the plan has five phases. | Remove or regenerate the diagram. |
| 12 | 232 vs 74, 91, 94, 258, 286, 351 | Progress is "as of commit `d78fa78`", but entries dated Nov 14 appear throughout. | Give one date for the snapshot. |
| 13 | 75, 92, 355 | The design says rank 0 reports Tier 0 under data parallelism; the as-built note says each rank writes its own file; line 355 lists it as open. | Make the three statements agree. |
| 14 | 93–94, 239, 252 | Two float64 checks run every 50 steps (stalled weights, cross-GPU). Line 93 puts the weight check in full capture only, but does not say whether the cross-GPU check runs in the router-only default or whether the two share work. | State which checks run in each capture mode. |
| 15 | 87 | Tier 3 replay is one phrase and does not say which weights it uses; current weights would not reproduce the flagged step. | Say how replay gets the right weights, or mark it undesigned. |
| 16 | 9, 24, 42 | Rerun and "blueprint" (Rerun's saved view layout) are not described; a PyTorch engineer may not know them. | Add a one-line definition of each at first use. |
| 17 | 91 | `session.log(...)` appears without saying where `session` comes from. | Show `session = tensorlens.init(...)`. |
| 18 | 245, 258 | The A100 router-only overhead is given in percent (245) and in ms (258), with no baseline step time to connect them. | State each GPU's baseline step time. |
| 19 | 46 | "Your own VLM agent loop" and "agent-written layout blueprints" assume the reader knows the v2 plan. | Add a few words on each. |
| 20 | 118 | TrainCheck is cited as an authority but not introduced. | Add a short description: a research tool that catches silent errors in deep-learning training. |

## Overlap with Stage 5

Findings 1, 6, 12 and 13 repeat Stage 5's M1, M2 and N5 from a reader's side: the same contradictions confuse someone with no project history. The new problems Stage 6 raises are undefined terms (Rune, the bake-off, incident, N and M) and the diagrams that don't render.
