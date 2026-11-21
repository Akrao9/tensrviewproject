# Draft of a Gate 1 amendment

**Entered on Nov 21, 2026**, in `experiments/gate1/PREREGISTRATION.md` on the branch `contracts/wave-ad`, on the author's approval with two edits: the author's own words for how the score is read, and for what was verified. The text below is the draft as it was proposed; the entered text is the preregistration's.

## What was measured (Nov 21, 2026)

`harness.py build`, all four arms, with lane AD7's base (`9a78db1`) and its head (`586181f`), compared file by file. Lane AD7 made the same comparison without arm R and gave the same list.

- **Arm A:** no file differs.
- **Arms C and R:** `evidence.json` and `summary.txt` are byte for byte the same as each other on all seven fixtures, before and after. Every difference below is in both arms.
- **Arm B:** its trial folder is unchanged; the evidence file its app opens is the changed one.
- **Arm R's recording** (`run.rrd`): on `unseen-2` and `unseen-5` the recorded summary text gains the score, as `summary.txt` does. On the other five no score text is in the recording. The recordings differ in their bytes between any two builds, so their other content was compared only by that text.

| Fixtures | What differs in `evidence.json` | What differs in `summary.txt` |
|---|---|---|
| All seven | `kind_checks.activation_drift.records_needed`: 60 becomes 110 | Nothing |
| `unseen-2` | The one incident and its first parameter each gain `score`. The evidence gains the block `gradient_trend` | Line 1 gains the two scores in words |
| `unseen-5` | The one incident's first module gains `score`. The evidence gains the block `gradient_trend` | Line 1 gains the score in words |

No incident's kind, step or location differs on any fixture. The per-step samples and the run state are the same. Neither changed rule fires or falls silent on a fixture.

Still to be confirmed on the trial tree, where the server's detectors and the last two lanes are merged: that the differences from `eb5a71b` are exactly these.

## Proposed text

### Nov 21, 2026: a score beside a ratio, a gradient trend, and 110 records for `activation_drift`

Entered before any trial was run. It changes what arms C and R are shown (`evidence.json` and `summary.txt`), and the evidence file arm B's app opens. It changes no question, no answer key and no scoring.

- **What changed.** Three things, from the detector rules contract of Nov 21, 2026 (`DETECTOR_RULES_CONTRACT.md`):
  - An incident that records a `ratio` against a window of earlier values also records a `score`: how far the value is from the window's median, in units of the window's spread. The summary line says it in words after the ratio. This shows on the two fixtures whose incident carries such a ratio.
  - The evidence gains a block `gradient_trend` where the records hold gradient norms: the trend of the total gradient norm over the run. This shows on the same two fixtures.
  - `activation_drift` is judged against a trailing baseline and needs 110 records, where it needed 60. On every fixture its entry in `kind_checks` says 110.
- **What did not change.** The incidents' kinds, steps and locations, the per-step samples and the run state of all seven fixtures. The fixtures and `generate.py`. The answer key.
- **Fairness.** Arms C and R are shown byte-identical `evidence.json` and `summary.txt`, as before.
- **What a participant may make of it.** The score and the trend are descriptions. No rule reads either, and neither is a finding. A large score beside a ratio says the same thing the ratio says, in other units. A falling or rising trend is not an incident.
- **Checked.** The files of all four arms built before and after the change and compared, as recorded in the plan. Not run: a trial with an agent.
