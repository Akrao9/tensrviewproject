# Replica observations: the record and how it is compared

Step 0 of the SDK-22 wave (Nov 20, 2026). The decision behind it is in `TensorQuill v3 — Product and Build Plan.md`, "Decision, SDK-22" and "Decision for the SDK-22 wave": the automatic drift check no longer gathers inside the training step. Each rank records what it sees; a reader that has every rank's record compares them.

This page is what the writer, the Python reader, the server and the driver build against. The cases in `sdk/tests/fixtures/replica/` are its examples, and `sdk/tests/test_replica_contract.py` holds a small reference reading of the rules that every case is checked with. Where this page and a case disagree, that is a fault in one of them, to be fixed before a lane builds on either.

Nothing here is built yet. No lane starts until the author releases it.

## 1. What a rank writes

On each scheduled step (`(step + 1) % replica_every == 0`, as today) a rank that was given a `replica_group` adds one key, `replica`, to that step's record:

```json
"replica": {
  "v": 1,
  "launch": "rank0:1795000000101",
  "group": [0, 1, 2],
  "device": "cuda",
  "projection": "name-v1",
  "signature": "fold64-v1",
  "params": {
    "layers.0.weight": {"shape": [8, 4], "dtype": "float32",
                        "fp": [1.25, -0.5, 0.125], "sig": "-4611686018427387904"}
  },
  "unread": {"layers.1.weight": "on meta"}
}
```

- **`launch`**: which start of the job this record belongs to (section 2). A string, or null when the rank could not learn it.
- **`group`**: the global ranks of the replica group this rank expects to be compared with, itself included, sorted. For the world group that is every rank. It is what "every required rank" means to a reader.
- **`device`**: the device type the fingerprints were computed on (`cpu`, `cuda`, `mps`).
- **`projection`**: `"name-v1"` when the third fingerprint is recorded, null when it is not. `name-v1` seeds a parameter's projection weights from its name, not from its place in a list, so two ranks that list the same parameters in another order compute the same number.
- **`signature`**: the name of the signature method, or null when no signature is recorded.
- **`params`**: one entry for each compared parameter, keyed by the parameter's name. `fp` is `[norm, sum]` or `[norm, sum, projection]` in float64. `sig` is the exact signature as a decimal string: a 64-bit number written as a JSON number loses digits in some readers. A fingerprint that is not finite is written as the string `"nan"`, `"inf"` or `"-inf"`, as layer statistics are, because the stream's encoder refuses a bare NaN.
- **`unread`**: parameters this rank could not read, with why. A parameter is in `params` or in `unread`, never both.

The key travels in `extra_json` on the stream and in Parquet, where every key without a column of its own already goes, so the stream's schema and protocol number do not change.

A signature is equal on two ranks exactly when the parameter's bytes, shape and dtype are equal. Readers compare signatures for equality only, and only when both ranks name the same method.

Old records keep `replica_spread`, the spread the gather computed. A reader gives such a record the verdict it always did (section 6).

## 2. Which start of the job a record belongs to

Each process makes its own `attempt` from its own clock, so healthy ranks of one start have different attempts. Attempts are therefore not matched between ranks. `launch` is.

`launch` has one source: **rank 0's attempt**, written `"rank0:<attempt>"`. Rank 0 shares it through the process group's store, in the exchange that already shares the run ID at `init`. Rank 0 never waits; the other ranks wait as long as they do for the run ID, and no longer. A rank whose store failed, or whose wait ran out, records `launch` as null. A job of one rank uses its own attempt.

It has to be a value that is new at every start of the job. A launcher's restart count is not: torchrun's `TORCHELASTIC_RESTART_COUNT` is 0 on every fresh launch, so a job stopped and started again by hand under the same run ID would carry the same count twice, and the old start's rows would pair with the new one's. One source also means two ranks of one start can never disagree about which source they used.

Two records can be compared only when they name the same run, the same `launch` (not null) and the same `group`. A reader never pairs records across launches: after a restart, an earlier launch's row at step 50 is not compared with the new launch's row at step 50.

Within one rank, when several attempts carry the same `launch` (which the SDK does not produce, but a file can hold), the newest attempt is used, by contract 1's order.

## 3. What can be compared

For one run, launch, group and step, a reader has zero or one observation from each rank of the group. Between the ranks that have one, each parameter is compared by name, when all of these hold on every rank that reports it:

| Check | When it fails |
|---|---|
| The parameter is in `params` on every reporting rank | Not compared; why: the `unread` reason, or "not recorded on rank N" |
| Same `shape` and `dtype` | Not compared; why names both |
| Same `device` | Fingerprints not compared; why: device types differ. Signatures are still compared |
| Same `projection` | The projection column is not compared; norm and sum are |
| Same `signature` method | Signatures not compared |

A parameter that is not compared is listed with its reason. It is never counted as agreeing.

## 4. The four facts a reader reports

They are separate, and a reader gives each one for each compared step.

**Coverage.**
- `complete`: every rank in `group` has an observation for this launch and step.
- `insufficient`: some do not. The reader names the missing ranks. This is not a clean result and not an incident.
- `not_established`: the records cannot say whether they belong to one start: a `launch` is null, or the ranks' groups differ. The reader says which. When some ranks of a step name a launch and one names none, both are said: the launch's step is `insufficient` without that rank, and that rank's row is `not_established`.

Only a step with `complete` coverage can agree or drift. A lone rank's file is `insufficient` for a group of more than one rank.

**Fingerprints.** For each compared parameter and each fingerprint column, the spread is the largest value minus the smallest, divided by the mean of the ranks' norms (never less than `1e-12`), as `spread_from_fingerprints` computes today. `drift` when any spread is above the tolerance (`1e-6` by default), or when any rank's fingerprint is not finite; else `agree`.

**Signatures.** `agree` when every rank's signature of every compared parameter is equal, `differ` otherwise, with the parameters that differ. Null when not compared.

**Attribution (D1, held).** Not part of this wave. When released it is computed from these same records: `agree`, `attributed` when a strict majority of the ranks hold one value and the others are named, `inconclusive` otherwise. Two ranks that differ are inconclusive.

## 5. What raises `replica_drift`

One rule for v1: **`replica_drift` is raised when coverage is `complete` and fingerprints `drift`.** The incident names the parameters, the step and the ranks compared.

- Fingerprints within tolerance and signatures different: the evidence says the signatures differ and names the parameters. No incident. Whether this raises on its own is decided after the eight-GPU run.
- Coverage `insufficient` or `not_established`: no incident, and the kind does not read as checked for that step.
- One step raises at most one `replica_drift` for a parameter. When a record also holds a `replica_spread` from an explicit `record_replica_spread()` call at the same step, that gathered spread decides the step, whether it shows drift or not, and the comparison of observations is shown as evidence only.

## 6. Old captures

A record with `replica_spread` and no `replica` is judged as before: drift when a spread is above the tolerance, on that rank's record alone. Its verdict does not change, and it is not mixed with observations.

## 7. When a verdict is due

- **The clock starts** when the last required observation is accepted: for the server, when it has stored and acknowledged the batch that holds it; for a reader of files, when the file holds the line.
- **The clock stops** when the verdict is visible to a reader: for the server, when the status or report that carries it is written; for a reader of files, at its next look.
- **The bound**: one server status interval (`--watchdog-seconds`), or one look of a watching reader.

A rank that is late leaves the step `insufficient` until its observation arrives, then the step is judged. A rank that never reports leaves it `insufficient` for good, and the reader keeps saying which rank is missing. Neither holds up a training step on any rank: nothing in the SDK waits for another rank's observation.

The proof for this uses a rank that is alive and training while its telemetry is delayed or suppressed. Killing a rank would stall the training's own collectives and prove nothing about TensorQuill.

## 8. What each lane takes from this

- **Writer** (`capture.py`, `distributed.py`): sections 1 and 2; no collective in a step by default; time the signature and the device synchronisation it needs, and report the record's size: with a parameter's name, shape and dtype written at every scheduled step, an observation of a few hundred parameters is tens of kilobytes (one recorded capture names 290).
- **Python reader** (`evidence.py`) and **server** (`detect.rs`, `ingest.rs`, `detect/data.rs`): sections 2 to 7; both pass every case in the fixtures with the same result.
- **Driver** (`replica_check.py`): section 7's two proofs, on two and three ranks; its end-to-end results are taken on the trial merge, where the real writer and readers are together.

## 9. Not settled here

- Whether a signature mismatch alone raises (after the eight-GPU run).
- D1's wording and acceptance test (held by the author).
- SDK-06: whether to record again after a second optimizer steps. Its cost is hashing, device synchronisation and record size, to be measured, not assumed free.
