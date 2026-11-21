# Cross-rank activation compare (K2): design

**A design page for the author, Nov 21, 2026. No code in wave AD.** It answers the design choice the v3 plan's row K2 leaves open, and names the first test.

## What it is for

`replica_drift` compares the weights of data-parallel replicas. A fault that corrupts a rank's forward pass and leaves its weights equal is not seen by it: a bad GPU that computes a wrong activation, whose effect is then averaged into every rank's gradient. K2 compares what the ranks compute, not what they hold.

## The problem with comparing training steps

Ranks in data parallelism see different batches. Their activation statistics at one step differ by far more than rounding, so a comparison of ordinary training forwards needs a tolerance, and a tolerance wide enough for healthy ranks hides a small fault.

## The design proposed: one shared probe input

Every rank in a replica group runs the same input through the same weights, and the results are compared.

- **The input** is the probe batch TensorQuill already has (`probe=`), which the author approved as the starting point. It is fixed for the run, so every rank holds the same tensor.
- **When:** on the probe's own schedule (`probe_every` steps), the step every rank reaches by counting. No rank waits for another.
- **What is recorded:** per watched block, a short signature of the block's output on the probe input: its exact byte signature, as the replica observation has for parameters, and two numbers (`std`, `abs_max`) for a reader.
- **Who compares:** not the ranks. Each rank writes its own observation, and the server or the all-ranks reader compares them, exactly as replica observations are compared since wave Z. No collective is issued and nothing can block training.
- **What a difference means:** the ranks held equal weights (the replica observation of the same step says so) and computed different outputs from the same input. That names the rank that differs from the others, by the rule D1 already has: with three ranks or more, the one apart is named; with two, the attribution is inconclusive.

## Equivalent peers

Only ranks that should compute the same thing are compared:

- The same replica group, as `replica_group` names it.
- The same weights at that step: the comparison is made only where the replica observation of that step is equal across the ranks compared. Where the weights differ, the activation difference says nothing new and the reader says so.
- The same device type and dtype. A CPU rank and a GPU rank, or float32 and bfloat16, are not peers.
- A model sharded across ranks (tensor parallelism, FSDP's sharded parameters) has no equivalent peers for a block. The kind is `not_checked` there, with that reason.

## Determinism

Equal bits are expected only when the computation is deterministic on that hardware. So:

- The exact signature is evidence, as the parameter signature is, and raises nothing in the first build.
- The two numbers are compared with a tolerance taken from healthy runs on the same hardware, which do not exist yet. Until they do, the kind records and does not raise.
- Dropout and other training-mode randomness would make every rank differ. The probe already runs in evaluation mode and restores the mode afterwards.

## Coverage

A reader must be able to tell "compared and equal" from "not compared":

- Which ranks reported an observation at the probe step, which were compared, and which had no peer.
- A rank whose probe failed or was skipped is listed as not observed, never as equal.
- With one rank observed, zero peers were compared, in the words the replica contract already uses.

## The first test, before anything else is built

**Turning the probe on leaves the next ordinary training step unchanged.** Two runs from the same seed on CPU, one with the cross-rank probe and one without. After the probe step, the next training step's loss, every parameter and the random-number state must be bit for bit the same in both. If this fails, the probe disturbs training and the design stops there.

## Then, in order

1. CPU, two and three ranks under gloo, clean: every rank's signature equal at every probe step.
2. CPU, three ranks, one rank's block output corrupted on the probe forward only: the rank is named, and `replica_drift` stays quiet.
3. The same with two ranks: inconclusive, both named as differing.
4. A real distributed job on GPUs: how often healthy ranks differ in the exact signature, and by how much in the two numbers. That measurement sets the tolerance, and is run on the allocation proof result 1 uses.

## What it costs

One extra recording per watched block per probe, on a forward the probe already runs. No extra forward, no collective. The cost is re-timed on the probe's own benchmark before the kind is on by default.

## Not decided here

- Whether the signature covers the whole output or a fixed sample of it. A whole output of a large block is slow to hash; the parameter signature's sampling rule is the candidate.
- Whether the comparison also covers the gradient of the probe loss, as the shadow-replica design in the literature does. It would see more and costs a backward pass.
- The kind's name.
