"""B8, PT-115607: a missing guard lets TorchDynamo reuse a stale compiled step, so the model does not learn.

Source: TrainCheck's own reproduction, `example_pipelines/115607_bug.py`, recovered from the
clone's history (`git show 471f57e^:example_pipelines/115607_bug.py`; it was deleted in
471f57e). Kept as written, except that `torch._dynamo.optimize("eager", save_config=False)`
loses `save_config=False`, a keyword PyTorch 2.15 no longer takes (it raises a TypeError), and
the three TensorQuill lines (start, log, finish). TensorQuill is started inside the compiled
function, after the model is built there, because the script builds it nowhere else; Dynamo
then traces `tensorquill.init`, which fails (an AssertionError on `mark_static_address`), so
`--disable` wraps `init` and `session.log` in `torch.compiler.disable`.

The script runs one training loop eagerly and once under Dynamo. At iteration 3 no backward
runs, so the gradients are None and Adam's `step` must be a no-op; the bug made the compiled
loop step anyway (its step counter reaching 6 where eager stops at 5). Buggy: the compiled
loop. Healthy: the eager one.
"""

from __future__ import annotations

import torch

import harness

BUG = "B08-PT-115607"
COUNTS: list[float] = []


def make_loop(run_id: str, weights: int | None, disable: bool):  # type: ignore[no-untyped-def]
    start = torch.compiler.disable(harness.start) if disable else harness.start

    def training_loop():  # type: ignore[no-untyped-def]
        input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(3, 2)

        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.Sigmoid(),
            torch.nn.Linear(3, 1),
            torch.nn.Sigmoid(),
        )

        params = list(model.parameters())
        optimizer = torch.optim.Adam(params)
        session = start(model, run_id, weights=weights)  # TensorQuill
        log = torch.compiler.disable(session.log) if disable else session.log

        for i in range(6):
            optimizer.zero_grad()
            # Test that step behaves as expected (a no-op) when grads are set to None
            if i != 3:
                output = model(input)
                loss = output.sum()
                loss.backward()
                log({"train/loss": loss})  # TensorQuill

            optimizer.step()
            print("step", optimizer.state[params[0]]["step"])
            COUNTS.append(float(optimizer.state[params[0]]["step"]))
        return session

    return training_loop


def main() -> None:
    p = harness.parser(__doc__)
    p.add_argument("--disable", action="store_true",
                   help="wrap tensorquill.init and session.log in torch.compiler.disable; without it, Dynamo "
                        "traces init and fails on PyTorch 2.15")
    args = p.parse_args()
    run_id = harness.run_id_for(BUG, args)
    harness.folder_for(run_id, clean=True)
    training_loop = make_loop(run_id, args.weights, args.disable)
    if args.variant == "buggy":
        training_loop = torch._dynamo.optimize("eager")(training_loop)
    session = training_loop()
    symptom = {
        "adam_step_counter": COUNTS,
        "expected_in_eager": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
        "bug_present": COUNTS != [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
    }
    settings = {"iterations": 6, "optimizer": "Adam(lr=1e-3)", "compiled": args.variant == "buggy",
                "tensorquill_calls_under_compiler_disable": args.disable,
                "source": "TrainCheck 115607_bug.py from the clone's history; save_config dropped"}
    harness.finish(session, run_id, bug=BUG, variant=args.variant, symptom=symptom, settings=settings,
                   relevant=("stalled_weights", "update_spike", "repeated_steps"))


if __name__ == "__main__":
    main()
