"""Train a tiny MoE model, break its router part-way, and capture the run.

    python examples/toy_moe.py            # writes tensorquill-out/toy-moe.rank0.jsonl

At step 200 the router's bias is pushed towards expert 0, so the expert load collapses
while the loss keeps looking ordinary. The script then reads the capture back and prints
the first step where one expert took more than half the tokens.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tensorquill  # noqa: E402

N_EXPERTS, DIM, BATCH, STEPS, BREAK_AT = 8, 32, 256, 300, 200


class Router(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(DIM, N_EXPERTS)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h):
        logits = self.proj(h)
        return logits, logits.argmax(dim=-1)


class ToyMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(DIM, DIM)
        self.router = Router()
        self.experts = nn.ModuleList(nn.Linear(DIM, DIM) for _ in range(N_EXPERTS))
        self.out = nn.Linear(DIM, 1)

    def forward(self, x):
        h = torch.relu(self.inp(x))
        # Route on the zero-mean input so the untrained router starts roughly balanced.
        _, choice = self.router(x)
        stacked = torch.stack([expert(h) for expert in self.experts], dim=1)
        picked = stacked.gather(1, choice[:, None, None].expand(-1, 1, DIM)).squeeze(1)
        return self.out(picked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="also write tensorquill-out/toy-moe.rrd")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = ToyMoE()
    out = Path("tensorquill-out") / "toy-moe.rank0.jsonl"
    out.unlink(missing_ok=True)
    session = tensorquill.init(model, run_id="toy-moe", every=50, routers={"router": N_EXPERTS})
    compiled = torch.compile(model, backend="aot_eager")  # no C++ toolchain needed
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(STEPS):
        if step == BREAK_AT:
            with torch.no_grad():
                model.router.proj.bias[0] += 50.0  # the seeded failure
        x = torch.randn(BATCH, DIM)
        loss = (compiled(x) - x.sum(dim=1, keepdim=True)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    session.close()

    records = tensorquill.load_records(out)
    evidence = tensorquill.build_evidence(records)
    evidence_path = Path("tensorquill-out") / "toy-moe.evidence.json"
    evidence_path.write_text(json.dumps(evidence, separators=(",", ":")) + "\n", encoding="utf-8")

    first = tensorquill.first_incident(evidence)["step"]
    print(f"captured {len(records)} steps, {len(records[0]['layers'])} layers -> {out}")
    print(f"router load at step {BREAK_AT - 1}: {records[BREAK_AT - 1]['moe']['router']}")
    print(f"router load at step {first}: {records[first]['moe']['router']}")
    print(f"first step with one expert over half the tokens: {first} (seeded at {BREAK_AT})")
    print(f"evidence -> {evidence_path}")
    if args.rerun:
        rrd_path = tensorquill.export_rerun(
            evidence, Path("tensorquill-out") / "toy-moe.rrd", records=records
        )
        print(f"rerun recording -> {rrd_path}")


if __name__ == "__main__":
    main()
