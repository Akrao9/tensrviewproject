"""Two models, two optimizers and two sessions that both train in one loop.

A GAN steps its discriminator and then its generator on every iteration; an actor and a critic
may step the critic twice for each step of the actor. The other tests that open two sessions close
one before the other trains. Here both stay open and their steps alternate, and each session must
record its own model only: its own steps, the step count its own optimizer gives, its own
parameters' gradient norms and update sizes, a `clip_grad_norm_` call on its own parameters, and
the values logged to it. Closing one leaves the other recording.

The reference for "its own" is the same seeded loop run with one session only, on that model:
each two-session record must equal that run's record, apart from when it was written (`ts`,
`attempt`). The two models' parameter names differ, so a name from the wrong model would show.

The last test measures where one model's forward inside the other's update (a GAN's `D(G(z))`,
and the generator's forward that makes the fake batch) lands in that model's records, as
today's behaviour.

Only torch and pytest are needed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import evidence as ev

QUIET = dict(heartbeat_seconds=None, stack_seconds=None)
OPTIONS = dict(capture="full", weights=1, updates=True, **QUIET)
WHEN = ("ts", "attempt", "run_id")


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(4, 8)
        self.out = nn.Linear(8, 4)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.up(z)))


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(4, 8)
        self.score = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(torch.relu(self.body(x)))


def _total_norm(model: nn.Module) -> float:
    return float(torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)))


class Loop:
    """A seeded loop over a generator (or actor) `g` and a critic (or discriminator) `d`.

    Each iteration steps `d` `d_steps` times on real data against `g`'s detached output, then
    `g` once through `d`. `sessions` maps "g" and "d" to an open session or None; a session that
    is given logs its model's loss before each of its steps. What each model's gradient norm was
    just before its step, and each clip call's return, are kept for the assertions."""

    def __init__(self, d_steps: int = 1, clip: str | None = None) -> None:
        torch.manual_seed(0)
        self.g, self.d = Generator(), Critic()
        self.opt_g = torch.optim.AdamW(self.g.parameters(), lr=1e-3)
        self.opt_d = torch.optim.AdamW(self.d.parameters(), lr=1e-3)
        self.d_steps, self.clip = d_steps, clip
        self.data = torch.Generator().manual_seed(1)
        self.norms: dict[str, list[float]] = {"g": [], "d": []}
        self.clips: dict[str, list[float]] = {"g": [], "d": []}

    def _step(self, which: str, loss: torch.Tensor, session) -> None:
        model, optimizer = (self.g, self.opt_g) if which == "g" else (self.d, self.opt_d)
        if session is not None:
            session.log({f"{which}/loss": loss.detach()})
        loss.backward()
        if self.clip in (which, "both"):
            self.clips[which].append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)))
        self.norms[which].append(_total_norm(model))
        optimizer.step()

    def run(self, iterations: int, sessions: dict, *, close_d_after: int | None = None) -> None:
        bce = nn.BCEWithLogitsLoss()
        for it in range(iterations):
            if close_d_after is not None and it == close_d_after and sessions.get("d") is not None:
                sessions["d"].close()
                sessions = {**sessions, "d": None}
            for _ in range(self.d_steps):
                real = torch.randn(8, 4, generator=self.data) + 2
                z = torch.randn(8, 4, generator=self.data)
                self.opt_d.zero_grad()
                fake = self.g(z).detach()
                loss_d = bce(self.d(real), torch.ones(8, 1)) + bce(self.d(fake), torch.zeros(8, 1))
                self._step("d", loss_d, sessions.get("d"))
            z = torch.randn(8, 4, generator=self.data)
            self.opt_g.zero_grad()
            loss_g = bce(self.d(self.g(z)), torch.ones(8, 1))
            self._step("g", loss_g, sessions.get("g"))


def _run(open_: tuple[str, ...], iterations: int = 5, **loop) -> tuple[Loop, dict[str, list[dict]]]:
    close_d_after = loop.pop("close_d_after", None)
    state = Loop(**loop)
    sinks = {k: tensorquill.MemorySink() for k in open_}
    models = {"g": state.g, "d": state.d}
    sessions = {k: tensorquill.init(models[k], sink=sinks[k], run_id=f"two-{k}", **OPTIONS) for k in open_}
    state.run(iterations, dict(sessions), close_d_after=close_d_after)
    for session in sessions.values():
        session.close()
    return state, {k: s.records for k, s in sinks.items()}


def _when_left_out(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in WHEN} for r in records]


def _names(model: nn.Module) -> set[str]:
    return {n for n, _ in model.named_parameters()}


SHAPES = pytest.mark.parametrize("d_steps", [1, 2], ids=["gan_one_each", "critic_twice"])


@SHAPES
def test_each_record_holds_its_own_models_steps(d_steps):
    iterations = 5
    state, records = _run(("g", "d"), iterations, d_steps=d_steps)
    assert [r["step"] for r in records["g"]] == list(range(iterations))
    assert [r["step"] for r in records["d"]] == list(range(iterations * d_steps))
    for which, model in (("g", state.g), ("d", state.d)):
        names = _names(model)
        for record in records[which]:
            assert set(record["grad_norms"]) == names
            assert set(record["update_rms"]) == names
            assert set(record["weight_norms"]) == names
            assert set(record["scalars"]) == {f"{which}/loss"}
        assert [r["grad_norm"] for r in records[which]] == pytest.approx(state.norms[which], rel=1e-5)


@SHAPES
@pytest.mark.parametrize("which", ["g", "d"])
def test_a_second_open_session_changes_nothing_in_a_record(d_steps, which):
    """The same loop with only `which`'s session open gives the same records."""
    _, both = _run(("g", "d"), d_steps=d_steps)
    _, alone = _run((which,), d_steps=d_steps)
    assert _when_left_out(both[which]) == _when_left_out(alone[which])


@SHAPES
def test_a_clip_on_one_models_parameters_is_recorded_by_its_session_only(d_steps):
    state, records = _run(("g", "d"), d_steps=d_steps, clip="d")
    assert [r.get("clip_grad_norm") for r in records["d"]] == pytest.approx(state.clips["d"], rel=1e-6)
    assert all("clip_grad_norm" not in r for r in records["g"])
    assert all("clip_calls" not in r for side in records.values() for r in side)


@SHAPES
def test_clips_on_both_models_each_go_to_their_own_session(d_steps):
    state, records = _run(("g", "d"), d_steps=d_steps, clip="both")
    for which in ("g", "d"):
        assert [r.get("clip_grad_norm") for r in records[which]] == pytest.approx(state.clips[which], rel=1e-6)
        assert all("clip_calls" not in r for r in records[which])


@SHAPES
def test_closing_one_session_leaves_the_other_recording(d_steps):
    iterations, closed_at = 6, 3
    state, records = _run(("g", "d"), iterations, d_steps=d_steps, clip="both", close_d_after=closed_at)
    assert [r["step"] for r in records["d"]] == list(range(closed_at * d_steps))
    assert [r["step"] for r in records["g"]] == list(range(iterations))
    assert [r["grad_norm"] for r in records["g"]] == pytest.approx(state.norms["g"], rel=1e-5)
    assert [r.get("clip_grad_norm") for r in records["g"]] == pytest.approx(state.clips["g"], rel=1e-6)
    assert all(set(r["scalars"]) == {"g/loss"} for r in records["g"])
    # The records the generator's session wrote after the other closed equal those of a run in
    # which the other was never opened.
    _, alone = _run(("g",), iterations, d_steps=d_steps, clip="both")
    assert _when_left_out(records["g"]) == _when_left_out(alone["g"])


@SHAPES
def test_a_short_healthy_two_model_run_raises_nothing_on_either(d_steps):
    _, records = _run(("g", "d"), d_steps=d_steps)
    for which in ("g", "d"):
        assert ev.build_evidence(records[which], detectors="python")["incidents"] == [], which


# ---- a model's forward inside the other model's update ------------------------------------------

LOUD = 100.0


def _cross_forward_run(how: str, iterations: int):
    """A GAN loop in which each model's forward inside the other's update is made the largest:
    the generator's in making the fake batch (from `z * LOUD`) and the discriminator's in `D(G(z))`
    (on `G(z) * LOUD`). Hooks on `g.up` and `d.body` keep each forward's four numbers, as the
    capture computes them, labelled by whose update it ran in. `how` makes the fake batch with
    `.detach()` or under `torch.no_grad()`."""
    torch.manual_seed(0)
    g, d = Generator(), Critic()
    opt_g = torch.optim.AdamW(g.parameters(), lr=1e-3)
    opt_d = torch.optim.AdamW(d.parameters(), lr=1e-3)
    data = torch.Generator().manual_seed(1)
    seen: dict[str, list[tuple[str, int, dict]]] = {"g": [], "d": []}
    now: list = [None]

    def keep(which: str):
        def hook(module, args, output):
            x = output.detach().float()
            std, mean = torch.std_mean(x)
            seen[which].append((*now[0], {"mean": float(mean), "std": float(std), "abs_max": float(x.abs().amax()),
                                          "zero_frac": float((x == 0).float().mean())}))
        return hook

    g.up.register_forward_hook(keep("g"))
    d.body.register_forward_hook(keep("d"))
    sinks = {k: tensorquill.MemorySink() for k in "gd"}
    sessions = {k: tensorquill.init(m, sink=sinks[k], run_id=f"cross-{k}", **OPTIONS) for k, m in (("g", g), ("d", d))}
    bce = nn.BCEWithLogitsLoss()
    for it in range(iterations):
        real = torch.randn(8, 4, generator=data) + 2
        z = torch.randn(8, 4, generator=data)
        opt_d.zero_grad()
        now[0] = ("d", it)
        if how == "no_grad":
            with torch.no_grad():
                fake = g(z * LOUD)
        else:
            fake = g(z * LOUD).detach()
        loss_d = bce(d(real), torch.ones(8, 1)) + bce(d(fake / LOUD), torch.zeros(8, 1))
        sessions["d"].log({"d/loss": loss_d.detach()})
        loss_d.backward()
        opt_d.step()
        z = torch.randn(8, 4, generator=data)
        opt_g.zero_grad()
        now[0] = ("g", it)
        loss_g = bce(d(g(z) * LOUD), torch.ones(8, 1))
        sessions["g"].log({"g/loss": loss_g.detach()})
        loss_g.backward()
        opt_g.step()
    for session in sessions.values():
        session.close()
    return seen, {k: s.records for k, s in sinks.items()}


@pytest.mark.parametrize("how", ["detach", "no_grad"])
def test_a_models_forward_inside_the_other_models_update_raises_only_its_abs_max(how):
    """Today's behaviour, measured (Nov 26, 2026). A model's forward run inside the other model's
    update is a training forward of that model, and the capture keeps a block's largest magnitude
    over every forward until that model's own `optimizer.step()`, as it does for microbatches, while
    `mean`, `std` and `zero_frac` are those of the last forward before it. So:

    - the generator's forward in the discriminator's update (the fake batch) is in the
      generator's `abs_max` of the same iteration;
    - the discriminator's forward in the generator's update (`D(G(z))`) is in the discriminator's
      `abs_max` of its next record, and the last iteration's is in no record;
    - neither shows in the other three numbers."""
    iterations = 4
    seen, records = _cross_forward_run(how, iterations)
    assert [r["step"] for r in records["g"]] == [r["step"] for r in records["d"]] == list(range(iterations))

    def forwards(which: str, update: str, it: int) -> list[dict]:
        return [stats for who, at, stats in seen[which] if (who, at) == (update, it)]

    def same(record: dict, stats: dict, keys=("mean", "std", "zero_frac")) -> bool:
        return all(record[k] == pytest.approx(stats[k], rel=1e-6, abs=1e-12) for k in keys)

    for it, record in enumerate(records["g"]):
        [inside_d], [own] = forwards("g", "d", it), forwards("g", "g", it)
        block = record["layers"]["up"]
        assert inside_d["abs_max"] > 10 * own["abs_max"]
        assert block["abs_max"] == pytest.approx(inside_d["abs_max"], rel=1e-6)
        assert same(block, own)

    for it, record in enumerate(records["d"]):
        own = forwards("d", "d", it)
        assert len(own) == 2  # D(real), then D(fake)
        block = record["layers"]["body"]
        if it == 0:
            assert block["abs_max"] == pytest.approx(max(f["abs_max"] for f in own), rel=1e-6)
        else:
            [inside_g] = forwards("d", "g", it - 1)  # D(G(z)) of the iteration before, after the last step
            assert inside_g["abs_max"] > 5 * max(f["abs_max"] for f in own)
            assert block["abs_max"] == pytest.approx(inside_g["abs_max"], rel=1e-6)
        assert same(block, own[-1])
    [last] = forwards("d", "g", iterations - 1)
    assert all(r["layers"]["body"]["abs_max"] != pytest.approx(last["abs_max"], rel=1e-6) for r in records["d"])
