"""Probe and head coverage: what TensorQuill's probe pass covered, against counts taken from the model.

    python examples/gpu_probe_coverage.py --out runs/probe-coverage                         # SmolLM2-360M, BF16
    python examples/gpu_probe_coverage.py --out runs/probe-coverage-granite --model ibm-granite/granite-3.0-1b-a400m-base
    python examples/gpu_probe_coverage.py --out runs/probe-coverage-toy --toy                # CPU, tiny random Llama

Trains `--steps` steps with full capture and the probe pass every `--probe-every` steps on one fixed
batch, then checks every probe pass in the capture against counts the script takes itself, before
TensorQuill is attached:

- from the model's own module tree and config: the decoder blocks (the `ModuleList` as long as
  `num_hidden_layers`), `num_attention_heads`, and the language-model head's output size;
- an attention entry for every decoder layer, so a model that never calls SDPA fails;
- from running the model once on the same fixed batch with the script's own counter around
  `F.scaled_dot_product_attention`: how many attention calls a forward makes, and each call's
  heads, query length and key length.

Each probe pass must hold a histogram for every block; one attention entry per counted call, each
with one value per head (`max_logit`, `entropy`) and `rows` = batch x heads x query length; and
the head's numbers over every position of the batch (`positions` + `nonfinite_positions` +
`overflow_positions` = batch x sequence length) with `vocab` the head's output size. It writes
`coverage.json` with the counts of both sides and the coverage (heads with a finite largest logit,
attention rows neither masked nor invalid, head positions read) and exits 0 when every check holds, 1
when one does not. A CPU run with `--toy` checks the harness on a toy, not the GPU claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

import tensorquill

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_outlier_check import MODEL, build_models, load_batches  # noqa: E402


def build(args, device):
    """One model, in training mode: the toy Llama of dense_outlier_check, or the named model in BF16 with SDPA."""
    if args.toy:
        models, _ = build_models(SimpleNamespace(toy=True, model=None, lr=1e-5), device)
        return models[0]
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    return AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).train()


def blocks_from_tree(model: torch.nn.Module) -> list[str]:
    """The decoder blocks, from the module tree: the children of the ModuleList as long as num_hidden_layers."""
    layers = model.config.num_hidden_layers
    found = [name for name, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and len(m) == layers]
    if len(found) != 1:
        raise SystemExit(f"cannot tell the decoder blocks: ModuleLists of length {layers}: {found}")
    return [f"{found[0]}.{i}" for i in range(layers)]


def head_size(model: torch.nn.Module) -> int | None:
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    return int(head.out_features) if isinstance(head, torch.nn.Linear) else None


def count_attention_calls(model: torch.nn.Module, batch: torch.Tensor) -> list[dict[str, int]]:
    """Run the model once on `batch` (eval, no gradients) and count its SDPA calls, with each call's shapes."""
    calls: list[dict[str, int]] = []
    original = F.scaled_dot_product_attention

    def counting(query, key, *args, **kwargs):
        calls.append({"heads": int(query.shape[-3]), "query_len": int(query.shape[-2]), "key_len": int(key.shape[-2]),
                      "batch": int(max(query.shape[0], key.shape[0]))})
        return original(query, key, *args, **kwargs)

    training = model.training
    F.scaled_dot_product_attention = counting
    try:
        model.eval()
        with torch.no_grad():
            model(input_ids=batch, use_cache=False)
    finally:
        F.scaled_dot_product_attention = original
        model.train(training)
    return calls


def check_probe(step: int, probe: dict[str, Any], expected: dict[str, Any]) -> tuple[list[dict], dict]:
    """One probe pass against the expected counts: (checks, coverage)."""
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"step": step, "name": name, "ok": bool(ok), "detail": detail})

    if "error" in probe:
        check("the probe pass ran", False, probe["error"])
        return checks, {}
    hist = probe.get("histograms") or {}
    missing = [b for b in expected["blocks"] if b not in hist]
    check("a histogram for every decoder block", not missing and len(hist) == len(expected["blocks"]),
          f"{len(hist)} histograms, {len(expected['blocks'])} blocks" + (f"; missing {missing[:3]}" if missing else ""))
    attention = probe.get("attention") or {}
    calls = expected["attention_calls"]
    check("one attention entry per attention call of a forward", len(attention) == len(calls),
          f"{len(attention)} entries, {len(calls)} calls counted, {expected['layers']} layers")
    check("attention entries for every decoder layer", len(attention) == expected["layers"],
          f"{len(attention)} entries, {expected['layers']} layers"
          + ("; the model made no SDPA call, so nothing can be read" if not calls else ""))
    errors = {k: v["error"] for k, v in attention.items() if "error" in v}
    check("no attention entry failed", not errors, str(list(errors.items())[:2]))
    heads_total = heads_finite = rows_total = rows_masked = rows_invalid = 0
    wrong: list[str] = []
    for (name, entry), call in zip(attention.items(), calls):
        if "error" in entry:
            continue
        heads = call["heads"]
        if heads != expected["heads"]:
            wrong.append(f"{name}: the call has {heads} heads, the config {expected['heads']}")
        for key in ("max_logit", "entropy", "entropy_share"):
            if len(entry.get(key) or []) != heads:
                wrong.append(f"{name}: {key} has {len(entry.get(key) or [])} values for {heads} heads")
        want_rows = call["batch"] * heads * call["query_len"]
        if entry.get("rows") != want_rows or entry.get("query_len") != call["query_len"] or entry.get("key_len") != call["key_len"]:
            wrong.append(f"{name}: rows {entry.get('rows')}, query {entry.get('query_len')}, keys {entry.get('key_len')}; "
                         f"counted {want_rows}, {call['query_len']}, {call['key_len']}")
        heads_total += heads
        heads_finite += sum(1 for v in entry.get("max_logit") or [] if isinstance(v, (int, float)) and math.isfinite(v))
        rows_total += entry.get("rows", 0)
        rows_masked += entry.get("masked_rows", 0)
        rows_invalid += entry.get("invalid_rows", 0)
    check("every attention entry has one value per head and the counted rows", not wrong, "; ".join(wrong[:3]) or
          f"{len(attention)} entries x {expected['heads']} heads")
    head = probe.get("head") or {}
    positions = expected["positions"]
    read = head.get("positions", 0)
    seen = read + head.get("nonfinite_positions", 0) + head.get("overflow_positions", 0)
    check("the head's numbers cover every position of the probe batch", "not_read" not in head and seen == positions,
          f"{head.get('not_read') or ''}{read} read + {head.get('nonfinite_positions', 0)} non-finite + "
          f"{head.get('overflow_positions', 0)} overflowed = {seen}, batch x sequence {positions}")
    check("the head's vocabulary is the head's output size", head.get("vocab") == expected["vocab"],
          f"{head.get('vocab')} against {expected['vocab']}")
    coverage = {
        "blocks_with_histogram": f"{len(hist)}/{len(expected['blocks'])}",
        "attention_layers": f"{len(attention)}/{expected['layers']}",
        "heads_with_finite_max_logit": f"{heads_finite}/{expected['layers'] * expected['heads']}",
        "attention_rows_valid": f"{rows_total - rows_masked - rows_invalid}/{rows_total}",
        "head_positions_read": f"{read}/{positions}",
    }
    return checks, coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the capture and coverage.json")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--probe-every", type=int, default=50)
    parser.add_argument("--toy", action="store_true", help="tiny random Llama on random tokens, CPU allowed")
    args = parser.parse_args(argv)
    if args.probe_every < 1 or args.steps < args.probe_every:
        parser.error("--steps must reach at least one probe pass")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.toy and device.type != "cuda":
        parser.error("a CUDA GPU is required; use --toy to check the script on CPU")
    args.out.mkdir(parents=True)

    model = build(args, device)
    if args.toy:
        batches = torch.randint(0, model.config.vocab_size, (args.steps, 2, 32),
                                generator=torch.Generator().manual_seed(0)).to(device)
    else:
        batches = load_batches(args.model, args.batch_size, args.seq_len, args.steps, device)
    fixed = batches[0]
    config = model.config
    expected = {
        "blocks": blocks_from_tree(model), "layers": int(config.num_hidden_layers),
        "heads": int(config.num_attention_heads), "vocab": head_size(model),
        "positions": int(fixed.shape[0] * fixed.shape[1]), "attention_calls": count_attention_calls(model, fixed),
    }
    name = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
    print(f"{torch.__version__} on {name} | {'toy Llama' if args.toy else args.model} | {expected['layers']} blocks, "
          f"{expected['heads']} heads, {len(expected['attention_calls'])} attention calls per forward, "
          f"probe batch {tuple(fixed.shape)}", flush=True)

    jsonl = args.out / "coverage.jsonl"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=device.type == "cuda")
    with tensorquill.init(model, run_id="probe-coverage", capture="full", every=50, sink=tensorquill.JsonlSink(jsonl),
                          probe=lambda m: m(input_ids=fixed, use_cache=False), probe_every=args.probe_every):
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            model(input_ids=batch, labels=batch, use_cache=False).loss.backward()
            optimizer.step()

    probes = [(r["step"], r["probe"]) for r in tensorquill.load_records(jsonl) if "probe" in r]
    want_passes = args.steps // args.probe_every
    checks = [{"step": None, "name": "one probe pass per probe interval", "ok": len(probes) == want_passes,
               "detail": f"{len(probes)} passes at steps {[s for s, _ in probes]}, expected {want_passes}"}]
    coverage = {}
    for step, probe in probes:
        found, coverage[step] = check_probe(step, probe, expected)
        checks += found
    passed = all(c["ok"] for c in checks)
    result = {"status": "pass" if passed else "fail", "toy": args.toy, "model": "toy Llama" if args.toy else args.model,
              "device": name, "expected": {**expected, "attention_calls": len(expected["attention_calls"]),
                                           "attention_call_shapes": expected["attention_calls"][:1]},
              "coverage_by_probe_step": coverage, "checks": checks, "capture": str(jsonl),
              "claim": "a toy dry run: checks the harness, not the GPU claim" if args.toy else "this model on this GPU"}
    (args.out / "coverage.json").write_text(json.dumps(result, indent=1, default=str))
    for step, cov in coverage.items():
        print(f"Probe at step {step}: " + ", ".join(f"{k} {v}" for k, v in cov.items()))
    for c in checks:
        if not c["ok"]:
            print(f"FAIL (step {c['step']}): {c['name']}: {c['detail']}")
    print(f"{'PASS' if passed else 'FAIL'}: {sum(c['ok'] for c in checks)} of {len(checks)} checks; "
          f"result {args.out / 'coverage.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
