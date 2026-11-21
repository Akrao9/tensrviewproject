"""One training step of nanoMoE with TensorQuill watching each router's inner linear layer
(`w_g`), and a forward hook of our own on the Router module, whose first output is
`used_capacity`: the assignments each expert kept after the capacity limit. Prints both, so
what TensorQuill counts (the assignments the router asked for) can be set against what was
dispatched. Run from the work copy, with it and sdk/src on PYTHONPATH:  python ../check_router_hook.py
"""
import glob, json, math, shutil, tempfile, os
import torch
import tensorquill
from model import GPT, GPTConfig

torch.manual_seed(1337)
B, T, n_exp, top_k, capacity_factor = 16, 128, 8, 2, 1.25
config = GPTConfig(n_layer=6, n_head=6, n_embd=192, block_size=T, bias=False, vocab_size=65, dropout=0.0,
                   n_exp=n_exp, top_k=top_k, use_aux_loss=True, use_router_z_loss=True, use_noisy_top_k=False,
                   aux_loss_weight=0.01, router_z_loss_weight=0.001, train_capacity=capacity_factor,
                   eval_capacity=2.0, min_capacity=4, stride=2, use_switch_tfm_init=True,
                   switch_tfm_init_scale=1.0, router_use_full_prec=True)
model = GPT(config)
optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cpu")
kept = {}
for name, module in model.named_modules():
    if name.endswith(".router"):
        module.register_forward_hook(lambda m, a, out, name=name: kept.__setitem__(name + ".w_g", out[0].tolist()))
out = tempfile.mkdtemp()
cwd = os.getcwd(); os.chdir(out)
session = tensorquill.init(model, capture="routers", every=1, routers={
    name: tensorquill.Router(n_exp, top_k=top_k) for name, _ in model.named_modules() if name.endswith("router.w_g")})
x = torch.randint(65, (B, T)); y = torch.randint(65, (B, T))
_, loss = model(x, y)
loss.backward(); optimizer.step(); session.close()
record = json.loads(open(glob.glob("tensorquill-out/*.rank0.jsonl")[0]).readline())
os.chdir(cwd); shutil.rmtree(out)
capacity = math.floor(top_k * capacity_factor * B * T / n_exp)
print(f"tokens {B * T}, top_k {top_k}, capacity per expert {capacity}")
for name, asked in record["moe"].items():
    limited = [min(a, capacity) for a in asked]
    print(name)
    print("  TensorQuill counted (asked):", asked, "sum", sum(asked))
    print("  nanoMoE kept (used_capacity):", kept[name], "sum", sum(kept[name]))
    print("  asked, cut at capacity, equals kept:", limited == kept[name],
          "| assignments dropped at capacity:", sum(asked) - sum(kept[name]))
