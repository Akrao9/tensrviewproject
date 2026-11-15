"""Check TensorQuill replica drift with two distributed PyTorch workers."""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import tensorquill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="use Gloo instead of CUDA/NCCL")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        parser.error("launch with torchrun --standalone --nproc-per-node=2")
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.cpu:
        device = torch.device("cpu")
        device_ids = None
    else:
        if torch.cuda.device_count() < 2:
            parser.error("two visible CUDA GPUs are required")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        device_ids = [local_rank]

    backend = "gloo" if args.cpu else "nccl"
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        model = torch.nn.Linear(4, 1, bias=False, device=device)
        with torch.no_grad():
            model.weight.copy_(torch.arange(1.0, 5.0))
        ddp = DDP(model, device_ids=device_ids)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0)
        sink = tensorquill.MemorySink()
        x = torch.ones(2, 4, device=device)

        with tensorquill.init(
            model, sink=sink, auto_step=False, replica_group=dist.group.WORLD, replica_every=1
        ) as session:
            for step in range(6):
                optimizer.zero_grad(set_to_none=True)
                ddp(x).sum().backward()
                optimizer.step()

                with torch.no_grad():
                    model.weight.copy_(torch.arange(1.0, 5.0))
                    if rank == 1 and step == 1:
                        model.weight[0, 0].add_(0.25)
                    if rank == 1 and step == 3:
                        model.weight[0, 0].neg_()
                    if rank == 1 and step == 5:
                        model.weight[0, [1, 2]] = model.weight[0, [2, 1]].clone()  # swap two elements
                session.step(step)

        records = sink.records
        assert [record["step"] for record in records] == list(range(6))
        for step in (0, 2, 4):
            assert records[step]["replica_spread"]["weight"] == [0.0, 0.0, 0.0]
        drift = tensorquill.detect_replica_drift(records)
        assert drift and drift["step"] == 1 and "weight" in drift["parameters"], drift
        sign_flip = tensorquill.detect_replica_drift(records[2:])
        assert sign_flip and sign_flip["step"] == 3, sign_flip
        spread = sign_flip["parameters"]["weight"]
        assert abs(spread["norm_spread"]) < 1e-6 and spread["sum_spread"] > 1e-6, spread
        swap = tensorquill.detect_replica_drift(records[4:])
        assert swap and swap["step"] == 5, swap
        spread = swap["parameters"]["weight"]
        # The older norm and sum fingerprint sees nothing here; only the projection does.
        assert spread["norm_spread"] < 1e-6 and spread["sum_spread"] < 1e-6, spread
        assert spread["projection_spread"] > 1e-6, spread
        import time

        timings = {}
        for projection in (False, True):
            tensorquill.parameter_fingerprints(model.parameters(), projection=projection)  # warm up
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                tensorquill.parameter_fingerprints(model.parameters(), projection=projection)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings[projection] = (time.perf_counter() - start) / 20 * 1000
        print(
            f"rank {rank} fingerprint time on {device.type}: {timings[False]:.3f} ms norm and sum,"
            f" {timings[True]:.3f} ms with the seeded projection (a tiny model: only the ratio and a GPU"
            " run mean anything)"
        )
        print(f"PASS {backend} rank {rank}: healthy at 0, 2 and 4, drift at 1, sign flip at 3, swap at 5")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
