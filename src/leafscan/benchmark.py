"""Measure model cost: parameters, FLOPs, checkpoint size, CPU latency and GPU throughput.

    python -m leafscan.benchmark models/*/best.pt

Writes reports/<run>/benchmark.json next to the test metrics. Run it on an idle machine:
latency numbers are meaningless while something else (e.g. a training) uses the CPU/GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from leafscan.model import get_device, load_checkpoint


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: torch.nn.Module, img_size: int) -> int:
    """FLOPs of one forward pass on a single image (multiply-add counted as 2 FLOPs)."""
    x = torch.randn(1, 3, img_size, img_size, device=next(model.parameters()).device)
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(x)
    return counter.get_total_flops()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def time_forward(model, x: torch.Tensor, warmup: int = 5, runs: int = 30) -> float:
    """Median wall-clock time (seconds) of one forward pass."""
    for _ in range(warmup):
        model(x)
    _sync(x.device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        _sync(x.device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def benchmark(ckpt_path: Path, runs: int = 30, gpu_batch: int = 64) -> dict:
    cpu = torch.device("cpu")
    model, ckpt = load_checkpoint(ckpt_path, cpu)
    size = ckpt["img_size"]
    result = {
        "model": ckpt["model_name"],
        "params_m": count_params(model) / 1e6,
        "gflops": count_flops(model, size) / 1e9,
        "checkpoint_mb": ckpt_path.stat().st_size / 2**20,
        "cpu_latency_ms": 1000 * time_forward(model, torch.randn(1, 3, size, size), runs=runs),
        "cpu": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
    }
    device = get_device()
    if device.type != "cpu":
        model.to(device)
        x = torch.randn(gpu_batch, 3, size, size, device=device)
        result["gpu_device"] = device.type
        result["gpu_throughput_img_s"] = gpu_batch / time_forward(model, x, runs=max(5, runs // 3))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", type=Path, nargs="+")
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.add_argument("--runs", type=int, default=30)
    args = p.parse_args()

    for ckpt in args.checkpoints:
        res = benchmark(ckpt, runs=args.runs)
        out = args.reports_dir / ckpt.parent.name / "benchmark.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"{ckpt.parent.name:16s} {res['params_m']:6.1f}M  {res['gflops']:5.2f} GFLOPs  "
              f"CPU {res['cpu_latency_ms']:6.1f} ms  "
              f"GPU {res.get('gpu_throughput_img_s', float('nan')):7.0f} img/s")


if __name__ == "__main__":
    main()
