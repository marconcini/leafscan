import torch

from leafscan.benchmark import benchmark, count_flops, count_params
from leafscan.model import create_model, save_checkpoint


def test_count_params_and_flops_on_a_known_layer():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, kernel_size=1, bias=False))
    assert count_params(model) == 3 * 8
    # 1x1 conv on 4x4 image: 16 positions x 8 outputs x 3 inputs multiply-adds = 384 MACs
    assert count_flops(model, img_size=4) == 2 * 384


def test_benchmark_reports_all_fields(tmp_path):
    ckpt = tmp_path / "run" / "best.pt"
    save_checkpoint(ckpt, create_model("resnet18", 3, pretrained=False), "resnet18",
                    ["a", "b", "c"], img_size=32)
    res = benchmark(ckpt, runs=2)
    assert res["model"] == "resnet18"
    assert 11 < res["params_m"] < 12  # resnet18 ~11.2M params
    assert res["gflops"] > 0 and res["cpu_latency_ms"] > 0 and res["checkpoint_mb"] > 0
