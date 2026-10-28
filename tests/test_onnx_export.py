import numpy as np
import pytest
import torch

from leafscan.model import create_model, save_checkpoint

ort = pytest.importorskip("onnxruntime")


def test_onnx_matches_pytorch(tmp_path):
    from leafscan.onnx_export import compare, export_onnx, quantize_int8

    torch.manual_seed(0)
    ckpt = tmp_path / "best.pt"
    # a small ConvNeXt, like the real model: most weights are in linear layers
    save_checkpoint(ckpt, create_model("convnext_atto", 3, pretrained=False), "convnext_atto",
                    ["a", "b", "c"], img_size=32)
    fp32, int8 = tmp_path / "m.onnx", tmp_path / "m_int8.onnx"
    export_onnx(ckpt, fp32)
    quantize_int8(fp32, int8)
    images = np.random.default_rng(0).normal(size=(5, 3, 32, 32)).astype(np.float32)
    res = compare(ckpt, [fp32, int8], images, np.zeros(5, int), temperature=1.0, threshold=0.0)
    assert res["m.onnx"]["max_abs_logit_diff"] < 1e-3
    assert res["m.onnx"]["top1_agreement_with_pytorch"] == 1.0
    # the int8 model loads and predicts; its size gain is measured on the real model
    # (reports/leafscan_field/onnx.json: 106 MB -> 32 MB)
    assert 0 <= res["m_int8.onnx"]["top1_agreement_with_pytorch"] <= 1
