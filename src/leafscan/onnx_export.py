"""Export to ONNX for in-browser inference, optionally int8-quantised, and check it.

    python -m leafscan.onnx_export models/leafscan_field/best.pt hub/leafscan-convnext-tiny

Writes model.onnx (fp32) and model_int8.onnx (dynamic int8 quantisation of the weights) next
to config.json, and reports their agreement with PyTorch on PlantDoc test photos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from leafscan.model import load_checkpoint


def export_onnx(ckpt_path: Path, out_path: Path, opset: int = 17) -> None:
    model, ckpt = load_checkpoint(ckpt_path, torch.device("cpu"))
    x = torch.randn(1, 3, ckpt["img_size"], ckpt["img_size"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (x,), str(out_path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, dynamo=False)


def constants_to_initializers(src: Path, dst: Path) -> None:
    """Move Constant-node tensors into graph initializers.

    The TorchScript exporter emits some weights as Constant nodes; the quantiser only
    quantises weights stored as initializers, so without this step nothing is quantised.
    """
    import onnx

    m = onnx.load(str(src))
    kept = []
    for node in m.graph.node:
        value = next((a for a in node.attribute if a.name == "value"), None)
        if node.op_type == "Constant" and value is not None and value.t.dims:
            tensor = onnx.TensorProto()
            tensor.CopyFrom(value.t)
            tensor.name = node.output[0]
            m.graph.initializer.append(tensor)
        else:
            kept.append(node)
    del m.graph.node[:]
    m.graph.node.extend(kept)
    onnx.save(m, str(dst))


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """Dynamic int8 quantisation of the linear layers' weights (activations stay fp32).

    Convolutions stay fp32: ONNX Runtime Web has no ConvInteger kernel. In ConvNeXt most
    weights are in the pointwise MLPs (MatMul) anyway.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp = int8_path.with_suffix(".prep.onnx")
    constants_to_initializers(fp32_path, tmp)
    quantize_dynamic(str(tmp), str(int8_path), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    tmp.unlink()


def onnx_logits(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([sess.run(["logits"], {"image": x[i:i + 32]})[0]
                           for i in range(0, len(x), 32)])


def compare(ckpt_path: Path, onnx_paths: list[Path], images: np.ndarray, labels: np.ndarray,
            temperature: float, threshold: float) -> dict:
    """Agreement of each ONNX file with PyTorch: top-1, max |logit diff|, accuracy, abstention."""
    from leafscan.reliability import softmax

    model, _ = load_checkpoint(ckpt_path, torch.device("cpu"))
    with torch.no_grad():
        ref = model(torch.from_numpy(images)).numpy()

    def summary(logits):
        p = softmax(logits, temperature)
        answered = p.max(1) >= threshold
        return {"accuracy": float((p.argmax(1) == labels).mean()),
                "answered": float(answered.mean()),
                "accuracy_when_answering": float((p.argmax(1) == labels)[answered].mean())}

    out = {"pytorch": summary(ref)}
    for path in onnx_paths:
        logits = onnx_logits(path, images)
        out[path.name] = {**summary(logits),
                          "size_mb": path.stat().st_size / 2**20,
                          "top1_agreement_with_pytorch": float((logits.argmax(1)
                                                                == ref.argmax(1)).mean()),
                          "max_abs_logit_diff": float(np.abs(logits - ref).max())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    fp32, int8 = args.out_dir / "model.onnx", args.out_dir / "model_int8.onnx"
    export_onnx(args.checkpoint, fp32)
    quantize_int8(fp32, int8)

    from PIL import Image

    from leafscan.data import build_transforms
    from leafscan.field import load_plantdoc

    _, ckpt = load_checkpoint(args.checkpoint, torch.device("cpu"))
    tf = build_transforms(ckpt["img_size"], train=False)
    test = [i for i in load_plantdoc()[0] if i["split"] == "test"]
    images = np.stack([tf(Image.open(i["path"]).convert("RGB")).numpy() for i in test])
    labels = np.array([ckpt["classes"].index(i["label"]) for i in test])
    res = compare(args.checkpoint, [fp32, int8], images, labels,
                  ckpt["temperature"], ckpt["threshold"])

    # Quantisation changes some predictions, so the int8 model gets its own temperature and
    # threshold, fitted on the same field calibration split as the PyTorch model.
    from leafscan.deploy import TARGET_ACCURACY, selective, split_calibration
    from leafscan.reliability import fit_temperature, lab_threshold, softmax

    train = [i for i in load_plantdoc()[0] if i["split"] == "train"]
    _, calib = split_calibration(train, 0.15, seed=0)
    cx = np.stack([tf(Image.open(i["path"]).convert("RGB")).numpy() for i in calib])
    cy = np.array([ckpt["classes"].index(i["label"]) for i in calib])
    cl = onnx_logits(int8, cx)
    t = fit_temperature(cl, cy)
    pc = softmax(cl, t)
    thr = lab_threshold(pc.max(1), pc.argmax(1) == cy, TARGET_ACCURACY)
    pt = softmax(onnx_logits(int8, images), t)
    res["model_int8.onnx (recalibrated)"] = {"temperature": t, "threshold": thr,
                                             "calibration_set": selective(pc, cy, thr),
                                             "plantdoc_test": selective(pt, labels, thr)}
    config_path = args.out_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["onnx_int8"] = {"file": int8.name, "temperature": t, "threshold": thr}
    config_path.write_text(json.dumps(config, indent=2))
    report = Path("reports") / args.checkpoint.parent.name / "onnx.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
