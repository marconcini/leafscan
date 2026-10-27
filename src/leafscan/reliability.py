"""Can the model say "I don't know"? Calibration, abstention and out-of-distribution detection.

    python -m leafscan.reliability models/convnext_tiny/best.pt [--device cpu]
    python -m leafscan.reliability --figure models/convnext_tiny/best.pt

Logits are computed once per data source and cached in models/<run>/logits/.
Sources: PlantVillage validation / test, PlantDoc (field), Oxford Flowers-102 test (near-OOD)
and DTD test (far-OOD textures), the last two via torchvision into data/ood/.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from leafscan.data import DEFAULT_DATA_DIR, build_transforms, load_splits
from leafscan.field import load_plantdoc
from leafscan.model import get_device, load_checkpoint

OOD_DIR = Path("data/ood")
MAX_OOD_IMAGES = 2000
LAB_TARGET_ACCURACY = 0.999


# --------------------------------------------------------------------------- data sources


def source_items(name: str, classes: list[str], data_dir: Path = DEFAULT_DATA_DIR
                 ) -> list[tuple[Path, int]]:
    """(image path, class index) pairs; class index is -1 for out-of-distribution images."""
    c2i = {c: i for i, c in enumerate(classes)}
    if name in ("pv_val", "pv_test"):
        _, val, test, _ = load_splits(data_dir)
        return [(s.path, c2i[s.label]) for s in (val if name == "pv_val" else test)]
    if name == "plantdoc":
        items, _ = load_plantdoc()
        return [(it["path"], c2i[it["label"]]) for it in items]
    if name in ("flowers", "dtd"):
        from torchvision.datasets import DTD, Flowers102

        ds = (Flowers102 if name == "flowers" else DTD)(OOD_DIR, split="test", download=False)
        files = list(ds._image_files)  # torchvision keeps the file list here
        random.Random(0).shuffle(files)
        return [(Path(f), -1) for f in sorted(files[:MAX_OOD_IMAGES])]
    raise ValueError(f"unknown source {name!r}")


class PathDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


@torch.no_grad()
def cached_logits(ckpt_path: Path, source: str, device: torch.device | None = None,
                  num_workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    cache = ckpt_path.parent / "logits" / f"{source}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["logits"], d["labels"]
    device = device or get_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    items = source_items(source, ckpt["classes"])
    loader = DataLoader(PathDataset(items, build_transforms(ckpt["img_size"], train=False)),
                        batch_size=64, num_workers=num_workers)
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    logits, labels = torch.cat(logits).numpy(), torch.cat(labels).numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, logits=logits, labels=labels)
    return logits, labels


# --------------------------------------------------------------------------- metrics


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return F.softmax(torch.from_numpy(logits / temperature), dim=1).numpy()


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: mean |accuracy - confidence| over confidence bins."""
    conf, correct = probs.max(1), probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimising the NLL on held-out data [Guo et al. 2017]."""
    log_t = torch.zeros(1, requires_grad=True)
    x, y = torch.from_numpy(logits).float(), torch.from_numpy(labels).long()
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy when answering only the k most confident inputs, for k = 1..n."""
    order = np.argsort(-conf, kind="stable")
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    coverage = np.arange(1, len(conf) + 1) / len(conf)
    return coverage, cum_acc


def accuracy_at_coverage(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    k = max(1, int(round(coverage * len(conf))))
    return float(correct[np.argsort(-conf, kind="stable")[:k]].mean())


def lab_threshold(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest confidence threshold whose selective accuracy reaches `target`."""
    order = np.argsort(-conf, kind="stable")
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.flatnonzero(cum_acc >= target)
    return float(conf[order][ok[-1]]) if len(ok) else 1.0


def energy(logits: np.ndarray) -> np.ndarray:
    """Negative free energy (logsumexp of logits): higher = more in-distribution."""
    return torch.logsumexp(torch.from_numpy(logits), dim=1).numpy()


def ood_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    return float(roc_auc_score(y, np.r_[id_scores, ood_scores]))


# --------------------------------------------------------------------------- analysis


def analyse(ckpt_path: Path, device: torch.device | None = None) -> dict:
    L = {s: cached_logits(ckpt_path, s, device)
         for s in ("pv_val", "pv_test", "plantdoc", "flowers", "dtd")}
    t = fit_temperature(*L["pv_val"])
    out = {"temperature_fitted_on_pv_val": t}

    for src in ("pv_test", "plantdoc"):
        logits, y = L[src]
        p1, pt = softmax(logits), softmax(logits, t)
        conf, correct = p1.max(1), p1.argmax(1) == y
        out[src] = {
            "n": len(y),
            "accuracy": float(correct.mean()),
            "mean_confidence": float(conf.mean()),
            "ece": ece(p1, y),
            "ece_after_temperature": ece(pt, y),
            "accuracy_at_coverage": {str(c): accuracy_at_coverage(conf, correct, c)
                                     for c in (1.0, 0.75, 0.5, 0.25, 0.1)},
        }

    vlog, vy = L["pv_val"]
    vp = softmax(vlog)
    thr = lab_threshold(vp.max(1), vp.argmax(1) == vy, LAB_TARGET_ACCURACY)
    out["lab_threshold"] = {"target_lab_accuracy": LAB_TARGET_ACCURACY, "threshold": thr}
    for src in ("pv_test", "plantdoc"):
        logits, y = L[src]
        p = softmax(logits)
        answered = p.max(1) >= thr
        out["lab_threshold"][src] = {
            "coverage": float(answered.mean()),
            "accuracy_when_answering": float((p.argmax(1) == y)[answered].mean())
            if answered.any() else None,
        }

    out["ood_auroc"] = {}
    for id_src in ("pv_test", "plantdoc"):
        for ood_src in ("flowers", "dtd", "plantdoc"):
            if ood_src == id_src:
                continue
            for score_name, fn in (("msp", lambda z: softmax(z).max(1)), ("energy", energy)):
                key = f"{id_src} vs {ood_src} ({score_name})"
                out["ood_auroc"][key] = ood_auroc(fn(L[id_src][0]), fn(L[ood_src][0]))
    out["mean_confidence_ood"] = {s: float(softmax(L[s][0]).max(1).mean())
                                  for s in ("flowers", "dtd")}
    return out


def figure(ckpt_path: Path, out: Path) -> None:
    """Reliability diagrams (lab vs field) and risk-coverage curves for one model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2328", "#59636e", "#e6e8eb"
    colors = {"pv_test": "#2f7d4f", "plantdoc": "#c2571a"}
    names = {"pv_test": "PlantVillage test (lab)", "plantdoc": "PlantDoc (field)"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    edges = np.linspace(0, 1, 11)
    for src in ("pv_test", "plantdoc"):
        logits, y = cached_logits(ckpt_path, src)
        p = softmax(logits)
        conf, correct = p.max(1), p.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 10:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        a1.plot(xs, ys, marker="o", color=colors[src], linewidth=2, markersize=7,
                markeredgecolor="white", markeredgewidth=2, label=names[src])
        cov, acc = risk_coverage(conf, correct)
        keep = np.arange(1, len(cov) + 1) >= 50  # tiny subsets give meaningless 0% / 100%
        a2.plot(100 * cov[keep], 100 * acc[keep], color=colors[src], linewidth=2,
                label=names[src])
    a1.plot([0, 1], [0, 1], color=muted, linestyle="--", linewidth=1)
    a1.text(0.62, 0.55, "perfect calibration", color=muted, fontsize=8, rotation=38)
    a1.set_xlabel("confidence", color=muted)
    a1.set_ylabel("accuracy", color=muted)
    a1.set_title("Reliability: does confidence mean accuracy?", color=ink, loc="left", fontsize=10)
    a2.set_xlabel("photos answered (%), most confident first (≥ 50 photos)", color=muted)
    a2.set_ylabel("accuracy on answered photos (%)", color=muted)
    a2.set_title("Abstention: answer only when confident", color=ink, loc="left", fontsize=10)
    a2.invert_xaxis()
    for ax in (a1, a2):
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right" if ax is a1 else "center left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--device", help="e.g. cpu (default: cuda > mps > cpu)")
    ap.add_argument("--figure", action="store_true", help="draw reports/reliability.png")
    args = ap.parse_args()

    if args.figure:
        figure(args.checkpoint, args.reports_dir / "reliability.png")
        return
    res = analyse(args.checkpoint, torch.device(args.device) if args.device else None)
    out = args.reports_dir / args.checkpoint.parent.name / "reliability.json"
    out.write_text(json.dumps(res, indent=2))
    pv, pd = res["pv_test"], res["plantdoc"]
    lt = res["lab_threshold"]["plantdoc"]
    print(f"{args.checkpoint.parent.name}: T={res['temperature_fitted_on_pv_val']:.2f} | "
          f"ECE lab {pv['ece']:.3f}->{pv['ece_after_temperature']:.3f} "
          f"field {pd['ece']:.3f}->{pd['ece_after_temperature']:.3f} | "
          f"field acc@50% {pd['accuracy_at_coverage']['0.5']:.3f} | lab thr on field: "
          f"cov {lt['coverage']:.3f} acc {lt['accuracy_when_answering']} | "
          f"AUROC lab vs flowers {res['ood_auroc']['pv_test vs flowers (msp)']:.3f} "
          f"dtd {res['ood_auroc']['pv_test vs dtd (msp)']:.3f}")


if __name__ == "__main__":
    main()
