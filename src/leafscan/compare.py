"""Gather test metrics and benchmarks of every run into a comparison table and plot.

    python -m leafscan.compare            # -> reports/comparison.md, reports/comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-T",
    "vit_small": "ViT-S/16",
}


def load_runs(reports_dir: Path) -> list[dict]:
    """One row per run that has both test metrics and a benchmark."""
    rows = []
    for metrics_path in sorted(reports_dir.glob("*/test_metrics.json")):
        bench_path = metrics_path.with_name("benchmark.json")
        if not bench_path.exists():
            continue
        m = json.loads(metrics_path.read_text())
        b = json.loads(bench_path.read_text())
        run = metrics_path.parent.name
        rows.append({
            "run": run,
            "name": DISPLAY_NAMES.get(run, run),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "n_errors": m["errors"]["n_errors"] if m.get("errors") else None,
            **{k: b[k] for k in ("params_m", "gflops", "checkpoint_mb", "cpu_latency_ms")},
            "gpu_throughput_img_s": b.get("gpu_throughput_img_s"),
        })
    return sorted(rows, key=lambda r: r["params_m"])


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| Model | Params | GFLOPs | Size | Accuracy | Macro-F1 | Errors "
        "| CPU latency | GPU throughput |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    best_acc = max(r["accuracy"] for r in rows)
    for r in rows:
        acc = f"{100 * r['accuracy']:.2f}%"
        acc = f"**{acc}**" if r["accuracy"] == best_acc else acc
        gpu = f"{r['gpu_throughput_img_s']:.0f} img/s" if r["gpu_throughput_img_s"] else "–"
        lines.append(
            f"| {r['name']} | {r['params_m']:.1f}M | {r['gflops']:.1f} "
            f"| {r['checkpoint_mb']:.0f} MB "
            f"| {acc} | {100 * r['macro_f1']:.2f}% | {r['n_errors']} "
            f"| {r['cpu_latency_ms']:.0f} ms | {gpu} |"
        )
    return "\n".join(lines) + "\n"


def error_overlap(reports_dir: Path, runs: list[str]) -> dict:
    """Do models fail on the same images? Shared errors point at the data, not the model."""
    errors = {r: {e["path"] for e in json.loads((reports_dir / r / "test_errors.json").read_text())}
              for r in runs if (reports_dir / r / "test_errors.json").exists()}
    if len(errors) < 2:
        return {}
    names = list(errors)
    jaccard = {
        f"{a} / {b}": len(errors[a] & errors[b]) / max(1, len(errors[a] | errors[b]))
        for i, a in enumerate(names) for b in names[i + 1:]
    }
    common = set.intersection(*errors.values())
    return {
        "n_errors": {r: len(e) for r, e in errors.items()},
        "jaccard": jaccard,
        "n_wrong_for_all": len(common),
        "n_wrong_for_any": len(set.union(*errors.values())),
        "wrong_for_all": sorted(common),
    }


def plot(rows: list[dict], path: Path) -> None:
    """Accuracy vs CPU latency: the trade-off that matters for a CPU-hosted demo."""
    ink, muted, grid, mark = "#1f2328", "#59636e", "#e6e8eb", "#2f7d4f"
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in rows:
        ax.scatter(r["cpu_latency_ms"], 100 * r["accuracy"], s=70, color=mark,
                   edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(f"{r['name']}  ({r['params_m']:.0f}M)",
                    (r["cpu_latency_ms"], 100 * r["accuracy"]),
                    xytext=(8, 6), textcoords="offset points", fontsize=9, color=ink)
    ax.set_xlabel("CPU latency, 1 image (ms) — lower is better", color=muted)
    ax.set_ylabel("Test accuracy (%)", color=muted)
    ax.set_title("Accuracy vs. inference cost on PlantVillage", color=ink, loc="left")
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted)
    xmax = max(r["cpu_latency_ms"] for r in rows)
    ax.set_xlim(0, xmax * 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = p.parse_args()

    rows = load_runs(args.reports_dir)
    if not rows:
        raise SystemExit("No run has both test_metrics.json and benchmark.json.")
    table = to_markdown(rows)
    (args.reports_dir / "comparison.md").write_text(table)
    overlap = error_overlap(args.reports_dir, [r["run"] for r in rows])
    (args.reports_dir / "error_overlap.json").write_text(json.dumps(overlap, indent=2))
    plot(rows, args.reports_dir / "comparison.png")
    print(table)
    if overlap:
        print(f"Wrong for all models: {overlap['n_wrong_for_all']} / "
              f"wrong for at least one: {overlap['n_wrong_for_any']}")


if __name__ == "__main__":
    main()
