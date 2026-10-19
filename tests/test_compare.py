import json

from leafscan.compare import error_overlap, load_runs, plot, to_markdown


def _write_run(reports, run, acc, params, with_bench=True):
    d = reports / run
    d.mkdir(parents=True)
    (d / "test_metrics.json").write_text(json.dumps(
        {"accuracy": acc, "macro_f1": acc - 0.01, "errors": {"n_errors": 3}}))
    if with_bench:
        (d / "benchmark.json").write_text(json.dumps(
            {"params_m": params, "gflops": 1.0, "checkpoint_mb": 10.0, "cpu_latency_ms": 20.0}))


def test_load_runs_skips_unbenchmarked_and_sorts_by_size(tmp_path):
    _write_run(tmp_path, "resnet50", 0.99, 25.6)
    _write_run(tmp_path, "efficientnet_b0", 0.98, 5.3)
    _write_run(tmp_path, "vit_small", 0.97, 22.0, with_bench=False)
    rows = load_runs(tmp_path)
    assert [r["run"] for r in rows] == ["efficientnet_b0", "resnet50"]
    assert rows[0]["name"] == "EfficientNet-B0"


def test_markdown_bolds_best_accuracy_and_plot_is_written(tmp_path):
    _write_run(tmp_path, "resnet50", 0.99, 25.6)
    _write_run(tmp_path, "efficientnet_b0", 0.98, 5.3)
    rows = load_runs(tmp_path)
    table = to_markdown(rows)
    assert "**99.00%**" in table and "**98.00%**" not in table
    assert table.count("\n") == 4  # header, separator, 2 rows
    plot(rows, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0


def test_error_overlap(tmp_path):
    for run, paths in {"a": ["x", "y"], "b": ["y", "z"], "c": ["y"]}.items():
        (tmp_path / run).mkdir()
        (tmp_path / run / "test_errors.json").write_text(json.dumps([{"path": p} for p in paths]))
    ov = error_overlap(tmp_path, ["a", "b", "c"])
    assert ov["wrong_for_all"] == ["y"]
    assert ov["n_wrong_for_any"] == 3
    assert ov["jaccard"]["a / b"] == 1 / 3
