import numpy as np
from PIL import Image

from leafscan.data import read_split, segmented_path
from leafscan.explain import figure, gradcam_metrics
from leafscan.model import create_model, save_checkpoint


def _add_segmented_twins(samples, leaf_fraction_cols):
    """Black background, 'leaf' = the first columns of the image."""
    for s in samples:
        img = np.zeros((48, 48, 3), np.uint8)
        img[:, :leaf_fraction_cols] = (40, 160, 40)
        seg = segmented_path(s.path)
        seg.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(seg, format="JPEG")


def test_gradcam_metrics_and_figure(fake_data_dir, tmp_path):
    test = read_split(fake_data_dir, "test")
    _add_segmented_twins(test[:-2], leaf_fraction_cols=24)  # last 2 images have no mask
    classes = sorted({s.label for s in test})
    ckpt = tmp_path / "run" / "best.pt"
    save_checkpoint(ckpt, create_model("resnet18", len(classes), pretrained=False), "resnet18",
                    classes, img_size=32)

    res = gradcam_metrics(ckpt, test, batch_size=4, num_workers=0)
    assert res["excluded_no_mask"] == 2
    a = res["all"]
    assert a["n"] + a["n_empty_cam"] == len(test) - 2
    if a["n"]:  # an untrained model may give all-zero CAMs, which are counted apart
        assert 0.4 < a["leaf_area"] < 0.6  # half the image, after resize + crop
        assert 0 <= a["leaf_focus"] <= 1
    n = lambda k: res[k]["n"] + res[k].get("n_empty_cam", 0)  # noqa: E731
    assert n("correct") + n("wrong") == n("all")

    out = tmp_path / "fig.jpg"
    figure([ckpt], [test[0].path, test[-1].path], out)
    assert out.stat().st_size > 0
