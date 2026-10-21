import numpy as np
from PIL import Image

from leafscan.background import (
    FILL,
    background_only,
    background_only_box,
    eligible,
    evaluate,
    leaf_only,
    pick_donors,
    swap,
)
from leafscan.data import read_split, segmented_path
from leafscan.model import create_model, save_checkpoint


def _img_and_mask(value):
    img = np.full((20, 20, 3), value, np.uint8)
    mask = np.zeros((20, 20), bool)
    mask[5:10, 5:10] = True
    return img, mask


def test_leaf_only_and_background_only():
    img, mask = _img_and_mask(200)
    lo = leaf_only(img, mask)
    assert (lo[mask] == 200).all() and (lo[~mask] == FILL).all()
    bo = background_only(img, mask, dilate=2)
    assert (bo[3:12, 3:12] == FILL).all()  # leaf + 2 px margin is greyed out
    assert (bo[0, 0] == 200).all()
    mask[5, 9] = False  # notch: the box version must hide it anyway
    box = background_only_box(img, mask, dilate=1)
    assert (box[4:11, 4:11] == FILL).all()
    assert (box[0, 0] == 200).all() and (box[12, 12] == 200).all()


def test_swap_keeps_leaf_and_takes_donor_background():
    img, mask = _img_and_mask(200)
    donor, donor_mask = _img_and_mask(50)
    donor_mask[:] = False
    donor_mask[12:15, 12:15] = True
    out = swap(img, mask, donor, donor_mask, dilate=0)
    assert (out[mask] == 200).all()  # our leaf
    assert (out[0, 0] == 50).all()  # donor background
    assert (out[13, 13] == FILL).all()  # donor's own leaf is greyed out


def test_pick_donors_always_from_another_class(fake_data_dir):
    test = read_split(fake_data_dir, "test")
    items = [(s, None) for s in test]
    donors = pick_donors(items, seed=1)
    assert all(items[j][0].label != s.label for (s, _), j in zip(items, donors, strict=True))


def test_evaluate_end_to_end(fake_data_dir, tmp_path):
    test = read_split(fake_data_dir, "test")
    for s in test:  # segmented twins: leaf = left half on black
        seg = np.zeros((48, 48, 3), np.uint8)
        seg[:, :24] = (40, 160, 40)
        segmented_path(s.path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(seg).save(segmented_path(s.path), format="PNG")
    assert len(eligible(test)) == len(test)

    classes = sorted({s.label for s in test})
    ckpt = tmp_path / "run" / "best.pt"
    save_checkpoint(ckpt, create_model("resnet18", len(classes), pretrained=False), "resnet18",
                    classes, img_size=32)
    res = evaluate(ckpt, test, num_workers=0)
    for cond in ("original", "leaf_only", "background_only", "background_only_box"):
        r = res[cond]
        assert r["n"] == len(test) and 0 <= r["accuracy"] <= 1 and 0 <= r["balanced_accuracy"] <= 1
        assert sum(r["per_class_n"].values()) == len(test)
        assert 1 / len(classes) <= r["most_predicted_share"] <= 1
    sw = res["swap"]
    assert abs(sw["follows_leaf"] + sw["follows_background"] + sw["neither"] - 1) < 1e-9
