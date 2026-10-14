from pathlib import Path

import torch

from leafscan.data import (
    PlantVillageDataset,
    build_transforms,
    group_split,
    leaf_group,
    load_leaf_map,
    load_splits,
    pretty_label,
    subsample,
)


def test_leaf_group_uses_leaf_map_and_falls_back_to_photo_id():
    leaf_map = {
        "rs_hl 9155": ["Potato___healthy:::3.0", "Tomato___healthy:::7.0"],  # shared id
        "rs_hl 9160": ["Tomato___healthy:::7.0"],
    }
    a = leaf_group(Path("raw/color/Tomato___healthy/aaa-111___RS_HL 9155.JPG"), leaf_map)
    b = leaf_group(Path("raw/color/Tomato___healthy/bbb-222___RS_HL 9160.JPG"), leaf_map)
    c = leaf_group(Path("raw/color/Tomato___healthy/ccc-333___RS_HL 9156.JPG"), leaf_map)
    assert a == b == "Tomato___healthy:::7.0"
    assert c == "Tomato___healthy/rs_hl 9156"


def test_load_splits_counts_and_classes(fake_data_dir):
    train, val, test, classes = load_splits(fake_data_dir, val_fraction=0.2)
    assert classes == sorted(classes) and len(classes) == 3
    assert len(train) + len(val) == 3 * 6 * 2
    assert len(test) == 3 * 2 * 2


def test_group_split_has_no_leaf_leakage(fake_data_dir):
    train, val, _, _ = load_splits(fake_data_dir, val_fraction=0.3)
    leaf_map = load_leaf_map(fake_data_dir)
    train_leaves = {leaf_group(s.path, leaf_map) for s in train}
    val_leaves = {leaf_group(s.path, leaf_map) for s in val}
    assert val_leaves and not train_leaves & val_leaves
    assert len(val) == 2 * len(val_leaves)  # both photos of each val leaf stay together


def test_group_split_is_deterministic(fake_data_dir):
    train, _, _, _ = load_splits(fake_data_dir)
    a, _ = group_split(train, 0.2, seed=1)
    b, _ = group_split(train, 0.2, seed=1)
    assert a == b


def test_subsample_caps_each_class(fake_data_dir):
    train, _, _, _ = load_splits(fake_data_dir)
    small = subsample(train, per_class=3)
    assert len(small) == 9


def test_dataset_returns_normalized_tensor(fake_data_dir):
    train, _, _, classes = load_splits(fake_data_dir)
    ds = PlantVillageDataset(train, classes, build_transforms(32, train=True))
    x, y = ds[0]
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32
    assert 0 <= y < len(classes)


def test_pretty_label():
    assert pretty_label("Tomato___Early_blight") == "Tomato — Early blight"
    assert pretty_label("Corn_(maize)___healthy") == "Corn (maize) — healthy"
