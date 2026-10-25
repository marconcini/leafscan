import numpy as np
import pytest
from PIL import Image

from leafscan.field import (
    PLANTDOC_TO_PLANTVILLAGE,
    crop_of,
    evaluate,
    field_metrics,
    is_healthy,
    load_plantdoc,
)
from leafscan.model import create_model, save_checkpoint

CLASSES = ["Apple___Black_rot", "Tomato___Early_blight"]


def test_mapping_targets_real_plantvillage_classes():
    # the 38 PlantVillage class names, as in the official splits
    pv = {"Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
          "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
          "Cherry_(including_sour)___healthy",
          "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
          "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
          "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
          "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
          "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
          "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
          "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
          "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
          "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
          "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
          "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
          "Tomato___Tomato_mosaic_virus", "Tomato___healthy"}
    assert len(pv) == 38
    assert set(PLANTDOC_TO_PLANTVILLAGE.values()) <= pv
    assert len(PLANTDOC_TO_PLANTVILLAGE) == 28


def test_crop_and_health():
    assert crop_of("Tomato___Late_blight") == "Tomato"
    assert is_healthy("Apple___healthy") and not is_healthy("Apple___Apple_scab")


def test_field_metrics_levels():
    classes = ["Apple___Apple_scab", "Apple___healthy", "Tomato___Late_blight"]
    y = np.array([0, 0, 1, 2])
    probs = np.eye(3)[[0, 1, 1, 1]] * 0.9 + 0.1 / 3  # pred: scab, healthy, healthy, healthy
    m = field_metrics(y, probs, classes)
    assert m["accuracy"] == 0.5
    assert m["crop_accuracy"] == 0.75  # only the tomato -> apple error changes crop
    assert m["health_accuracy"] == 0.5  # scab->healthy and blight->healthy are health errors
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1 + 0) / 3)


def test_load_and_evaluate(tmp_path):
    for i, (split, folder) in enumerate([("train", "Apple Scab Leaf"), ("test", "Apple leaf"),
                                         ("test", "Tomato leaf late blight"),
                                         ("train", "Tomato leaf late blight")]):
        d = tmp_path / split / folder
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 40), (90, 140, 20 * i)).save(d / f"{i}.jpg")
    (tmp_path / "test" / "Apple leaf" / "broken.jpg").write_bytes(b"not an image")
    # same image, same class (other split) -> kept once
    late = tmp_path / "test" / "Tomato leaf late blight" / "2.jpg"
    (tmp_path / "train" / "Tomato leaf late blight" / "copy.jpg").write_bytes(late.read_bytes())
    # same image filed under two classes -> both dropped
    scab = (tmp_path / "train" / "Apple Scab Leaf" / "0.jpg").read_bytes()
    (tmp_path / "test" / "Apple leaf" / "conflict.jpg").write_bytes(scab)
    items, skipped = load_plantdoc(tmp_path)
    reasons = sorted(s.split(" (")[1].split(":")[0].split(" of")[0].rstrip(")") for s in skipped)
    assert reasons == ["conflicting labels", "conflicting labels", "duplicate", "unreadable"]
    assert len(items) == 3  # Apple leaf/1, Tomato late blight/2 (once), Tomato late blight/3
    assert {it["label"] for it in items} == {"Apple___healthy", "Tomato___Late_blight"}

    classes = sorted({it["label"] for it in items} | set(CLASSES))
    ckpt = tmp_path / "run" / "best.pt"
    save_checkpoint(ckpt, create_model("resnet18", len(classes), pretrained=False), "resnet18",
                    classes, img_size=32)
    res = evaluate(ckpt, items, num_workers=0)
    assert res["all"]["n"] == 3 and res["plantdoc_test_split_only"]["n"] == 2


def test_unmapped_folder_is_an_error(tmp_path):
    (tmp_path / "train" / "Mango leaf").mkdir(parents=True)
    (tmp_path / "test").mkdir()
    with pytest.raises(KeyError):
        load_plantdoc(tmp_path)
