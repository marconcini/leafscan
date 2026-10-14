"""PlantVillage data: download, leak-free splits, PyTorch dataset and transforms."""

from __future__ import annotations

import json
import random
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

HF_REPO = "mohanty/PlantVillage"
VARIANT = "color"
DEFAULT_DATA_DIR = Path("data/plantvillage")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- download


def download(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download the PlantVillage archive from the HF Hub and extract the color images.

    Only `raw/color/` is extracted (~1/3 of the 2.2 GB archive). The official
    train/test split files are copied next to the images.
    """
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir)
    images_dir = data_dir / "raw" / VARIANT
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        src = hf_hub_download(HF_REPO, f"splits/{VARIANT}_{split}.txt", repo_type="dataset")
        shutil.copy(src, splits_dir / f"{split}.txt")
    leaf_map = hf_hub_download(HF_REPO, "leaf_grouping/leaf-map.json", repo_type="dataset")
    shutil.copy(leaf_map, data_dir / "leaf-map.json")

    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"Images already present in {images_dir}, skipping extraction.")
        return data_dir

    archive = hf_hub_download(HF_REPO, "data.zip", repo_type="dataset")
    prefix = f"raw/{VARIANT}/"
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if m.startswith(prefix) and not m.endswith("/")]
        print(f"Extracting {len(members)} images to {data_dir} ...")
        for m in members:
            zf.extract(m, data_dir)
    return data_dir


# --------------------------------------------------------------------------- splits


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def class_from_path(rel_path: str) -> str:
    """'raw/color/Tomato___Early_blight/xxx.JPG' -> 'Tomato___Early_blight'."""
    return Path(rel_path).parent.name


def photo_id(path: Path | str) -> str:
    """'<uuid>___RS_HL 9155.JPG' -> 'rs_hl 9155' (identifier of the photo in the original lab)."""
    stem = Path(path).stem.split("___", 1)[-1]
    return re.sub(r"(_final_masked)$", "", stem, flags=re.IGNORECASE).strip().lower()


LeafMap = dict[str, list[str]]


def load_leaf_map(data_dir: Path) -> LeafMap:
    """Photo id -> candidate leaf ids ('Class:::n'), from PlantVillage `leaf-map.json`.

    A photo id can appear in several classes, so the class must be used to disambiguate.
    """
    path = Path(data_dir) / "leaf-map.json"
    return json.loads(path.read_text()) if path.exists() else {}


def leaf_group(path: Path | str, leaf_map: LeafMap | None = None) -> str:
    """Identifier of the physical leaf an image was taken from.

    PlantVillage photographed each leaf several times (~4 photos per leaf). Images
    of the same leaf must stay in the same split, otherwise validation scores are
    inflated. The official leaf map covers ~76% of images; for the rest we fall back
    to the photo id (i.e. no grouping).
    """
    pid, label = photo_id(path), Path(path).parent.name
    for leaf in (leaf_map or {}).get(pid, []):
        if leaf.split(":::", 1)[0] == label:
            return leaf
    return f"{label}/{pid}"


def read_split(data_dir: Path, split: str) -> list[Sample]:
    data_dir = Path(data_dir)
    lines = (data_dir / "splits" / f"{split}.txt").read_text().splitlines()
    return [Sample(data_dir / line, class_from_path(line)) for line in lines if line.strip()]


def group_split(
    samples: list[Sample], val_fraction: float, seed: int = 42,
    leaf_map: LeafMap | None = None,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into train/val so that no leaf appears in both, stratified by class."""
    rng = random.Random(seed)
    by_class: dict[str, dict[str, list[Sample]]] = {}
    for s in samples:
        by_class.setdefault(s.label, {}).setdefault(leaf_group(s.path, leaf_map), []).append(s)

    train, val = [], []
    for label in sorted(by_class):
        groups = sorted(by_class[label])
        rng.shuffle(groups)
        n_val = max(1, round(len(groups) * val_fraction))
        for i, g in enumerate(groups):
            (val if i < n_val else train).extend(by_class[label][g])
    return train, val


def load_splits(
    data_dir: Path = DEFAULT_DATA_DIR, val_fraction: float = 0.1, seed: int = 42
) -> tuple[list[Sample], list[Sample], list[Sample], list[str]]:
    """Return (train, val, test, class_names). Test is the official PlantVillage test split."""
    train_full = read_split(data_dir, "train")
    test = read_split(data_dir, "test")
    train, val = group_split(train_full, val_fraction, seed, load_leaf_map(data_dir))
    classes = sorted({s.label for s in train_full})
    return train, val, test, classes


def subsample(samples: list[Sample], per_class: int, seed: int = 42) -> list[Sample]:
    """Keep at most `per_class` samples per class (for quick smoke runs)."""
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    counts: Counter[str] = Counter()
    out = []
    for s in shuffled:
        if counts[s.label] < per_class:
            counts[s.label] += 1
            out.append(s)
    return out


# --------------------------------------------------------------------------- torch


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PlantVillageDataset(Dataset):
    def __init__(self, samples: list[Sample], classes: list[str], transform=None):
        self.samples = samples
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[s.label]


def pretty_label(label: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato — Early blight'."""
    crop, _, disease = label.partition("___")
    crop = crop.replace("_", " ").replace(",", "").strip()
    disease = disease.replace("_", " ").strip()
    return f"{crop} — {disease}"
