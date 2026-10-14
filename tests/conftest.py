"""A tiny fake PlantVillage directory so the full pipeline can run in CI without the dataset."""

import json
import random
from pathlib import Path

import pytest
from PIL import Image

CLASSES = ["Apple___Black_rot", "Apple___healthy", "Tomato___Early_blight"]


@pytest.fixture
def fake_data_dir(tmp_path: Path) -> Path:
    rng = random.Random(0)
    lines = {"train": [], "test": []}
    leaf_map = {}
    for c_idx, cls in enumerate(CLASSES):
        folder = tmp_path / "raw" / "color" / cls
        folder.mkdir(parents=True)
        for leaf in range(8):
            for shot in range(2):  # two photos of each leaf, linked by the leaf map
                photo = f"LAB_HL {c_idx}{leaf}{shot}"
                name = f"{leaf:04d}-{shot}___{photo}.JPG"
                leaf_map[photo.lower()] = [f"{cls}:::{leaf}.0"]
                color = (40 + 80 * c_idx + rng.randint(0, 20), 120, 60)
                Image.new("RGB", (48, 48), color).save(folder / name, format="JPEG")
                split = "test" if leaf >= 6 else "train"
                lines[split].append(f"raw/color/{cls}/{name}")
    (tmp_path / "splits").mkdir()
    for split, items in lines.items():
        (tmp_path / "splits" / f"{split}.txt").write_text("\n".join(items) + "\n")
    (tmp_path / "leaf-map.json").write_text(json.dumps(leaf_map))
    return tmp_path
