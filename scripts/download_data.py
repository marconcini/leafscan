"""Download PlantVillage (color images + official splits) into data/plantvillage/."""

import argparse
from pathlib import Path

from leafscan.data import DEFAULT_DATA_DIR, download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    download(args.data_dir)
