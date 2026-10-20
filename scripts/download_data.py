"""Download PlantVillage (color images + official splits) into data/plantvillage/."""

import argparse
from pathlib import Path

from leafscan.data import DEFAULT_DATA_DIR, download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--segmented", action="store_true",
                        help="also extract background-removed images (leaf masks for Grad-CAM)")
    args = parser.parse_args()
    download(args.data_dir, segmented=args.segmented)
