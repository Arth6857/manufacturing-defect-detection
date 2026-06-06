"""
MVTec Dataset Preparation Script
─────────────────────────────────
Downloads and restructures the MVTec AD dataset into the
binary (good / defective) split expected by train.py.

MVTec AD has 15 categories. This script:
  1. Optionally downloads a single category (e.g. 'bottle')
  2. Merges all anomaly sub-types into one 'defective' folder
  3. Creates train / val / test splits

Usage:
    python utils/prepare_data.py --category bottle
    python utils/prepare_data.py --category all   # process every category
"""

import os
import shutil
import random
import argparse
from pathlib import Path

# MVTec categories
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42
random.seed(SEED)


def restructure_category(
    mvtec_root: str,
    category: str,
    output_root: str,
):
    """
    MVTec raw structure:
        <category>/
            train/good/
            test/good/
            test/<defect_type_1>/
            test/<defect_type_2>/
            ...

    Output structure:
        data/mvtec/<category>/
            train/good/  train/defective/
            val/good/    val/defective/
            test/good/   test/defective/
    """
    cat_root   = Path(mvtec_root) / category
    out_root   = Path(output_root) / category

    if not cat_root.exists():
        print(f"  ✗ {category} not found at {cat_root} — skipping.")
        return

    print(f"\n  Processing: {category}")

    # ── Collect good images (from both train & test splits)
    good_images = []
    for split in ["train", "test"]:
        good_dir = cat_root / split / "good"
        if good_dir.exists():
            good_images += list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg"))

    # ── Collect defective images (all anomaly sub-types)
    defective_images = []
    test_root = cat_root / "test"
    for sub in test_root.iterdir():
        if sub.is_dir() and sub.name != "good":
            defective_images += list(sub.glob("*.png")) + list(sub.glob("*.jpg"))

    print(f"    Good images      : {len(good_images)}")
    print(f"    Defective images : {len(defective_images)}")

    # ── Split each class
    def split_list(lst):
        random.shuffle(lst)
        n = len(lst)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    g_train, g_val, g_test = split_list(good_images)
    d_train, d_val, d_test = split_list(defective_images)

    splits = {
        "train": {"good": g_train, "defective": d_train},
        "val":   {"good": g_val,   "defective": d_val},
        "test":  {"good": g_test,  "defective": d_test},
    }

    # ── Copy files
    for split_name, classes in splits.items():
        for cls_name, files in classes.items():
            dest_dir = out_root / split_name / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                dst = dest_dir / src.name
                # Handle duplicate filenames
                if dst.exists():
                   dst = dest_dir / f"{src.parent.name}_{src.name}"
                shutil.copy(src, dst)
                os.chmod(dst, 0o644)

    # ── Summary
    for split_name, classes in splits.items():
        for cls_name, files in classes.items():
            print(f"    {split_name:5s}/{cls_name:10s}: {len(files)} images")


def main():
    parser = argparse.ArgumentParser(description="Prepare MVTec dataset")
    parser.add_argument(
        "--mvtec_root", type=str, default="data/mvtec_raw",
        help="Path to raw MVTec dataset root"
    )
    parser.add_argument(
        "--output_root", type=str, default="data/mvtec",
        help="Output path for restructured dataset"
    )
    parser.add_argument(
        "--category", type=str, default="bottle",
        help=f"Category to process. One of {CATEGORIES} or 'all'"
    )
    args = parser.parse_args()

    categories = CATEGORIES if args.category == "all" else [args.category]

    print("=" * 55)
    print("  MVTec Dataset Preparation")
    print("=" * 55)
    print(f"  Source  : {args.mvtec_root}")
    print(f"  Output  : {args.output_root}")
    print(f"  Categories: {categories}")

    for cat in categories:
        restructure_category(args.mvtec_root, cat, args.output_root)

    print("\n✓ Dataset preparation complete.")
    print(f"  Set DATA_DIR = '{args.output_root}/<category>' in train.py")


if __name__ == "__main__":
    main()
