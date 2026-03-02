#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)


def build_train_rows(data_root: Path):
    train_dir = data_root / "train"
    rows = []
    for cls_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        cls_name = cls_dir.name
        for img_path in iter_images(cls_dir):
            rows.append((img_path.relative_to(data_root).as_posix(), cls_name))
    return rows


def build_test_rows(data_root: Path):
    support_dir = data_root / "test" / "support"
    query_dir = data_root / "test" / "query"

    rows = []

    for cls_dir in sorted(p for p in support_dir.iterdir() if p.is_dir()):
        cls_name = cls_dir.name
        for img_path in iter_images(cls_dir):
            rows.append((img_path.relative_to(data_root).as_posix(), cls_name))

    for cls_dir in sorted(p for p in query_dir.iterdir() if p.is_dir()):
        cls_name = cls_dir.name
        for img_path in iter_images(cls_dir):
            rows.append((img_path.relative_to(data_root).as_posix(), cls_name))

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate LibFewShot CSV files for the MSTAR train/support-query layout."
    )
    parser.add_argument("data_root", type=Path, help="Dataset root containing train/ and test/")
    args = parser.parse_args()

    data_root = args.data_root.resolve()

    train_rows = build_train_rows(data_root)
    test_rows = build_test_rows(data_root)

    write_csv(train_rows, data_root / "train.csv")
    write_csv(test_rows, data_root / "test.csv")
    write_csv(test_rows, data_root / "val.csv")

    print(f"Generated train.csv with {len(train_rows)} samples")
    print(f"Generated test.csv/val.csv with {len(test_rows)} samples")


if __name__ == "__main__":
    main()
