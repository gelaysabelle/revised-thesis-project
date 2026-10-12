from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "watch-dataset" / "raw"
TRACKER_PATH = ROOT / "watch-dataset" / "tracker" / "Watch-Image-Tracker - Image_Log.csv"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


def normalize_image_id(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def iter_dataset_images() -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for image_path in sorted(RAW_ROOT.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        lookup[image_path.stem] = image_path
    return lookup


def detect_set(image_path: Path) -> str:
    rel = image_path.relative_to(RAW_ROOT).as_posix()
    if rel.startswith("pool/") or "/pool/" in rel:
        return "Pool"
    if rel.startswith("test/") or "/test/" in rel:
        return "Test"
    return ""


def compute_orig_short_side(image_path: Path) -> int:
    with Image.open(image_path) as image:
        width, height = image.size
    return min(width, height)


def compute_crop_short_side(image_path: Path, background_threshold: float = 35.0) -> int:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)

    if arr.size == 0:
        return min(rgb.size)

    border = np.concatenate(
        [
            arr[:10, :].reshape(-1, 3),
            arr[-10:, :].reshape(-1, 3),
            arr[:, :10].reshape(-1, 3),
            arr[:, -10:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_color = np.median(border, axis=0)
    diff = np.linalg.norm(arr - bg_color, axis=2)
    mask = diff > background_threshold

    if not np.any(mask):
        return min(rgb.size)

    rows, cols = np.where(mask)
    left, top, right, bottom = cols.min(), rows.min(), cols.max() + 1, rows.max() + 1
    crop = rgb.crop((left, top, right, bottom))
    return min(crop.size)


def compute_blur_pass(image_path: Path, threshold: float = 100.0) -> str:
    with Image.open(image_path) as image:
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        variance = float(np.asarray(edges, dtype=np.float32).var())

    return "Yes" if variance >= threshold else "No"


def compute_duplicate_flag(image_lookup: dict[str, Path]) -> dict[str, str]:
    hashes: dict[str, list[str]] = {}
    for image_id, image_path in image_lookup.items():
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(image_id)

    duplicate_ids = {
        image_id
        for image_ids in hashes.values()
        for image_id in image_ids
        if len(image_ids) > 1
    }

    flags = {image_id: ("Yes" if image_id in duplicate_ids else "No") for image_id in image_lookup}
    return flags


def main() -> None:
    tracker_path = TRACKER_PATH
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker CSV not found: {tracker_path}")

    df = pd.read_csv(tracker_path, on_bad_lines="skip")
    required_cols = {
        "image_id",
        "orig_short_side_px",
        "crop_short_side_px",
        "blur_pass",
        "duplicate_flag",
        "set",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Tracker CSV is missing columns: {missing}")

    image_lookup = iter_dataset_images()
    duplicate_flags = compute_duplicate_flag(image_lookup)

    for col in ["set", "blur_pass", "duplicate_flag"]:
        df[col] = df[col].astype("object")

    updated = 0
    for idx, row in df.iterrows():
        image_id = normalize_image_id(row.get("image_id"))
        if not image_id or image_id not in image_lookup:
            continue

        image_path = image_lookup[image_id]

        df.at[idx, "set"] = detect_set(image_path)
        df.at[idx, "orig_short_side_px"] = compute_orig_short_side(image_path)
        df.at[idx, "crop_short_side_px"] = compute_crop_short_side(image_path)
        df.at[idx, "blur_pass"] = compute_blur_pass(image_path)
        df.at[idx, "duplicate_flag"] = duplicate_flags[image_id]
        updated += 1

    df.to_csv(tracker_path, index=False)

    print(f"Updated {updated} rows in {tracker_path.name}")
    print(f"Pool rows: {(df['set'] == 'Pool').sum()}")
    print(f"Test rows: {(df['set'] == 'Test').sum()}")
    print(f"Blur pass rows: {(df['blur_pass'] == 'Yes').sum()}")
    print(f"Duplicate rows: {(df['duplicate_flag'] == 'Yes').sum()}")


if __name__ == "__main__":
    main()
