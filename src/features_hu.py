"""
Extract 7 Hu moments from binary leaf masks.

Input:
    data/processed/masks/*_mask.png
    data/processed/preprocess_summary.csv

Output:
    data/processed/features/hu_moments_features.csv
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def extract_hu_moments(mask_path: Path) -> np.ndarray | None:
    """Compute the 7 log-scaled Hu moments from a binary leaf mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[read failed] {mask_path}")
        return None

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(binary, binaryImage=True)
    if moments["m00"] == 0:
        print(f"[empty mask] {mask_path}")
        return None

    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu_log.astype(np.float64)


def load_summary(processed_dir: Path) -> pd.DataFrame:
    """Load metadata produced by preprocessing."""
    df = pd.read_csv(processed_dir / "preprocess_summary.csv")
    df["image_id"] = df["image_id"].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 7 Hu moments from Flavia leaf masks.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(r"C:\AIA_workspace\data\processed"),
        help="Directory produced by preprocessing.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: processed/features/hu_moments_features.csv",
    )
    args = parser.parse_args()

    processed_dir: Path = args.processed_dir
    mask_dir = processed_dir / "masks"
    features_dir = processed_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    output_csv = args.output_csv or (features_dir / "hu_moments_features.csv")

    mask_paths = sorted(mask_dir.glob("*_mask.png"))
    print("Mask directory:", mask_dir)
    print("Mask count:", len(mask_paths))

    if not mask_paths:
        raise FileNotFoundError(f"No mask files found: {mask_dir}")

    summary_lookup = {
        str(row["image_id"]): row.to_dict()
        for _, row in load_summary(processed_dir).iterrows()
    }

    records: list[dict[str, object]] = []
    failed_files: list[str] = []

    for i, mask_path in enumerate(mask_paths, start=1):
        print(f"[{i}/{len(mask_paths)}] Extracting Hu moments: {mask_path.name}")

        hu = extract_hu_moments(mask_path)
        if hu is None:
            failed_files.append(str(mask_path))
            continue

        image_id = mask_path.stem.replace("_mask", "")
        metadata = summary_lookup[image_id]

        record: dict[str, object] = {
            "filename": metadata["filename"],
            "image_id": image_id,
            "label": metadata["label"],
            "scientific_name": metadata["scientific_name"],
            "common_name": metadata["common_name"],
        }

        record.update({f"hu_{j}": value for j, value in enumerate(hu, start=1)})

        records.append(record)

    pd.DataFrame(records).to_csv(output_csv, index=False)

    failed_path = features_dir / "hu_failed_files.txt"
    failed_path.write_text("".join(f"{item}\n" for item in failed_files), encoding="utf-8")

    print("\nHu moments extraction finished")
    print("Success:", len(records))
    print("Failed:", len(failed_files))
    print("Output CSV:", output_csv)
    print("Failed file list:", failed_path)


if __name__ == "__main__":
    main()
