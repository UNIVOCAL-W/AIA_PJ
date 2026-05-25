"""
features_hu.py

Extract the 7 Hu moments from binary leaf silhouettes produced by preprocessing.py.

Expected input:
    C:/AIA_workspace/data/processed/masks/*_mask.png
    C:/AIA_workspace/data/processed/preprocess_summary.csv  optional, but recommended

Output:
    C:/AIA_workspace/data/processed/features/hu_moments_features.csv

Run:
    python features_hu.py
or:
    python features_hu.py --processed-dir C:/AIA_workspace/data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


# -------------------------
# Flavia label fallback
# -------------------------

FLAVIA_RANGES: list[tuple[int, int, int, str, str]] = [
    (1001, 1059, 1, "Phyllostachys edulis (Carr.) Houz.", "pubescent bamboo"),
    (1060, 1122, 2, "Aesculus chinensis", "Chinese horse chestnut"),
    (1552, 1616, 3, "Berberis anhweiensis Ahrendt", "Anhui Barberry"),
    (1123, 1194, 4, "Cercis chinensis", "Chinese redbud"),
    (1195, 1267, 5, "Indigofera tinctoria L.", "true indigo"),
    (1268, 1323, 6, "Acer Palmatum", "Japanese maple"),
    (1324, 1385, 7, "Phoebe nanmu (Oliv.) Gamble", "Nanmu"),
    (1386, 1437, 8, "Kalopanax septemlobus (Thunb. ex A. Murr.) Koidz.", "castor aralia"),
    (1497, 1551, 9, "Cinnamomum japonicum Sieb.", "Chinese cinnamon"),
    (1438, 1496, 10, "Koelreuteria paniculata Laxm.", "goldenrain tree"),
    (2001, 2050, 11, "Ilex macrocarpa Olv.", "Big-fruited Holly"),
    (2051, 2113, 12, "Pittosporum tobira (Thunb.) Ait. f.", "Japanese cheesewood"),
    # Label 13 is absent in the original Flavia table used here.
    (2114, 2165, 14, "Chimonanthus praecox L.", "wintersweet"),
    (2166, 2230, 15, "Cinnamomum camphora (L.) J. Presl", "camphortree"),
    (2231, 2290, 16, "Viburnum awabuki K.Koch", "Japan Arrowwood"),
    (2291, 2346, 17, "Osmanthus fragrans Lour.", "sweet osmanthus"),
    (2347, 2423, 18, "Cedrus deodara (Roxb.) G. Don", "deodar"),
    (2424, 2485, 19, "Ginkgo biloba L.", "ginkgo, maidenhair tree"),
    (2486, 2546, 20, "Lagerstroemia indica (L.) Pers.", "Crape myrtle, Crepe myrtle"),
    (2547, 2612, 21, "Nerium oleander L.", "oleander"),
    (2616, 2675, 22, "Podocarpus macrophyllus (Thunb.) Sweet", "yew plum pine"),
    (3001, 3055, 23, "Prunus serrulata Lindl. var. lannesiana auct.", "Japanese Flowering Cherry"),
    (3056, 3110, 24, "Ligustrum lucidum Ait. f.", "Glossy Privet"),
    (3111, 3175, 25, "Toona sinensis M. Roem.", "Chinese Toon"),
    (3176, 3229, 26, "Prunus persica (L.) Batsch", "peach"),
    (3230, 3281, 27, "Manglietia fordiana Oliv.", "Ford Woodlotus"),
    (3282, 3334, 28, "Acer buergerianum Miq.", "trident maple"),
    (3335, 3389, 29, "Mahonia bealei (Fortune) Carr.", "Beale's barberry"),
    (3390, 3446, 30, "Magnolia grandiflora L.", "southern magnolia"),
    (3447, 3510, 31, "Populus × canadensis Moench", "Canadian poplar"),
    (3511, 3563, 32, "Liriodendron chinense (Hemsl.) Sarg.", "Chinese tulip tree"),
    (3566, 3621, 33, "Citrus reticulata Blanco", "tangerine"),
]


def flavia_metadata_from_image_id(image_id: str | int) -> dict[str, Any]:
    """Return label/species metadata from the Flavia filename number."""
    try:
        idx = int(image_id)
    except ValueError:
        return {"label": "unknown", "scientific_name": "unknown", "common_name": "unknown"}

    for start, end, label, scientific_name, common_name in FLAVIA_RANGES:
        if start <= idx <= end:
            return {
                "label": label,
                "scientific_name": scientific_name,
                "common_name": common_name,
            }
    return {"label": "unknown", "scientific_name": "unknown", "common_name": "unknown"}


# -------------------------
# Hu moments extraction
# -------------------------


def extract_hu_moments(mask_path: Path) -> np.ndarray | None:
    """
    Compute the 7 log-scaled Hu moments from a binary leaf mask.

    The mask should contain leaf region as non-zero pixels and background as zero.
    cv2.HuMoments returns values with very different magnitudes, so we use the
    common signed log transform:
        -sign(h) * log10(abs(h) + eps)
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[读取失败] {mask_path}")
        return None

    # Ensure binary image: leaf = 255, background = 0.
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # binaryImage=True makes all non-zero pixels count as 1 instead of 255.
    moments = cv2.moments(binary, binaryImage=True)
    if moments["m00"] == 0:
        print(f"[空 mask] {mask_path}")
        return None

    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu_log.astype(np.float64)


def load_summary(processed_dir: Path) -> pd.DataFrame:
    """Load preprocess_summary.csv if available; otherwise return an empty table."""
    summary_path = processed_dir / "preprocess_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(summary_path)
    if "image_id" not in df.columns and "filename" in df.columns:
        df["image_id"] = df["filename"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    if "filename" not in df.columns and "image_id" in df.columns:
        df["filename"] = df["image_id"].astype(str) + ".jpg"
    return df


def build_metadata_lookup(summary_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Create image_id -> metadata lookup from preprocessing summary."""
    if summary_df.empty or "image_id" not in summary_df.columns:
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in summary_df.iterrows():
        image_id = str(row["image_id"])
        lookup[image_id] = row.to_dict()
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 7 Hu moments from Flavia leaf masks.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(r"C:\AIA_workspace\data\processed"),
        help="Directory produced by preprocessing.py.",
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
    print("Mask 文件夹:", mask_dir)
    print("找到 mask 数量:", len(mask_paths))

    if not mask_paths:
        raise FileNotFoundError(f"没有找到 mask 文件: {mask_dir}")

    summary_df = load_summary(processed_dir)
    summary_lookup = build_metadata_lookup(summary_df)

    records: list[dict[str, Any]] = []
    failed_files: list[str] = []

    for i, mask_path in enumerate(mask_paths, start=1):
        print(f"[{i}/{len(mask_paths)}] 正在提取 Hu moments: {mask_path.name}")

        hu = extract_hu_moments(mask_path)
        if hu is None:
            failed_files.append(str(mask_path))
            continue

        image_id = mask_path.stem.replace("_mask", "")
        filename = f"{image_id}.jpg"

        metadata = summary_lookup.get(image_id, {})
        fallback = flavia_metadata_from_image_id(image_id)

        label = metadata.get("label", fallback["label"])
        scientific_name = metadata.get("scientific_name", fallback["scientific_name"])
        common_name = metadata.get("common_name", fallback["common_name"])

        record: dict[str, Any] = {
            "filename": metadata.get("filename", filename),
            "image_id": image_id,
            "label": label,
            "scientific_name": scientific_name,
            "common_name": common_name,
        }

        for j, value in enumerate(hu, start=1):
            record[f"hu_{j}"] = value

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    failed_path = features_dir / "hu_failed_files.txt"
    with open(failed_path, "w", encoding="utf-8") as f:
        for item in failed_files:
            f.write(item + "\n")

    print("\nHu moments 提取完成")
    print("成功数量:", len(records))
    print("失败数量:", len(failed_files))
    print("输出 CSV:", output_csv)
    print("失败列表:", failed_path)


if __name__ == "__main__":
    main()
