"""
features_fourier.py

Extract Fourier descriptors from ordered leaf boundary points produced by preprocessing.py.

Expected input:
    C:/AIA_workspace/data/processed/boundaries/*.npy
    C:/AIA_workspace/data/processed/masks/*_mask.png             optional fallback
    C:/AIA_workspace/data/processed/preprocess_summary.csv       optional, but recommended

Output:
    C:/AIA_workspace/data/processed/features/fourier_features.csv

Run:
    python features_fourier.py
or:
    python features_fourier.py --processed-dir C:/AIA_workspace/data/processed --num-descriptors 20
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
# Boundary loading
# -------------------------


def load_boundary_from_npy(boundary_path: Path) -> np.ndarray | None:
    """Load ordered boundary points saved by preprocessing.py."""
    try:
        points = np.load(boundary_path)
    except Exception as exc:
        print(f"[boundary 读取失败] {boundary_path}: {exc}")
        return None

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 8:
        print(f"[boundary 无效] {boundary_path}, shape={points.shape}")
        return None
    return points


def load_boundary_from_mask(mask_path: Path) -> np.ndarray | None:
    """
    Fallback: extract the largest ordered contour from a mask if boundary .npy is unavailable.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[mask 读取失败] {mask_path}")
        return None

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print(f"[未找到轮廓] {mask_path}")
        return None

    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0, :].astype(np.float64)
    if len(points) < 8:
        print(f"[轮廓点太少] {mask_path}")
        return None
    return points


# -------------------------
# Boundary normalization
# -------------------------


def signed_polygon_area(points: np.ndarray) -> float:
    """Signed area. Positive/negative sign indicates traversal orientation."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def make_orientation_consistent(points: np.ndarray) -> np.ndarray:
    """
    Make traversal direction consistent across leaves.
    Image coordinates have y downward, but consistency is more important than absolute convention.
    """
    if signed_polygon_area(points) < 0:
        points = points[::-1].copy()
    return points


def align_start_point(points: np.ndarray) -> np.ndarray:
    """
    Align the boundary start point.

    We choose the point with the smallest y coordinate, i.e. the top-most point in image coordinates.
    If several points have the same y, choose the left-most among them.
    This implements the slide requirement to align the start point before Fourier descriptors.
    """
    y = points[:, 1]
    x = points[:, 0]
    min_y = np.min(y)
    candidate_indices = np.where(y == min_y)[0]
    start_idx = candidate_indices[np.argmin(x[candidate_indices])]
    return np.roll(points, -int(start_idx), axis=0)


def resample_closed_boundary(points: np.ndarray, n_points: int = 256) -> np.ndarray:
    """
    Resample an ordered closed boundary to a fixed number of points by arc length.

    A fixed-length boundary signal makes all Fourier feature vectors have the same dimension.
    """
    points = np.asarray(points, dtype=np.float64)

    # Close the curve explicitly for distance computation.
    closed = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]

    if total_length <= 0:
        raise ValueError("Boundary has zero length.")

    sample_positions = np.linspace(0, total_length, n_points, endpoint=False)
    x_new = np.interp(sample_positions, cumulative, closed[:, 0])
    y_new = np.interp(sample_positions, cumulative, closed[:, 1])
    return np.column_stack([x_new, y_new])


def normalize_boundary(points: np.ndarray, n_points: int = 256) -> np.ndarray:
    """
    Normalize ordered boundary points:
    1. consistent traversal direction,
    2. fixed start point,
    3. fixed number of samples,
    4. center on centroid,
    5. scale normalization.
    """
    points = make_orientation_consistent(points)
    points = align_start_point(points)
    points = resample_closed_boundary(points, n_points=n_points)

    # Centre on centroid.
    centroid = points.mean(axis=0)
    points = points - centroid

    # Fix scale using RMS radius. This is stable for different leaf sizes.
    rms_radius = np.sqrt(np.mean(np.sum(points**2, axis=1)))
    if rms_radius <= 1e-12:
        raise ValueError("Boundary scale is too small.")
    points = points / rms_radius

    return points


# -------------------------
# Fourier descriptors
# -------------------------


def compute_fourier_descriptors(
    points: np.ndarray,
    n_points: int = 256,
    num_descriptors: int = 20,
    use_magnitude: bool = True,
) -> np.ndarray:
    """
    Convert normalized boundary into Fourier descriptors.

    Boundary is represented as a complex signal:
        z[k] = x[k] + i y[k]

    We compute the DFT with np.fft.fft and keep low-frequency coefficients.
    The descriptor uses positive and negative low frequencies around the DC component.
    DC is skipped because centering makes it approximately zero.

    If use_magnitude=True, the output is real-valued and less sensitive to rotation/start phase.
    """
    if num_descriptors < 2:
        raise ValueError("num_descriptors should be at least 2.")
    if num_descriptors >= n_points:
        raise ValueError("num_descriptors must be smaller than n_points.")

    norm_points = normalize_boundary(points, n_points=n_points)
    z = norm_points[:, 0] + 1j * norm_points[:, 1]

    coeffs = np.fft.fft(z) / len(z)

    # Scale normalization again in Fourier domain: divide by first harmonic magnitude.
    scale = np.abs(coeffs[1])
    if scale <= 1e-12:
        scale = np.max(np.abs(coeffs[1:]))
    if scale <= 1e-12:
        raise ValueError("Fourier coefficients have near-zero scale.")
    coeffs = coeffs / scale

    # Keep low-frequency coefficients: +1,+2,... and -1,-2,...
    half = num_descriptors // 2
    positive = coeffs[1 : half + 1]
    negative = coeffs[-half:]
    selected = np.concatenate([positive, negative])

    if num_descriptors % 2 == 1:
        selected = np.concatenate([selected, coeffs[half + 1 : half + 2]])

    if use_magnitude:
        features = np.abs(selected)
    else:
        # Real-valued classifier input: concatenate real and imaginary parts.
        features = np.concatenate([selected.real, selected.imag])

    return features.astype(np.float64)


def load_summary(processed_dir: Path) -> pd.DataFrame:
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
    if summary_df.empty or "image_id" not in summary_df.columns:
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in summary_df.iterrows():
        image_id = str(row["image_id"])
        lookup[image_id] = row.to_dict()
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Fourier descriptors from Flavia leaf boundaries.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(r"C:\AIA_workspace\data\processed"),
        help="Directory produced by preprocessing.py.",
    )
    parser.add_argument(
        "--n-boundary-points",
        type=int,
        default=256,
        help="Number of resampled boundary points per leaf.",
    )
    parser.add_argument(
        "--num-descriptors",
        type=int,
        default=20,
        help="Number of low-frequency Fourier descriptor features to keep.",
    )
    parser.add_argument(
        "--use-complex-parts",
        action="store_true",
        help="Save real and imaginary parts instead of magnitudes. Default saves magnitudes.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: processed/features/fourier_features.csv",
    )
    args = parser.parse_args()

    processed_dir: Path = args.processed_dir
    boundary_dir = processed_dir / "boundaries"
    mask_dir = processed_dir / "masks"
    features_dir = processed_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    output_csv = args.output_csv or (features_dir / "fourier_features.csv")

    boundary_paths = sorted(boundary_dir.glob("*.npy"))
    print("Boundary 文件夹:", boundary_dir)
    print("找到 boundary 数量:", len(boundary_paths))

    if not boundary_paths:
        # Fallback to masks if boundary files were not saved.
        print("未找到 .npy boundary 文件，将尝试从 masks 重新提取轮廓。")
        mask_paths = sorted(mask_dir.glob("*_mask.png"))
        if not mask_paths:
            raise FileNotFoundError(f"没有找到 boundary 或 mask 文件: {boundary_dir}, {mask_dir}")
        input_items = [(p.stem.replace("_mask", ""), p, "mask") for p in mask_paths]
    else:
        input_items = [(p.stem.replace("_boundary", ""), p, "boundary") for p in boundary_paths]

    summary_df = load_summary(processed_dir)
    summary_lookup = build_metadata_lookup(summary_df)

    records: list[dict[str, Any]] = []
    failed_files: list[str] = []

    use_magnitude = not args.use_complex_parts

    for i, (image_id, path, source_type) in enumerate(input_items, start=1):
        print(f"[{i}/{len(input_items)}] 正在提取 Fourier descriptors: {path.name}")

        if source_type == "boundary":
            points = load_boundary_from_npy(path)
        else:
            points = load_boundary_from_mask(path)

        if points is None:
            failed_files.append(str(path))
            continue

        try:
            features = compute_fourier_descriptors(
                points,
                n_points=args.n_boundary_points,
                num_descriptors=args.num_descriptors,
                use_magnitude=use_magnitude,
            )
        except Exception as exc:
            print(f"[Fourier 失败] {path}: {exc}")
            failed_files.append(str(path))
            continue

        metadata = summary_lookup.get(image_id, {})
        fallback = flavia_metadata_from_image_id(image_id)

        label = metadata.get("label", fallback["label"])
        scientific_name = metadata.get("scientific_name", fallback["scientific_name"])
        common_name = metadata.get("common_name", fallback["common_name"])

        record: dict[str, Any] = {
            "filename": metadata.get("filename", f"{image_id}.jpg"),
            "image_id": image_id,
            "label": label,
            "scientific_name": scientific_name,
            "common_name": common_name,
            "boundary_points_original": len(points),
            "boundary_points_resampled": args.n_boundary_points,
            "fourier_representation": "magnitude" if use_magnitude else "real_imag",
        }

        for j, value in enumerate(features, start=1):
            record[f"fd_{j:02d}"] = value

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    failed_path = features_dir / "fourier_failed_files.txt"
    with open(failed_path, "w", encoding="utf-8") as f:
        for item in failed_files:
            f.write(item + "\n")

    print("\nFourier descriptors 提取完成")
    print("成功数量:", len(records))
    print("失败数量:", len(failed_files))
    print("输出 CSV:", output_csv)
    print("失败列表:", failed_path)


if __name__ == "__main__":
    main()
