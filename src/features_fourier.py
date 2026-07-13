"""
Extract Fourier descriptors from ordered leaf boundary points.

Input:
    data/processed/boundaries/*_boundary.npy
    data/processed/preprocess_summary.csv

Output:
    data/processed/features/fourier_features.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_boundary(boundary_path: Path) -> np.ndarray:
    """Load ordered boundary points saved by preprocessing."""
    return np.asarray(np.load(boundary_path), dtype=np.float64)


def signed_polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def make_orientation_consistent(points: np.ndarray) -> np.ndarray:
    if signed_polygon_area(points) < 0:
        return points[::-1].copy()
    return points


def align_start_point(points: np.ndarray) -> np.ndarray:
    y = points[:, 1]
    x = points[:, 0]
    min_y = np.min(y)
    candidate_indices = np.where(y == min_y)[0]
    start_idx = candidate_indices[np.argmin(x[candidate_indices])]
    return np.roll(points, -int(start_idx), axis=0)


def resample_closed_boundary(points: np.ndarray, n_points: int = 256) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
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
    points = make_orientation_consistent(points)
    points = align_start_point(points)
    points = resample_closed_boundary(points, n_points=n_points)

    points = points - points.mean(axis=0)
    rms_radius = np.sqrt(np.mean(np.sum(points**2, axis=1)))
    if rms_radius <= 1e-12:
        raise ValueError("Boundary scale is too small.")

    return points / rms_radius


def compute_fourier_descriptors(
    points: np.ndarray,
    n_points: int = 256,
    num_descriptors: int = 20,
    use_magnitude: bool = True,
) -> np.ndarray:
    if num_descriptors < 2:
        raise ValueError("num_descriptors should be at least 2.")
    if num_descriptors >= n_points:
        raise ValueError("num_descriptors must be smaller than n_points.")

    norm_points = normalize_boundary(points, n_points=n_points)
    z = norm_points[:, 0] + 1j * norm_points[:, 1]
    coeffs = np.fft.fft(z) / len(z)

    scale = np.abs(coeffs[1])
    if scale <= 1e-12:
        scale = np.max(np.abs(coeffs[1:]))
    if scale <= 1e-12:
        raise ValueError("Fourier coefficients have near-zero scale.")

    coeffs = coeffs / scale

    half = num_descriptors // 2
    positive = coeffs[1 : half + 1]
    negative = coeffs[-half:]
    selected = np.concatenate([positive, negative])

    if num_descriptors % 2 == 1:
        selected = np.concatenate([selected, coeffs[half + 1 : half + 2]])

    if use_magnitude:
        features = np.abs(selected)
    else:
        features = np.concatenate([selected.real, selected.imag])

    return features.astype(np.float64)


def load_summary(processed_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(processed_dir / "preprocess_summary.csv")
    df["image_id"] = df["image_id"].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Fourier descriptors from Flavia leaf boundaries.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(r"C:\AIA_workspace\data\processed"),
        help="Directory produced by preprocessing.",
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
    features_dir = processed_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    output_csv = args.output_csv or (features_dir / "fourier_features.csv")

    boundary_paths = sorted(boundary_dir.glob("*.npy"))
    print("Boundary directory:", boundary_dir)
    print("Boundary count:", len(boundary_paths))

    if not boundary_paths:
        raise FileNotFoundError(f"No boundary files found: {boundary_dir}")

    summary_lookup = {
        str(row["image_id"]): row.to_dict()
        for _, row in load_summary(processed_dir).iterrows()
    }
    use_magnitude = not args.use_complex_parts

    records: list[dict[str, object]] = []
    failed_files: list[str] = []

    for i, boundary_path in enumerate(boundary_paths, start=1):
        print(f"[{i}/{len(boundary_paths)}] Extracting Fourier descriptors: {boundary_path.name}")

        image_id = boundary_path.stem.replace("_boundary", "")
        points = load_boundary(boundary_path)
        features = compute_fourier_descriptors(
            points,
            n_points=args.n_boundary_points,
            num_descriptors=args.num_descriptors,
            use_magnitude=use_magnitude,
        )
        metadata = summary_lookup[image_id]

        record: dict[str, object] = {
            "filename": metadata["filename"],
            "image_id": image_id,
            "label": metadata["label"],
            "scientific_name": metadata["scientific_name"],
            "common_name": metadata["common_name"],
            "boundary_points_original": len(points),
            "boundary_points_resampled": args.n_boundary_points,
            "fourier_representation": "magnitude" if use_magnitude else "real_imag",
        }

        record.update(
            {f"fd_{j:02d}": value for j, value in enumerate(features, start=1)}
        )

        records.append(record)

    pd.DataFrame(records).to_csv(output_csv, index=False)

    failed_path = features_dir / "fourier_failed_files.txt"
    failed_path.write_text("".join(f"{item}\n" for item in failed_files), encoding="utf-8")

    print("\nFourier descriptor extraction finished")
    print("Success:", len(records))
    print("Failed:", len(failed_files))
    print("Output CSV:", output_csv)
    print("Failed file list:", failed_path)


if __name__ == "__main__":
    main()
