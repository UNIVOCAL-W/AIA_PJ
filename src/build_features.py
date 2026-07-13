"""
build_features.py

Merge Hu moments features and Fourier descriptor features into one classical
feature table for Topic A: Leaf Species Recognition.

Expected input files by default:
    C:/AIA_workspace/data/processed/features/hu_moments_features.csv
    C:/AIA_workspace/data/processed/features/fourier_features.csv

Output files by default:
    C:/AIA_workspace/data/processed/features/classical_features.csv
    C:/AIA_workspace/data/processed/features/classical_feature_columns.txt
    C:/AIA_workspace/data/processed/features/build_features_summary.txt

This script does NOT standardize/normalize the feature values. Scaling should be
fit only on the training split later to avoid data leakage, especially for k-NN.
"""

import argparse
from pathlib import Path

import pandas as pd


def read_csv_checked(path: Path, name: str) -> pd.DataFrame:
    """Read a CSV file and raise a clear error if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{name} file does not exist: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{name} is an empty table: {path}")

    return df


def normalize_image_id_column(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Ensure the table has a string image_id column.

    Accepted cases:
    - image_id already exists
    - filename exists, such as 1001.jpg or 1001_mask.png
    """
    df = df.copy()

    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        return df

    if "filename" in df.columns:
        df["image_id"] = (
            df["filename"]
            .astype(str)
            .str.replace(".jpg", "", regex=False)
            .str.replace(".jpeg", "", regex=False)
            .str.replace(".png", "", regex=False)
            .str.replace("_mask", "", regex=False)
            .str.replace("_crop", "", regex=False)
            .str.replace("_contour", "", regex=False)
        )
        return df

    raise ValueError(
        f"{table_name} has neither image_id nor filename, so the two feature tables cannot be aligned."
    )


def find_feature_columns(
    df: pd.DataFrame,
    prefixes: tuple[str, ...],
    exclude_columns: set[str],
) -> list[str]:
    """Find numeric feature columns by prefix."""
    prefixes = tuple(prefixes)

    candidates = [
        col for col in df.columns
        if col not in exclude_columns and col.startswith(prefixes)
    ]

    numeric_candidates = []
    for col in candidates:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().all():
            df[col] = converted
            numeric_candidates.append(col)

    return numeric_candidates


def ensure_no_duplicate_image_ids(df: pd.DataFrame, table_name: str) -> None:
    """Check duplicated image_id values."""
    duplicated = df[df["image_id"].duplicated()]["image_id"].tolist()

    if duplicated:
        preview = duplicated[:10]
        raise ValueError(
            f"{table_name} contains duplicated image_id values, for example: {preview}. "
            "Check whether the feature extraction script wrote the same image more than once."
        )


def choose_metadata_columns(hu_df: pd.DataFrame, fourier_df: pd.DataFrame) -> list[str]:
    """
    Choose metadata columns to keep in the final output.

    Prefer metadata from the Hu table because Hu extraction normally reads masks
    and keeps label/species information. If some metadata only exists in Fourier,
    it will also be preserved after merge if available.
    """
    preferred = [
        "filename",
        "image_id",
        "label",
        "scientific_name",
        "common_name",
    ]

    return [
        col for col in preferred
        if col in hu_df.columns or col in fourier_df.columns
    ]


def fill_filename_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Create filename from image_id if filename is missing."""
    df = df.copy()
    if "filename" not in df.columns:
        df.insert(0, "filename", df["image_id"].astype(str) + ".jpg")
    return df


def save_feature_columns(feature_columns: list[str], output_path: Path) -> None:
    """Save the feature column names, one per line."""
    output_path.write_text(
        "".join(f"{column}\n" for column in feature_columns),
        encoding="utf-8",
    )


def write_summary(
    output_path: Path,
    hu_path: Path,
    fourier_path: Path,
    output_csv: Path,
    hu_count: int,
    fourier_count: int,
    merged_count: int,
    hu_feature_count: int,
    fourier_feature_count: int,
    label_count: int | None,
    missing_hu_count: int,
    missing_fourier_count: int,
) -> None:
    """Write a readable summary text file."""
    lines = [
        "Build classical features summary",
        "================================",
        f"Hu input: {hu_path}",
        f"Fourier input: {fourier_path}",
        f"Output CSV: {output_csv}",
        "",
        f"Hu rows: {hu_count}",
        f"Fourier rows: {fourier_count}",
        f"Merged rows: {merged_count}",
        f"Hu feature columns: {hu_feature_count}",
        f"Fourier feature columns: {fourier_feature_count}",
        f"Total feature columns: {hu_feature_count + fourier_feature_count}",
        f"Number of labels: {label_count if label_count is not None else 'label column not found'}",
        "",
        f"Rows missing Hu features after merge: {missing_hu_count}",
        f"Rows missing Fourier features after merge: {missing_fourier_count}",
        "",
        "Note:",
        "Feature scaling is intentionally not done here.",
        "Fit scalers only on the training set later to avoid data leakage.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_classical_features(
    processed_dir: Path,
    hu_csv: Path | None = None,
    fourier_csv: Path | None = None,
    output_csv: Path | None = None,
    join_type: str = "inner",
) -> Path:
    """Merge Hu moments and Fourier descriptors into one CSV."""

    features_dir = processed_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    hu_csv = hu_csv or features_dir / "hu_moments_features.csv"
    fourier_csv = fourier_csv or features_dir / "fourier_features.csv"
    output_csv = output_csv or features_dir / "classical_features.csv"

    hu_df = read_csv_checked(hu_csv, "Hu moments features")
    fourier_df = read_csv_checked(fourier_csv, "Fourier features")

    hu_df = normalize_image_id_column(hu_df, "Hu moments features")
    fourier_df = normalize_image_id_column(fourier_df, "Fourier features")

    ensure_no_duplicate_image_ids(hu_df, "Hu moments features")
    ensure_no_duplicate_image_ids(fourier_df, "Fourier features")

    metadata_columns = choose_metadata_columns(hu_df, fourier_df)

    exclude_columns = {
        "filename",
        "image_id",
        "label",
        "scientific_name",
        "common_name",
        "mask_path",
        "boundary_path",
        "source_path",
    }

    hu_feature_cols = find_feature_columns(
        hu_df,
        prefixes=("hu_",),
        exclude_columns=exclude_columns,
    )

    fourier_feature_cols = find_feature_columns(
        fourier_df,
        prefixes=("fd_", "fourier_", "fft_"),
        exclude_columns=exclude_columns,
    )

    if not hu_feature_cols:
        raise ValueError(
            "No Hu feature columns found. Check whether hu_moments_features.csv contains hu_1 ... hu_7."
        )

    if not fourier_feature_cols:
        raise ValueError(
            "No Fourier feature columns found. Check whether fourier_features.csv contains columns starting with fd_ or fourier_."
        )

    # Keep Hu metadata first. For Fourier, only keep image_id and feature columns.
    hu_keep_cols = [col for col in metadata_columns if col in hu_df.columns] + hu_feature_cols
    if "image_id" not in hu_keep_cols:
        hu_keep_cols.insert(0, "image_id")

    fourier_keep_cols = ["image_id"] + fourier_feature_cols

    merged = pd.merge(
        hu_df[hu_keep_cols],
        fourier_df[fourier_keep_cols],
        on="image_id",
        how=join_type,
        validate="one_to_one",
    )

    merged = fill_filename_if_missing(merged)

    # Put columns in a stable and readable order.
    final_metadata = [
        col for col in ["filename", "image_id", "label", "scientific_name", "common_name"]
        if col in merged.columns
    ]
    feature_columns = hu_feature_cols + fourier_feature_cols
    final_columns = final_metadata + feature_columns

    merged = merged[final_columns]

    # Sort by image_id numerically when possible.
    merged["_sort_id"] = pd.to_numeric(merged["image_id"], errors="coerce")
    merged = merged.sort_values(["_sort_id", "image_id"]).drop(columns=["_sort_id"])

    # Detect missing values before saving.
    missing_total = int(merged[feature_columns].isna().sum().sum())
    if missing_total > 0:
        raise ValueError(
            f"The merged feature table contains {missing_total} missing feature values. "
            "Check whether the Hu and Fourier feature files are complete."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    feature_columns_path = output_csv.parent / "classical_feature_columns.txt"
    save_feature_columns(feature_columns, feature_columns_path)

    # Some useful diagnostics for merge quality.
    hu_ids = set(hu_df["image_id"].astype(str))
    fourier_ids = set(fourier_df["image_id"].astype(str))
    missing_hu_count = len(fourier_ids - hu_ids)
    missing_fourier_count = len(hu_ids - fourier_ids)

    label_count = None
    if "label" in merged.columns:
        label_count = int(merged["label"].nunique())

    summary_path = output_csv.parent / "build_features_summary.txt"
    write_summary(
        output_path=summary_path,
        hu_path=hu_csv,
        fourier_path=fourier_csv,
        output_csv=output_csv,
        hu_count=len(hu_df),
        fourier_count=len(fourier_df),
        merged_count=len(merged),
        hu_feature_count=len(hu_feature_cols),
        fourier_feature_count=len(fourier_feature_cols),
        label_count=label_count,
        missing_hu_count=missing_hu_count,
        missing_fourier_count=missing_fourier_count,
    )

    print("\nMerge finished")
    print("Hu feature count:", len(hu_feature_cols))
    print("Fourier feature count:", len(fourier_feature_cols))
    print("Total feature count:", len(feature_columns))
    print("Merged sample count:", len(merged))
    if label_count is not None:
        print("Number of classes:", label_count)
    print("Output file:", output_csv)
    print("Feature column file:", feature_columns_path)
    print("Summary file:", summary_path)

    if join_type == "inner" and (missing_hu_count > 0 or missing_fourier_count > 0):
        print("\nNote: the inner join discarded samples that do not have both Hu and Fourier features.")
        print("Samples in Fourier but missing from Hu:", missing_hu_count)
        print("Samples in Hu but missing from Fourier:", missing_fourier_count)

    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Hu moments and Fourier descriptors into classical_features.csv."
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(r"C:\AIA_workspace\data\processed"),
        help="Processed data directory. Default: C:\\AIA_workspace\\data\\processed",
    )

    parser.add_argument(
        "--hu-csv",
        type=Path,
        default=None,
        help="Path to hu_moments_features.csv. Default: processed-dir/features/hu_moments_features.csv",
    )

    parser.add_argument(
        "--fourier-csv",
        type=Path,
        default=None,
        help="Path to fourier_features.csv. Default: processed-dir/features/fourier_features.csv",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: processed-dir/features/classical_features.csv",
    )

    parser.add_argument(
        "--join",
        choices=["inner", "left", "right", "outer"],
        default="inner",
        help="How to merge Hu and Fourier tables. Default: inner.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_classical_features(
        processed_dir=args.processed_dir,
        hu_csv=args.hu_csv,
        fourier_csv=args.fourier_csv,
        output_csv=args.output_csv,
        join_type=args.join,
    )


if __name__ == "__main__":
    main()
