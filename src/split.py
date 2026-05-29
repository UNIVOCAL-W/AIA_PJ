"""
split_data.py

Create a stratified train/test split for Topic A: Leaf Species Recognition.

Default input:
    C:\\AIA_workspace\\data\\processed\\features\\classical_features.csv

Default outputs:
    C:\\AIA_workspace\\data\\processed\\splits\\split_70_30_seed0.csv
    C:\\AIA_workspace\\data\\processed\\splits\\train_features_70_30_seed0.csv
    C:\\AIA_workspace\\data\\processed\\splits\\test_features_70_30_seed0.csv
    C:\\AIA_workspace\\data\\processed\\splits\\class_counts_70_30_seed0.csv
    C:\\AIA_workspace\\data\\processed\\splits\\split_summary_70_30_seed0.txt

This follows the Exercise 2 scikit-learn pattern:
    train_test_split(..., test_size=0.3, stratify=y, random_state=0)
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_FEATURES_CSV = Path(
    r"C:\AIA_workspace\data\processed\features\classical_features.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\AIA_workspace\data\processed\splits"
)


def read_features(features_csv: Path) -> pd.DataFrame:
    """Read classical feature table and do basic checks."""
    if not features_csv.exists():
        raise FileNotFoundError(f"找不到特征文件: {features_csv}")

    df = pd.read_csv(features_csv)

    if df.empty:
        raise ValueError(f"特征文件是空的: {features_csv}")

    required_columns = {"image_id", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"特征文件缺少必要列: {missing}")

    # 统一 image_id 格式，避免 1001 被读成 1001.0
    df["image_id"] = df["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    if "filename" not in df.columns:
        df.insert(0, "filename", df["image_id"].astype(str) + ".jpg")

    if df["image_id"].duplicated().any():
        duplicated = df.loc[df["image_id"].duplicated(), "image_id"].head(10).tolist()
        raise ValueError(f"存在重复 image_id，例如: {duplicated}")

    if df["label"].isna().any():
        raise ValueError("label 列中存在空值，请先检查 classical_features.csv。")

    return df


def make_stratified_split(
    df: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> pd.DataFrame:
    """Split dataframe into train/test with class stratification."""
    y = df["label"]

    class_counts = y.value_counts().sort_index()
    too_small = class_counts[class_counts < 2]
    if not too_small.empty:
        raise ValueError(
            "以下类别样本数少于 2，无法做 stratified train/test split:\n"
            + too_small.to_string()
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
        shuffle=True,
    )

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 排序，方便人工检查
    split_df["_sort_id"] = pd.to_numeric(split_df["image_id"], errors="coerce")
    split_df = split_df.sort_values(["_sort_id", "image_id"]).drop(columns=["_sort_id"])

    return split_df


def save_outputs(
    split_df: pd.DataFrame,
    output_dir: Path,
    test_size: float,
    random_seed: int,
    features_csv: Path,
) -> None:
    """Save split CSVs, class-count table, and summary file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ratio = int(round((1.0 - test_size) * 100))
    test_ratio = int(round(test_size * 100))
    tag = f"{train_ratio}_{test_ratio}_seed{random_seed}"

    split_csv = output_dir / f"split_{tag}.csv"
    train_csv = output_dir / f"train_features_{tag}.csv"
    test_csv = output_dir / f"test_features_{tag}.csv"
    counts_csv = output_dir / f"class_counts_{tag}.csv"
    summary_txt = output_dir / f"split_summary_{tag}.txt"

    split_df.to_csv(split_csv, index=False)

    train_df = split_df[split_df["split"] == "train"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    class_counts = (
        pd.crosstab(split_df["label"], split_df["split"])
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for col in ["train", "test"]:
        if col not in class_counts.columns:
            class_counts[col] = 0

    class_counts["total"] = class_counts["train"] + class_counts["test"]
    class_counts["train_ratio"] = class_counts["train"] / class_counts["total"]
    class_counts["test_ratio"] = class_counts["test"] / class_counts["total"]
    class_counts = class_counts[["label", "train", "test", "total", "train_ratio", "test_ratio"]]
    class_counts.to_csv(counts_csv, index=False)

    feature_columns = [
        col for col in split_df.columns
        if col.startswith("hu_") or col.startswith("fd_") or col.startswith("fourier_")
    ]

    summary_lines = [
        "Stratified train/test split summary",
        "===================================",
        f"Input features: {features_csv}",
        f"Output split: {split_csv}",
        f"Random seed: {random_seed}",
        f"Requested test size: {test_size}",
        "",
        f"Total samples: {len(split_df)}",
        f"Train samples: {len(train_df)}",
        f"Test samples: {len(test_df)}",
        f"Number of classes: {split_df['label'].nunique()}",
        f"Number of feature columns: {len(feature_columns)}",
        "",
        "Files written:",
        f"- {split_csv}",
        f"- {train_csv}",
        f"- {test_csv}",
        f"- {counts_csv}",
        "",
        "Note:",
        "This split follows the Exercise 2 scikit-learn pattern:",
        "train_test_split(X, y, test_size=0.3, stratify=y, random_state=0).",
        "All later classifiers should reuse this split for fair comparison.",
    ]

    summary_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n划分完成")
    print("总样本数:", len(split_df))
    print("训练集样本数:", len(train_df))
    print("测试集样本数:", len(test_df))
    print("类别数量:", split_df["label"].nunique())
    print("特征数量:", len(feature_columns))
    print("split 文件:", split_csv)
    print("train 文件:", train_csv)
    print("test 文件:", test_csv)
    print("类别统计:", counts_csv)
    print("摘要文件:", summary_txt)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified train/test split for classical_features.csv."
    )

    parser.add_argument(
        "--features-csv",
        type=Path,
        default=DEFAULT_FEATURES_CSV,
        help=r"Path to classical_features.csv. Default: C:\AIA_workspace\data\processed\features\classical_features.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=r"Directory for split outputs. Default: C:\AIA_workspace\data\processed\splits",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test set fraction. Exercise 2 uses 0.3 by default.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed. Exercise 2 uses random_state=0 by default.",
    )

    return parser.parse_args()



def main() -> None:
    args = parse_args()

    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size 必须在 0 和 1 之间，例如 0.3。")

    df = read_features(args.features_csv)
    split_df = make_stratified_split(
        df=df,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    save_outputs(
        split_df=split_df,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_seed=args.random_seed,
        features_csv=args.features_csv,
    )


if __name__ == "__main__":
    main()

