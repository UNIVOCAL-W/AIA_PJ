from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
CROPPED_DIR = WORKSPACE_DIR / "data" / "processed" / "cropped"
OUTPUT_DIR = WORKSPACE_DIR / "data" / "processed" / "splits"
TEST_SIZE = 0.3
RANDOM_STATE = 0


FLAVIA_RANGES = [
    (1001, 1059, 1),
    (1060, 1122, 2),
    (1552, 1616, 3),
    (1123, 1194, 4),
    (1195, 1267, 5),
    (1268, 1323, 6),
    (1324, 1385, 7),
    (1386, 1437, 8),
    (1497, 1551, 9),
    (1438, 1496, 10),
    (2001, 2050, 11),
    (2051, 2113, 12),
    (2114, 2165, 14),
    (2166, 2230, 15),
    (2231, 2290, 16),
    (2291, 2346, 17),
    (2347, 2423, 18),
    (2424, 2485, 19),
    (2486, 2546, 20),
    (2547, 2612, 21),
    (2616, 2675, 22),
    (3001, 3055, 23),
    (3056, 3110, 24),
    (3111, 3175, 25),
    (3176, 3229, 26),
    (3230, 3281, 27),
    (3282, 3334, 28),
    (3335, 3389, 29),
    (3390, 3446, 30),
    (3447, 3510, 31),
    (3511, 3563, 32),
    (3566, 3621, 33),
]


def get_label(image_id: str) -> int | None:
    number = int(image_id)
    for start, end, label in FLAVIA_RANGES:
        if start <= number <= end:
            return label
    return None


def build_table() -> pd.DataFrame:
    rows = []
    crop_paths = sorted(CROPPED_DIR.glob("*_crop.png"))

    for crop_path in crop_paths:
        image_id = crop_path.stem.replace("_crop", "")
        label = get_label(image_id)

        if label is None:
            print(f"[跳过] 无法识别类别: {crop_path.name}")
            continue

        rows.append(
            {
                "image_id": image_id,
                "filename": f"{image_id}.jpg",
                "crop_path": str(crop_path),
                "label": label,
            }
        )

    if not rows:
        raise FileNotFoundError(f"没有找到可用的裁剪图片: {CROPPED_DIR}")

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_table()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    train_df = train_df.sort_values("image_id").reset_index(drop=True)
    test_df = test_df.sort_values("image_id").reset_index(drop=True)

    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    ).sort_values("image_id")

    train_path = OUTPUT_DIR / "modern_train.csv"
    test_path = OUTPUT_DIR / "modern_test.csv"
    split_path = OUTPUT_DIR / "modern_split_70_30_seed0.csv"
    counts_path = OUTPUT_DIR / "modern_class_counts_70_30_seed0.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    split_df.to_csv(split_path, index=False)

    counts = (
        split_df.groupby(["label", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts.to_csv(counts_path, index=False)

    print("现代流程数据划分完成")
    print(f"总样本数: {len(df)}")
    print(f"训练集: {len(train_df)}")
    print(f"测试集: {len(test_df)}")
    print(f"类别数: {df['label'].nunique()}")
    print(f"训练集保存到: {train_path}")
    print(f"测试集保存到: {test_path}")
    print(f"完整划分保存到: {split_path}")


if __name__ == "__main__":
    main()
