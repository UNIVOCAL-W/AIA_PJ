import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = WORKSPACE_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

FEATURE_DIR = WORKSPACE_DIR / "data" / "processed" / "features"
SPLIT_DIR = WORKSPACE_DIR / "data" / "processed" / "splits"
MODEL_DIR = WORKSPACE_DIR / "models" / "modern"
PREDICTION_DIR = WORKSPACE_DIR / "data" / "processed" / "predictions" / "modern"
RESULT_DIR = WORKSPACE_DIR / "results" / "modern"
TABLE_DIR = RESULT_DIR / "tables"
FIGURE_DIR = RESULT_DIR / "figures"


def load_features(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    feature_columns = [col for col in df.columns if col.startswith("feat_")]

    if not feature_columns:
        raise ValueError(f"没有找到特征列: {path}")

    return df, feature_columns


def plot_confusion_matrix(cm, labels, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class_accuracy(per_class_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(per_class_df["label"].astype(str), per_class_df["accuracy"])
    ax.set_title("MobileNetV2 + Linear SVM Per-Class Accuracy")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overall_accuracy(train_accuracy: float, test_accuracy: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Train", "Test"], [train_accuracy, test_accuracy], color=["#4C78A8", "#F58518"])
    ax.set_title("MobileNetV2 + Linear SVM Overall Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)

    for index, value in enumerate([train_accuracy, test_accuracy]):
        ax.text(index, value + 0.02, f"{value:.4f}", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_example_grid(example_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if example_df.empty:
        return

    count = min(len(example_df), 12)
    cols = 4
    rows = (count + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for index, (_, row) in enumerate(example_df.head(count).iterrows()):
        image = plt.imread(row["crop_path"])
        axes[index].imshow(image)
        axes[index].set_title(
            f"True: {row['label']}  Pred: {row['prediction']}",
            fontsize=9,
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 MobileNetV2 特征 + Linear SVM 分类器。")
    parser.add_argument("--max-iter", type=int, default=2000, help="SGD linear SVM 最大迭代次数。")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    train_path = FEATURE_DIR / "mobilenet_train_features.csv"
    test_path = FEATURE_DIR / "mobilenet_test_features.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("请先运行: python src\\mobilenet_features.py --device cuda --batch-size 8")

    train_df, feature_columns = load_features(train_path)
    test_df, _ = load_features(test_path)

    x_train = train_df[feature_columns]
    y_train = train_df["label"]
    x_test = test_df[feature_columns]
    y_test = test_df["label"]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SGDClassifier(
                    loss="hinge",
                    alpha=1e-4,
                    max_iter=args.max_iter,
                    tol=1e-3,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("开始训练 MobileNetV2 + Linear SVM")
    print(f"训练样本数: {len(train_df)}")
    print(f"测试样本数: {len(test_df)}")
    print(f"特征维度: {len(feature_columns)}")

    start_time = time.time()
    model.fit(x_train, y_train)
    training_time = time.time() - start_time

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    model_path = MODEL_DIR / "mobilenet_linear_svm.joblib"
    config_path = MODEL_DIR / "mobilenet_linear_svm_config.json"
    prediction_path = PREDICTION_DIR / "mobilenet_svm_predictions.csv"

    joblib.dump(model, model_path)

    split_test_path = SPLIT_DIR / "modern_test.csv"
    split_test_df = pd.read_csv(split_test_path)
    crop_lookup = split_test_df[["image_id", "crop_path"]].copy()
    crop_lookup["image_id"] = crop_lookup["image_id"].astype(str)

    prediction_df = test_df[["image_id", "filename", "label"]].copy()
    prediction_df["image_id"] = prediction_df["image_id"].astype(str)
    prediction_df["prediction"] = test_pred
    prediction_df["correct"] = prediction_df["label"] == prediction_df["prediction"]
    prediction_df = prediction_df.merge(crop_lookup, on="image_id", how="left")
    prediction_df.to_csv(prediction_path, index=False)

    labels = sorted(pd.concat([y_train, y_test]).unique())
    report = classification_report(y_test, test_pred, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_path = TABLE_DIR / "classification_report_mobilenet_svm.csv"
    report_df.to_csv(report_path)

    cm = confusion_matrix(y_test, test_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_path = TABLE_DIR / "confusion_matrix_mobilenet_svm.csv"
    cm_df.to_csv(cm_path)

    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    cm_normalized_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    cm_normalized_path = TABLE_DIR / "confusion_matrix_normalized_mobilenet_svm.csv"
    cm_normalized_df.to_csv(cm_normalized_path)

    per_class_rows = []
    for label in labels:
        class_rows = prediction_df[prediction_df["label"] == label]
        correct_count = int(class_rows["correct"].sum())
        total_count = len(class_rows)
        per_class_rows.append(
            {
                "label": label,
                "correct": correct_count,
                "total": total_count,
                "accuracy": correct_count / total_count if total_count else 0,
            }
        )

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_path = TABLE_DIR / "per_class_accuracy_mobilenet_svm.csv"
    per_class_df.to_csv(per_class_path, index=False)

    easy_examples = prediction_df[prediction_df["correct"]].head(12)
    hard_examples = prediction_df[~prediction_df["correct"]].head(12)
    easy_path = TABLE_DIR / "easy_correct_examples_mobilenet_svm.csv"
    hard_path = TABLE_DIR / "hard_wrong_examples_mobilenet_svm.csv"
    easy_examples.to_csv(easy_path, index=False)
    hard_examples.to_csv(hard_path, index=False)

    figure_path = FIGURE_DIR / "confusion_matrix_mobilenet_svm.png"
    figure_normalized_path = FIGURE_DIR / "confusion_matrix_normalized_mobilenet_svm.png"
    per_class_figure_path = FIGURE_DIR / "per_class_accuracy_mobilenet_svm.png"
    overall_figure_path = FIGURE_DIR / "overall_accuracy_mobilenet_svm.png"
    easy_figure_path = FIGURE_DIR / "easy_correct_examples_mobilenet_svm.png"
    hard_figure_path = FIGURE_DIR / "hard_wrong_examples_mobilenet_svm.png"

    plot_confusion_matrix(cm, labels, figure_path, "MobileNetV2 + Linear SVM Confusion Matrix")
    plot_confusion_matrix(
        cm_normalized,
        labels,
        figure_normalized_path,
        "MobileNetV2 + Linear SVM Normalized Confusion Matrix",
    )
    plot_per_class_accuracy(per_class_df, per_class_figure_path)
    plot_overall_accuracy(train_accuracy, test_accuracy, overall_figure_path)
    plot_example_grid(easy_examples, easy_figure_path, "MobileNetV2 + Linear SVM Correct Examples")
    plot_example_grid(hard_examples, hard_figure_path, "MobileNetV2 + Linear SVM Wrong Examples")

    summary = pd.DataFrame(
        [
            {
                "model": "mobilenet_v2_linear_svm_sgd",
                "feature_count": len(feature_columns),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "training_time_seconds": training_time,
                "model_path": str(model_path),
                "prediction_path": str(prediction_path),
            }
        ]
    )
    summary_path = TABLE_DIR / "modern_training_summary.csv"
    summary.to_csv(summary_path, index=False)

    config = {
        "model": "MobileNetV2 frozen features + linear SVM trained with SGDClassifier hinge loss",
        "feature_count": len(feature_columns),
        "loss": "hinge",
        "alpha": 1e-4,
        "tol": 1e-3,
        "max_iter": args.max_iter,
        "random_state": 0,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("训练完成")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"模型保存到: {model_path}")
    print(f"预测结果保存到: {prediction_path}")
    print(f"评估摘要保存到: {summary_path}")
    print(f"现代流程表格保存到: {TABLE_DIR}")
    print(f"现代流程图片保存到: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
