"""
evaluate.py

Evaluate classical models for Topic A: Leaf Species Recognition.

Input:
    C:\\AIA_workspace\\data\\processed\\predictions\\classical\\*_predictions.csv

Output:
    C:\\AIA_workspace\\results\\tables\\
    C:\\AIA_workspace\\results\\figures\\

This script creates:
    - overall accuracy summary
    - per-class accuracy CSV
    - classification report CSV
    - confusion matrix CSV
    - confusion matrix plot
    - normalized confusion matrix plot
    - per-class accuracy plot
    - overall accuracy comparison plot
    - easy correct examples
    - hard wrong examples
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def find_prediction_files(predictions_dir: Path):
    files = sorted(predictions_dir.glob("*_predictions.csv"))

    if not files:
        raise FileNotFoundError(
            f"No prediction files found in: {predictions_dir}\n"
            "Expected files like gaussian_nb_predictions.csv or knn_k5_predictions.csv"
        )

    return files


def get_model_name(prediction_path: Path) -> str:
    return prediction_path.stem.replace("_predictions", "")


def find_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def prepare_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make prediction CSV robust to slightly different column names.
    Expected:
        true label column: label / y_true / true_label
        prediction column: prediction / y_pred / pred_label
    """

    df = df.copy()

    true_col = find_column(df, ["y_true", "true_label", "label"])
    pred_col = find_column(df, ["y_pred", "pred_label", "prediction", "predicted_label"])

    if true_col is None:
        raise ValueError(
            "Could not find true label column. Expected one of: "
            "y_true, true_label, label"
        )

    if pred_col is None:
        raise ValueError(
            "Could not find prediction column. Expected one of: "
            "y_pred, pred_label, prediction, predicted_label"
        )

    df["y_true"] = df[true_col].astype(str)
    df["y_pred"] = df[pred_col].astype(str)

    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    elif "filename" in df.columns:
        df["image_id"] = (
            df["filename"]
            .astype(str)
            .str.replace(".jpg", "", regex=False)
            .str.replace(".jpeg", "", regex=False)
            .str.replace(".png", "", regex=False)
        )
    else:
        df["image_id"] = np.arange(len(df)).astype(str)

    if "filename" not in df.columns:
        df["filename"] = df["image_id"].astype(str) + ".jpg"

    return df


def get_confidence_column(df: pd.DataFrame):
    """
    Try to find confidence column from prediction CSV.
    If not available, return None.
    """

    return find_column(
        df,
        ["confidence", "max_probability", "pred_probability", "probability", "score"],
    )


def get_labels_sorted(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred), key=lambda x: int(x) if str(x).isdigit() else str(x))
    return labels


def compute_per_class_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for label, group in df.groupby("y_true"):
        total = len(group)
        correct = int((group["y_true"] == group["y_pred"]).sum())
        acc = correct / total

        records.append({
            "label": label,
            "total": total,
            "correct": correct,
            "per_class_accuracy": acc,
        })

    result = pd.DataFrame(records)
    result["_sort_label"] = pd.to_numeric(result["label"], errors="coerce")
    result = result.sort_values(["_sort_label", "label"]).drop(columns=["_sort_label"])

    return result


def evaluate_one_model(prediction_path: Path, tables_dir: Path, figures_dir: Path):
    model_name = get_model_name(prediction_path)

    df = pd.read_csv(prediction_path)
    df = prepare_prediction_dataframe(df)

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    labels = get_labels_sorted(y_true, y_pred)

    overall_acc = accuracy_score(y_true, y_pred)

    # Per-class accuracy
    per_class_df = compute_per_class_accuracy(df)
    per_class_path = tables_dir / f"per_class_accuracy_{model_name}.csv"
    per_class_df.to_csv(per_class_path, index=False)

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_path = tables_dir / f"classification_report_{model_name}.csv"
    report_df.to_csv(report_path)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_path = tables_dir / f"confusion_matrix_{model_name}.csv"
    cm_df.to_csv(cm_path)

    # Normalized confusion matrix
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_norm_path = tables_dir / f"confusion_matrix_normalized_{model_name}.csv"
    cm_norm_df.to_csv(cm_norm_path)

    # Plots
    plot_confusion_matrix(
        cm,
        labels,
        figures_dir / f"confusion_matrix_{model_name}.png",
        title=f"Confusion Matrix - {model_name}",
        normalized=False,
    )

    plot_confusion_matrix(
        cm_norm,
        labels,
        figures_dir / f"confusion_matrix_normalized_{model_name}.png",
        title=f"Normalized Confusion Matrix - {model_name}",
        normalized=True,
    )

    plot_per_class_accuracy(
        per_class_df,
        figures_dir / f"per_class_accuracy_{model_name}.png",
        title=f"Per-Class Accuracy - {model_name}",
    )

    return {
        "model": model_name,
        "prediction_file": str(prediction_path),
        "num_samples": len(df),
        "num_classes": len(labels),
        "overall_accuracy": overall_acc,
        "per_class_accuracy_file": str(per_class_path),
        "classification_report_file": str(report_path),
        "confusion_matrix_file": str(cm_path),
        "confusion_matrix_normalized_file": str(cm_norm_path),
    }, df


def plot_confusion_matrix(cm, labels, output_path: Path, title: str, normalized: bool):
    plt.figure(figsize=(12, 10))

    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    tick_positions = np.arange(len(labels))
    plt.xticks(tick_positions, labels, rotation=90, fontsize=7)
    plt.yticks(tick_positions, labels, fontsize=7)

    if len(labels) <= 35:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                text = f"{value:.2f}" if normalized else str(int(value))
                if value > 0:
                    plt.text(j, i, text, ha="center", va="center", fontsize=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_class_accuracy(per_class_df: pd.DataFrame, output_path: Path, title: str):
    labels = per_class_df["label"].astype(str).tolist()
    acc = per_class_df["per_class_accuracy"].values

    plt.figure(figsize=(14, 5))
    plt.bar(labels, acc)
    plt.ylim(0, 1.05)
    plt.xlabel("Class label")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_overall_accuracy(summary_df: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["model"], summary_df["overall_accuracy"])
    plt.ylim(0, 1.05)
    plt.xlabel("Model")
    plt.ylabel("Overall accuracy")
    plt.title("Overall Accuracy Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_image(image_dir: Path, filename: str):
    path = image_dir / filename

    if not path.exists():
        image_id = Path(filename).stem
        candidates = list(image_dir.rglob(f"{image_id}.*"))
        if not candidates:
            return None
        path = candidates[0]

    img = cv2.imread(str(path))

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def select_easy_hard_examples(df: pd.DataFrame, max_examples: int):
    confidence_col = get_confidence_column(df)

    correct_df = df[df["y_true"] == df["y_pred"]].copy()
    wrong_df = df[df["y_true"] != df["y_pred"]].copy()

    if confidence_col is not None:
        correct_df[confidence_col] = pd.to_numeric(correct_df[confidence_col], errors="coerce")
        wrong_df[confidence_col] = pd.to_numeric(wrong_df[confidence_col], errors="coerce")

        easy = correct_df.sort_values(confidence_col, ascending=False).head(max_examples)
        hard = wrong_df.sort_values(confidence_col, ascending=False).head(max_examples)
    else:
        easy = correct_df.head(max_examples)
        hard = wrong_df.head(max_examples)

    return easy, hard


def plot_example_grid(
    examples: pd.DataFrame,
    image_dir: Path,
    output_path: Path,
    title: str,
    max_examples: int,
):
    if examples.empty:
        return

    examples = examples.head(max_examples)

    cols = 4
    rows = int(np.ceil(len(examples) / cols))

    plt.figure(figsize=(cols * 3, rows * 3))

    for idx, (_, row) in enumerate(examples.iterrows(), start=1):
        img = load_image(image_dir, str(row["filename"]))

        plt.subplot(rows, cols, idx)

        if img is not None:
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, "Image not found", ha="center", va="center")

        plt.title(
            f'{row["filename"]}\ntrue={row["y_true"]}, pred={row["y_pred"]}',
            fontsize=8,
        )
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_easy_hard_examples(
    model_name: str,
    df: pd.DataFrame,
    image_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    max_examples: int,
):
    easy, hard = select_easy_hard_examples(df, max_examples=max_examples)

    easy_path = tables_dir / f"easy_correct_examples_{model_name}.csv"
    hard_path = tables_dir / f"hard_wrong_examples_{model_name}.csv"

    easy.to_csv(easy_path, index=False)
    hard.to_csv(hard_path, index=False)

    plot_example_grid(
        easy,
        image_dir,
        figures_dir / f"easy_correct_examples_{model_name}.png",
        title=f"Easy Correct Examples - {model_name}",
        max_examples=max_examples,
    )

    plot_example_grid(
        hard,
        image_dir,
        figures_dir / f"hard_wrong_examples_{model_name}.png",
        title=f"Hard Wrong Examples - {model_name}",
        max_examples=max_examples,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate classical leaf classification models."
    )

    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path(r"C:\AIA_workspace\data\processed\predictions\classical"),
        help="Directory containing *_predictions.csv files.",
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(r"C:\AIA_workspace\data\raw\Leaves"),
        help="Directory containing original leaf images.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(r"C:\AIA_workspace\results"),
        help="Directory where figures and tables will be saved.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=12,
        help="Number of easy/hard examples to visualize.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    predictions_dir = args.predictions_dir
    image_dir = args.image_dir
    results_dir = args.results_dir

    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = find_prediction_files(predictions_dir)

    print("Prediction files found:")
    for p in prediction_files:
        print(" -", p)

    summary_records = []
    prediction_dfs = {}

    for prediction_path in tqdm(prediction_files, desc="Evaluating models", unit="model"):
        record, df = evaluate_one_model(
            prediction_path=prediction_path,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
        )

        summary_records.append(record)
        prediction_dfs[record["model"]] = df

    summary_df = pd.DataFrame(summary_records)
    summary_path = tables_dir / "classical_evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_overall_accuracy(
        summary_df,
        figures_dir / "overall_accuracy_comparison.png",
    )

    for model_name, df in tqdm(prediction_dfs.items(), desc="Plotting examples", unit="model"):
        save_easy_hard_examples(
            model_name=model_name,
            df=df,
            image_dir=image_dir,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            max_examples=args.max_examples,
        )

    print("\nEvaluation finished.")
    print("Summary file:", summary_path)
    print("Tables saved to:", tables_dir)
    print("Figures saved to:", figures_dir)

    print("\nOverall accuracy:")
    for _, row in summary_df.iterrows():
        print(f'{row["model"]}: {row["overall_accuracy"]:.4f}')


if __name__ == "__main__":
    main()
