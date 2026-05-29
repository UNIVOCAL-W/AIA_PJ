"""
train_classical.py

Train the classical classifiers for Topic A: Leaf Species Recognition.

Inputs by default:
    C:/AIA_workspace/data/processed/splits/train_features_70_30_seed0.csv
    C:/AIA_workspace/data/processed/splits/test_features_70_30_seed0.csv
    C:/AIA_workspace/data/processed/features/classical_feature_columns.txt

Outputs by default:
    C:/AIA_workspace/models/classical/*.joblib
    C:/AIA_workspace/data/processed/predictions/classical/*_predictions.csv
    C:/AIA_workspace/results/tables/classical_training_summary.csv

The code follows the course pipeline:
    Fourier descriptors + Hu moments -> naive Bayes and k-NN -> predictions.

Note about CUDA:
    scikit-learn's GaussianNB and KNeighborsClassifier run on CPU. CUDA is not used here.
    CUDA is more useful later for MobileNetV2 / PyTorch feature extraction.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm.auto import tqdm
except ImportError:  # fallback if tqdm is not installed
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


METADATA_COLUMNS = [
    "filename",
    "image_id",
    "label",
    "scientific_name",
    "common_name",
]


def read_feature_columns(feature_columns_path: Path, train_df: pd.DataFrame) -> List[str]:
    """Read feature column names from txt; if unavailable, infer from column prefixes."""
    if feature_columns_path.exists():
        with open(feature_columns_path, "r", encoding="utf-8") as f:
            feature_cols = [line.strip() for line in f if line.strip()]

        missing = [col for col in feature_cols if col not in train_df.columns]
        if missing:
            raise ValueError(
                "feature_columns.txt 中有列在 train CSV 中不存在，例如: "
                f"{missing[:10]}"
            )
        return feature_cols

    print(f"[警告] 没有找到特征列文件: {feature_columns_path}")
    print("将自动使用 hu_ / fd_ / fourier_ / fft_ 开头的列作为特征。")
    feature_cols = [
        col for col in train_df.columns
        if col.startswith(("hu_", "fd_", "fourier_", "fft_"))
    ]
    if not feature_cols:
        raise ValueError("没有找到特征列。请检查 classical_features / train_features 文件。")
    return feature_cols


def load_train_test(
    train_csv: Path,
    test_csv: Path,
    feature_columns_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if not train_csv.exists():
        raise FileNotFoundError(f"训练集文件不存在: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"测试集文件不存在: {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("train/test CSV 中必须包含 label 列。")

    feature_cols = read_feature_columns(feature_columns_path, train_df)

    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    if train_df[feature_cols].isna().any().any():
        raise ValueError("训练集特征中存在 NaN，请检查特征提取或合并步骤。")
    if test_df[feature_cols].isna().any().any():
        raise ValueError("测试集特征中存在 NaN，请检查特征提取或合并步骤。")

    return train_df, test_df, feature_cols


def make_models(knn_k: int, scale_nb: bool = True) -> Dict[str, Pipeline]:
    """Create sklearn pipelines. Scalers are fit only on training data."""
    if scale_nb:
        nb_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ])
    else:
        nb_model = Pipeline([
            ("classifier", GaussianNB()),
        ])

    knn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=knn_k)),
    ])

    return {
        "gaussian_nb": nb_model,
        f"knn_k{knn_k}": knn_model,
    }


def predict_with_confidence(model: Pipeline, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return predictions and max predicted probability if available."""
    y_pred = model.predict(X)

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidence = np.max(proba, axis=1)

    return y_pred, confidence


def build_prediction_table(
    base_df: pd.DataFrame,
    y_pred: np.ndarray,
    confidence: Optional[np.ndarray],
    model_name: str,
) -> pd.DataFrame:
    keep_cols = [col for col in METADATA_COLUMNS if col in base_df.columns]
    pred_df = base_df[keep_cols].copy()

    pred_df["model"] = model_name
    pred_df["y_true"] = base_df["label"].values
    pred_df["y_pred"] = y_pred
    pred_df["correct"] = pred_df["y_true"].astype(str) == pred_df["y_pred"].astype(str)

    if confidence is not None:
        pred_df["confidence"] = confidence
    else:
        pred_df["confidence"] = np.nan

    return pred_df


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def check_cuda_if_requested(check_cuda: bool) -> None:
    if not check_cuda:
        return

    print("\nCUDA 检查")
    print("说明：当前 classical sklearn 模型不使用 CUDA；这个检查只是确认环境。")
    try:
        import torch
        available = torch.cuda.is_available()
        print("torch.cuda.is_available():", available)
        if available:
            print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("没有成功导入/检查 PyTorch CUDA:", exc)


def train_and_save(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    output_model_dir: Path,
    output_predictions_dir: Path,
    output_tables_dir: Path,
    knn_k: int,
    scale_nb: bool,
) -> pd.DataFrame:
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_predictions_dir.mkdir(parents=True, exist_ok=True)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df["label"].to_numpy()
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    y_test = test_df["label"].to_numpy()

    models = make_models(knn_k=knn_k, scale_nb=scale_nb)
    summary_records = []

    for model_name, model in tqdm(models.items(), desc="Training classical models", unit="model"):
        print(f"\n正在训练: {model_name}")
        start_time = time.time()

        model.fit(X_train, y_train)

        train_pred, _ = predict_with_confidence(model, X_train)
        test_pred, test_conf = predict_with_confidence(model, X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        elapsed = time.time() - start_time

        model_path = output_model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        pred_df = build_prediction_table(
            base_df=test_df,
            y_pred=test_pred,
            confidence=test_conf,
            model_name=model_name,
        )
        pred_path = output_predictions_dir / f"{model_name}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        config = {
            "model_name": model_name,
            "feature_count": len(feature_cols),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "training_time_seconds": float(elapsed),
            "model_path": str(model_path),
            "prediction_path": str(pred_path),
            "knn_k": int(knn_k) if model_name.startswith("knn") else None,
            "standard_scaler": True if (model_name.startswith("knn") or scale_nb) else False,
        }
        save_json(output_model_dir / f"{model_name}_config.json", config)

        summary_records.append(config)

        print(f"完成: {model_name}")
        print(f"  train accuracy = {train_acc:.4f}")
        print(f"  test  accuracy = {test_acc:.4f}")
        print(f"  model saved to = {model_path}")
        print(f"  predictions    = {pred_path}")

    summary_df = pd.DataFrame(summary_records)
    summary_path = output_tables_dir / "classical_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    feature_info_path = output_tables_dir / "used_classical_feature_columns.txt"
    with open(feature_info_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")

    print("\n训练全部完成")
    print("训练摘要:", summary_path)
    print("使用的特征列:", feature_info_path)

    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Gaussian Naive Bayes and k-NN on classical leaf features."
    )

    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path(r"C:\AIA_workspace"),
        help=r"Workspace directory. Default: C:\AIA_workspace",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Path to train_features_70_30_seed0.csv.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Path to test_features_70_30_seed0.csv.",
    )
    parser.add_argument(
        "--feature-columns",
        type=Path,
        default=None,
        help="Path to classical_feature_columns.txt.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="k value for k-NN. Default: 5.",
    )
    parser.add_argument(
        "--no-scale-nb",
        action="store_true",
        help="Do not apply StandardScaler before GaussianNB.",
    )
    parser.add_argument(
        "--check-cuda",
        action="store_true",
        help="Print CUDA status. Classical sklearn models still run on CPU.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workspace_dir = args.workspace_dir
    processed_dir = workspace_dir / "data" / "processed"

    train_csv = args.train_csv or processed_dir / "splits" / "train_features_70_30_seed0.csv"
    test_csv = args.test_csv or processed_dir / "splits" / "test_features_70_30_seed0.csv"
    feature_columns_path = args.feature_columns or processed_dir / "features" / "classical_feature_columns.txt"

    output_model_dir = workspace_dir / "models" / "classical"
    output_predictions_dir = processed_dir / "predictions" / "classical"
    output_tables_dir = workspace_dir / "results" / "tables"

    check_cuda_if_requested(args.check_cuda)

    print("输入训练集:", train_csv)
    print("输入测试集:", test_csv)
    print("特征列文件:", feature_columns_path)

    train_df, test_df, feature_cols = load_train_test(
        train_csv=train_csv,
        test_csv=test_csv,
        feature_columns_path=feature_columns_path,
    )

    print("训练样本数:", len(train_df))
    print("测试样本数:", len(test_df))
    print("特征数量:", len(feature_cols))
    print("类别数量:", train_df["label"].nunique())

    train_and_save(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        output_model_dir=output_model_dir,
        output_predictions_dir=output_predictions_dir,
        output_tables_dir=output_tables_dir,
        knn_k=args.knn_k,
        scale_nb=not args.no_scale_nb,
    )


if __name__ == "__main__":
    main()
