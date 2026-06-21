import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = WORKSPACE_DIR / "data" / "processed" / "features"
OUTPUT_DIR = WORKSPACE_DIR / "results" / "modern_validation" / "cross_validation"


def load_all_features() -> tuple[pd.DataFrame, list[str]]:
    train_path = FEATURE_DIR / "mobilenet_train_features.csv"
    test_path = FEATURE_DIR / "mobilenet_test_features.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "MobileNet feature files not found. Run src\\mobilenet_features.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates(subset=["image_id"]).sort_values("image_id").reset_index(drop=True)

    feature_columns = [col for col in df.columns if col.startswith("feat_")]
    if not feature_columns:
        raise ValueError("No MobileNet feature columns found.")

    return df, feature_columns


def make_model(max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SGDClassifier(
                    loss="hinge",
                    alpha=1e-4,
                    max_iter=max_iter,
                    tol=1e-3,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stratified cross-validation for the modern pipeline.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_columns = load_all_features()

    x = df[feature_columns]
    y = df["label"]

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    print("Modern stratified cross-validation")
    print(f"Total samples: {len(df)}")
    print(f"Classes: {df['label'].nunique()}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Folds: {args.folds}")

    rows = []

    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y), start=1):
        print(f"Running fold {fold_index}...")

        train_df = df.iloc[train_index].copy()
        test_df = df.iloc[test_index].copy()

        model = make_model(max_iter=args.max_iter)
        model.fit(train_df[feature_columns], train_df["label"])

        train_pred = model.predict(train_df[feature_columns])
        test_pred = model.predict(test_df[feature_columns])

        train_accuracy = accuracy_score(train_df["label"], train_pred)
        test_accuracy = accuracy_score(test_df["label"], test_pred)

        fold_dir = OUTPUT_DIR / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        prediction_df = test_df[["image_id", "filename", "label"]].copy()
        prediction_df["prediction"] = test_pred
        prediction_df["correct"] = prediction_df["label"] == prediction_df["prediction"]
        prediction_df.to_csv(fold_dir / "predictions.csv", index=False)

        report = classification_report(test_df["label"], test_pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(fold_dir / "classification_report.csv")

        joblib.dump(model, fold_dir / "model.joblib")

        rows.append(
            {
                "fold": fold_index,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "feature_count": len(feature_columns),
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "correct": int(prediction_df["correct"].sum()),
                "wrong": int((~prediction_df["correct"]).sum()),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "cross_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    aggregate = pd.DataFrame(
        [
            {
                "num_folds": len(summary_df),
                "mean_test_accuracy": summary_df["test_accuracy"].mean(),
                "std_test_accuracy": summary_df["test_accuracy"].std(ddof=1),
                "min_test_accuracy": summary_df["test_accuracy"].min(),
                "max_test_accuracy": summary_df["test_accuracy"].max(),
            }
        ]
    )
    aggregate_path = OUTPUT_DIR / "cross_validation_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    print("Cross-validation finished.")
    print(f"Summary saved to: {summary_path}")
    print(f"Aggregate saved to: {aggregate_path}")
    print(summary_df[["fold", "test_accuracy", "correct", "wrong"]].to_string(index=False))


if __name__ == "__main__":
    main()
