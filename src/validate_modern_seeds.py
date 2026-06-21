import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = WORKSPACE_DIR / "data" / "processed" / "features"
OUTPUT_DIR = WORKSPACE_DIR / "results" / "modern_validation" / "seeds"


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


def run_one_seed(
    df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    test_size: float,
    max_iter: int,
) -> dict:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )

    x_train = train_df[feature_columns]
    y_train = train_df["label"]
    x_test = test_df[feature_columns]
    y_test = test_df["label"]

    model = make_model(max_iter=max_iter)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    seed_dir = OUTPUT_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = test_df[["image_id", "filename", "label"]].copy()
    prediction_df["prediction"] = test_pred
    prediction_df["correct"] = prediction_df["label"] == prediction_df["prediction"]
    prediction_df.to_csv(seed_dir / "predictions.csv", index=False)

    report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(seed_dir / "classification_report.csv")

    joblib.dump(model, seed_dir / "model.joblib")

    return {
        "seed": seed,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "feature_count": len(feature_columns),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "correct": int(prediction_df["correct"].sum()),
        "wrong": int((~prediction_df["correct"]).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate modern pipeline with multiple random seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_columns = load_all_features()

    print("Modern multi-seed validation")
    print(f"Total samples: {len(df)}")
    print(f"Classes: {df['label'].nunique()}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Seeds: {args.seeds}")

    rows = []
    for seed in args.seeds:
        print(f"Running seed {seed}...")
        rows.append(
            run_one_seed(
                df=df,
                feature_columns=feature_columns,
                seed=seed,
                test_size=args.test_size,
                max_iter=args.max_iter,
            )
        )

    summary_df = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "seed_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    aggregate = pd.DataFrame(
        [
            {
                "num_runs": len(summary_df),
                "mean_test_accuracy": summary_df["test_accuracy"].mean(),
                "std_test_accuracy": summary_df["test_accuracy"].std(ddof=1),
                "min_test_accuracy": summary_df["test_accuracy"].min(),
                "max_test_accuracy": summary_df["test_accuracy"].max(),
            }
        ]
    )
    aggregate_path = OUTPUT_DIR / "seed_validation_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    print("Validation finished.")
    print(f"Summary saved to: {summary_path}")
    print(f"Aggregate saved to: {aggregate_path}")
    print(summary_df[["seed", "test_accuracy", "correct", "wrong"]].to_string(index=False))


if __name__ == "__main__":
    main()
