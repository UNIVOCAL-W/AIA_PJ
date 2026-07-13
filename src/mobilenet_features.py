import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = WORKSPACE_DIR / "data" / "processed" / "splits"
FEATURE_DIR = WORKSPACE_DIR / "data" / "processed" / "features"


class LeafCropDataset(Dataset):
    def __init__(self, csv_path: Path, transform) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["crop_path"]).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "image_id": str(row["image_id"]),
            "filename": row["filename"],
            "label": int(row["label"]),
        }


def build_model(device: torch.device):
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model, weights


def extract_features(csv_path: Path, output_path: Path, device: torch.device, batch_size: int) -> None:
    model, weights = build_model(device)
    transform = weights.transforms()

    dataset = LeafCropDataset(csv_path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    rows = []

    print(f"Start extracting features: {csv_path.name}")
    print(f"Number of samples: {len(dataset)}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting MobileNetV2 features"):
            images = batch["image"].to(device)
            features = model(images).cpu().numpy()

            for i in range(features.shape[0]):
                feature_values = features[i].astype(np.float32)
                row = {
                    "image_id": batch["image_id"][i],
                    "filename": batch["filename"][i],
                    "label": int(batch["label"][i]),
                }

                row.update(
                    {f"feat_{index}": float(value) for index, value in enumerate(feature_values)}
                )

                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Feature dimension: {len([c for c in rows[0].keys() if c.startswith('feat_')])}")
    print(f"Features saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract leaf image features with MobileNetV2.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size. GTX 1050 Ti users can start with 8.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Runtime device. The default auto setting uses CUDA first when available.",
    )
    return parser.parse_args()


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; switching to CPU.")
        device_name = "cpu"

    return torch.device(device_name)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    train_csv = SPLIT_DIR / "modern_train.csv"
    test_csv = SPLIT_DIR / "modern_test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Run this first: python src\\split_modern.py")

    print("MobileNetV2 feature extraction started")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")

    extract_features(
        train_csv,
        FEATURE_DIR / "mobilenet_train_features.csv",
        device,
        args.batch_size,
    )
    extract_features(
        test_csv,
        FEATURE_DIR / "mobilenet_test_features.csv",
        device,
        args.batch_size,
    )

    print("MobileNetV2 feature extraction finished")


if __name__ == "__main__":
    main()
