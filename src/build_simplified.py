import pandas as pd
from pathlib import Path

features_dir = Path(r"C:\AIA_workspace\data\processed\features")

hu = pd.read_csv(features_dir / "hu_moments_features.csv")
fd = pd.read_csv(features_dir / "fourier_features.csv")

merged = pd.merge(
    hu,
    fd,
    on="image_id",
    how="inner"
)

merged.to_csv(features_dir / "classical_features.csv", index=False)

print("合并完成:", merged.shape)