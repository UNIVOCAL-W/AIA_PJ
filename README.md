# Topic A - Shape and Contour: Leaf Species Recognition

## Task

Dataset: Flavia Leaf Dataset, 1,907 leaf images, 32 plant species. Alternative: Leafsnap.

Core task: Given a segmented leaf silhouette, classify the plant species.

Main question: How far can shape alone go, and when do learned image features add useful information?

Required work:
- Classical pipeline: Fourier descriptors of the normalized boundary and Hu moments.
- Classifiers: naive Bayes and k-NN on a stratified train/test split.
- Modern pipeline: frozen MobileNetV2 features with a linear SVM.
- Compare all methods on the same split.

Expected outputs: confusion matrices, per-class accuracy, easy and hard example leaves, and a short discussion of boundary information.

Theory focus: Fourier descriptor invariance, Hu moments, and the independence assumption in naive Bayes.

## Ideal File Structure

```text
FINAL_AIA/
  README.md
  requirements.txt

  src/
    new_preprocessing.py
    features_hu.py
    features_fourier.py
    build_features.py
    split.py
    train.py
    evaluate.py
    split_modern.py
    mobilenet_features.py
    train_mobilenet_svm.py
    validate_modern_seeds.py
    cross_validate_modern.py

  data/
    raw/
      Leaves/
        1001.jpg
        1002.jpg
        ...
      standardleaves/

    processed/
      masks/
        1001_mask.png
        ...
      boundaries/
        1001_boundary.npy
        ...
      cropped/
        1001_crop.png
        ...
      cropped_preview/
      contours_preview/
      preprocess_summary.csv
      failed_files.txt

      features/
        hu_moments_features.csv
        fourier_features.csv
        classical_features.csv
        classical_feature_columns.txt
        mobilenet_train_features.csv
        mobilenet_test_features.csv

      splits/
        

      predictions/
        classical/
          gaussian_nb_predictions.csv
          knn_k5_predictions.csv
        modern/
          mobilenet_svm_predictions.csv

  models/
    classical/
      gaussian_nb.joblib
      knn_k5.joblib
    modern/
      mobilenet_linear_svm.joblib

  results/
    tables/
    figures/
    modern/
      tables/
      figures/
    modern_validation/
      seeds/
      cross_validation/

  outputs/
  tools/
```

## Code Run Order

Run commands from the repository root.

### Classical Shape Pipeline

```powershell
python src\new_preprocessing.py --input-dir data\raw\Leaves --output-dir data\processed
python src\features_hu.py --processed-dir data\processed
python src\features_fourier.py --processed-dir data\processed --num-descriptors 20
python src\build_features.py --processed-dir data\processed
python src\split.py --features-csv data\processed\features\classical_features.csv --output-dir data\processed\splits
python src\train.py --workspace-dir .
python src\evaluate.py --predictions-dir data\processed\predictions\classical --image-dir data\raw\Leaves --results-dir results
```

### Modern Learned-Feature Pipeline

Before this pipeline, `data/processed/cropped/` should contain one cropped leaf image per sample.

```powershell
python src\split_modern.py
python src\mobilenet_features.py --device auto --batch-size 8
python src\train_mobilenet_svm.py
```

### Modern Validation

```powershell
python src\validate_modern_seeds.py --seeds 0 1 2 3 4
python src\cross_validate_modern.py --folds 5 --seed 0
```
