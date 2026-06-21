# AIA Topic A: Leaf Species Recognition

## 1. 项目介绍

本项目是 Automatic Image Analysis 课程 Topic A：Leaf Species Recognition。

项目目标是使用 Flavia leaf dataset，对叶片图像进行 32 类 species classification。每张图像的类别由 Flavia 数据集的 image ID 范围确定，例如 `1001.jpg`、`1002.jpg` 等文件名会映射到对应的 leaf species label。

本项目包含两条分类路线：

```text
Classical pipeline:
binary leaf mask / contour
-> Hu moments + Fourier descriptors
-> Gaussian Naive Bayes / k-NN
-> evaluation

Modern pipeline:
cropped leaf image
-> frozen MobileNetV2 feature extractor
-> Linear SVM classifier
-> evaluation
```

Classical pipeline 主要用于传统形状特征 baseline；modern pipeline 用于比较 pretrained deep visual features 的表现。

---

## 2. 项目文件夹结构

推荐项目结构如下：

```text
D:\AIA_workspace\
│
├── README.md
├── requirements.txt
│
├── src\
│   ├── preprocessing.py
│   │
│   ├── features_hu.py
│   ├── features_fourier.py
│   ├── build_features.py
│   ├── split.py
│   ├── train.py
│   ├── evaluate.py
│   │
│   ├── split_modern.py
│   ├── mobilenet_features.py
│   └── train_mobilenet_svm.py
│
├── data\
│   ├── raw\
│   │   └── Leaves\
│   │       ├── 1001.jpg
│   │       ├── 1002.jpg
│   │       └── ...
│   │
│   └── processed\
│       ├── masks\
│       │   ├── 1001_mask.png
│       │   ├── 1002_mask.png
│       │   └── ...
│       │
│       ├── cropped\
│       │   ├── 1001_crop.png
│       │   ├── 1002_crop.png
│       │   └── ...
│       │
│       ├── contours\
│       │   ├── 1001_contour.png
│       │   ├── 1002_contour.png
│       │   └── ...
│       │
│       ├── preprocess_summary.csv
│       ├── failed_files.txt
│       │
│       ├── splits\
│       │   ├── split_70_30_seed0.csv
│       │   ├── train_features_70_30_seed0.csv
│       │   ├── test_features_70_30_seed0.csv
│       │   ├── class_counts_70_30_seed0.csv
│       │   ├── split_summary_70_30_seed0.txt
│       │   │
│       │   ├── modern_split_70_30_seed0.csv
│       │   ├── modern_train.csv
│       │   ├── modern_test.csv
│       │   └── modern_class_counts_70_30_seed0.csv
│       │
│       ├── features\
│       │   ├── hu_moments_features.csv
│       │   ├── fourier_features.csv
│       │   ├── classical_features.csv
│       │   ├── classical_feature_columns.txt
│       │   ├── build_features_summary.txt
│       │   │
│       │   ├── mobilenet_train_features.csv
│       │   └── mobilenet_test_features.csv
│       │
│       └── predictions\
│           ├── classical\
│           │   ├── gaussian_nb_predictions.csv
│           │   └── knn_k5_predictions.csv
│           │
│           └── modern\
│               └── mobilenet_svm_predictions.csv
│
├── models\
│   ├── classical\
│   │   ├── gaussian_nb.joblib
│   │   ├── knn_k5.joblib
│   │   ├── gaussian_nb_config.json
│   │   └── knn_k5_config.json
│   │
│   └── modern\
│       ├── mobilenet_linear_svm.joblib
│       └── mobilenet_linear_svm_config.json
│
└── results\
    ├── tables\
    │   ├── classical_training_summary.csv
    │   ├── classical_evaluation_summary.csv
    │   ├── classification_report_gaussian_nb.csv
    │   ├── classification_report_knn_k5.csv
    │   ├── confusion_matrix_gaussian_nb.csv
    │   ├── confusion_matrix_knn_k5.csv
    │   ├── confusion_matrix_normalized_gaussian_nb.csv
    │   ├── confusion_matrix_normalized_knn_k5.csv
    │   ├── per_class_accuracy_gaussian_nb.csv
    │   ├── per_class_accuracy_knn_k5.csv
    │   ├── easy_correct_examples_gaussian_nb.csv
    │   ├── hard_wrong_examples_gaussian_nb.csv
    │   ├── easy_correct_examples_knn_k5.csv
    │   ├── hard_wrong_examples_knn_k5.csv
    │   └── used_classical_feature_columns.txt
    │
    ├── figures\
    │   ├── confusion_matrix_gaussian_nb.png
    │   ├── confusion_matrix_knn_k5.png
    │   ├── confusion_matrix_normalized_gaussian_nb.png
    │   ├── confusion_matrix_normalized_knn_k5.png
    │   ├── per_class_accuracy_gaussian_nb.png
    │   ├── per_class_accuracy_knn_k5.png
    │   ├── easy_correct_examples_gaussian_nb.png
    │   ├── hard_wrong_examples_gaussian_nb.png
    │   ├── easy_correct_examples_knn_k5.png
    │   ├── hard_wrong_examples_knn_k5.png
    │   └── overall_accuracy_comparison.png
    │
    └── modern\
        ├── tables\
        │   ├── modern_training_summary.csv
        │   ├── classification_report_mobilenet_svm.csv
        │   ├── confusion_matrix_mobilenet_svm.csv
        │   ├── confusion_matrix_normalized_mobilenet_svm.csv
        │   ├── per_class_accuracy_mobilenet_svm.csv
        │   ├── easy_correct_examples_mobilenet_svm.csv
        │   └── hard_wrong_examples_mobilenet_svm.csv
        │
        └── figures\
            ├── confusion_matrix_mobilenet_svm.png
            ├── confusion_matrix_normalized_mobilenet_svm.png
            ├── per_class_accuracy_mobilenet_svm.png
            ├── overall_accuracy_mobilenet_svm.png
            ├── easy_correct_examples_mobilenet_svm.png
            └── hard_wrong_examples_mobilenet_svm.png
```

说明：

- `data/raw/Leaves/` 保存原始 Flavia leaf images。
- `data/processed/` 保存 preprocessing、splits、features 和 predictions。
- `models/classical/` 保存 classical models。
- `models/modern/` 保存 modern Linear SVM model。
- `results/tables/` 和 `results/figures/` 保存 classical pipeline 结果。
- `results/modern/tables/` 和 `results/modern/figures/` 保存 modern pipeline 结果。
- `data/` 被 `.gitignore` 忽略，因此 raw images 和中间 feature/prediction files 默认不会被 push。

---

## 3. 环境设置

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

如果需要使用 GPU 版本 PyTorch，可以使用 CUDA wheel index：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

检查 CUDA 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## 4. Classical Pipeline 及运行方式

Classical pipeline 使用手工设计的 shape descriptors 进行分类。

整体流程：

```text
raw leaf image
-> preprocessing
-> binary mask / contour
-> Hu moments
-> Fourier descriptors
-> feature concatenation
-> stratified 70/30 train/test split
-> Gaussian Naive Bayes / k-NN
-> evaluation
```

### Step 1: Preprocessing

脚本：

```text
src/preprocessing.py
```

功能：

- 读取 `data/raw/Leaves/` 中的 raw leaf images
- 使用 Otsu threshold 分割叶片
- 使用 morphology open / close 去噪和补洞
- 提取最大外部轮廓
- 保存 binary masks
- 保存 cropped leaf images
- 保存 contour checking images
- 保存 preprocessing summary

运行：

```powershell
python src\preprocessing.py
```

输出：

```text
data/processed/masks/
data/processed/cropped/
data/processed/contours/
data/processed/preprocess_summary.csv
data/processed/failed_files.txt
```

### Step 2: Hu Moments Feature Extraction

脚本：

```text
src/features_hu.py
```

功能：

- 读取 `data/processed/masks/`
- 对每张 binary leaf mask 计算 7 个 Hu moments
- 使用 log transform 稳定数值范围
- 保存 Hu moments feature table

运行：

```powershell
python src\features_hu.py
```

输出：

```text
data/processed/features/hu_moments_features.csv
```

### Step 3: Fourier Descriptors Feature Extraction

脚本：

```text
src/features_fourier.py
```

功能：

- 从 leaf mask 或 boundary 中提取叶片边界
- 统一边界方向
- 对齐起始点
- 重采样到固定长度
- 中心化到 centroid
- 尺度归一化
- 使用 FFT 计算 Fourier descriptors
- 保留 low-frequency coefficients

运行：

```powershell
python src\features_fourier.py --num-descriptors 20
```

输出：

```text
data/processed/features/fourier_features.csv
```

### Step 4: Build Classical Feature Table

脚本：

```text
src/build_features.py
```

功能：

- 读取 `hu_moments_features.csv`
- 读取 `fourier_features.csv`
- 按 `image_id` 合并
- 得到 classical feature vector

运行：

```powershell
python src\build_features.py
```

输出：

```text
data/processed/features/classical_features.csv
data/processed/features/classical_feature_columns.txt
data/processed/features/build_features_summary.txt
```

### Step 5: Train/Test Split

脚本：

```text
src/split.py
```

功能：

- 读取 `classical_features.csv`
- 使用 stratified train/test split
- `test_size = 0.3`
- `random_state = 0`

运行：

```powershell
python src\split.py
```

输出：

```text
data/processed/splits/split_70_30_seed0.csv
data/processed/splits/train_features_70_30_seed0.csv
data/processed/splits/test_features_70_30_seed0.csv
data/processed/splits/class_counts_70_30_seed0.csv
data/processed/splits/split_summary_70_30_seed0.txt
```

### Step 6: Train Classical Models

脚本：

```text
src/train.py
```

功能：

- 读取 train/test feature CSV
- 使用 `classical_feature_columns.txt` 确定特征列
- 训练 Gaussian Naive Bayes
- 训练 k-NN，默认 `k = 5`
- 保存 model 和 prediction CSV

运行：

```powershell
python src\train.py
```

如果需要改变 k 值：

```powershell
python src\train.py --knn-k 3
```

输出：

```text
models/classical/gaussian_nb.joblib
models/classical/knn_k5.joblib
data/processed/predictions/classical/gaussian_nb_predictions.csv
data/processed/predictions/classical/knn_k5_predictions.csv
results/tables/classical_training_summary.csv
```

### Step 7: Classical Evaluation

脚本：

```text
src/evaluate.py
```

功能：

- 读取 classical prediction CSV
- 计算 overall accuracy
- 计算 per-class accuracy
- 保存 classification report
- 保存 confusion matrix 和 normalized confusion matrix
- 绘制 per-class accuracy plot
- 绘制 overall accuracy comparison plot
- 保存 easy correct examples 和 hard wrong examples

运行：

```powershell
python src\evaluate.py
```

输出：

```text
results/tables/
results/figures/
```

Classical pipeline 完整运行顺序：

```powershell
python src\preprocessing.py
python src\features_hu.py
python src\features_fourier.py --num-descriptors 20
python src\build_features.py
python src\split.py
python src\train.py
python src\evaluate.py
```

当前已有 classical baseline 结果：

```text
GaussianNB test accuracy: 0.7522
k-NN k=5 test accuracy: 0.7190
```

---

## 5. Modern Pipeline 及运行方式

Modern pipeline 使用 pretrained MobileNetV2 提取图像特征，然后使用 linear SVM style classifier 进行分类。

整体流程：

```text
raw leaf image
-> preprocessing
-> cropped leaf image
-> resize to 224 x 224
-> frozen MobileNetV2 feature extractor
-> 1280-dimensional feature vector
-> linear SVM classifier
-> evaluation
```

说明：

- MobileNetV2 使用 pretrained weights。
- MobileNetV2 是 frozen feature extractor，不进行 fine-tuning。
- 每张 cropped leaf image 会被转换为 1280 维 feature vector。
- 分类器使用 scikit-learn 的 `SGDClassifier(loss="hinge")`。
- `SGDClassifier(loss="hinge")` 是 linear SVM style classifier，比 `LinearSVC` 在本项目中训练更快。
- batch size 主要影响速度和 GPU memory，不应该明显影响分类结果。

### Step 1: Preprocessing

如果还没有运行 preprocessing，先运行：

```powershell
python src\preprocessing.py
```

modern pipeline 主要使用：

```text
data/processed/cropped/
```

### Step 2: Modern Train/Test Split

脚本：

```text
src/split_modern.py
```

功能：

- 读取 `data/processed/cropped/`
- 从文件名提取 image ID
- 根据 Flavia image ID 范围生成 label
- 创建 stratified 70/30 split
- `random_state = 0`

运行：

```powershell
python src\split_modern.py
```

输出：

```text
data/processed/splits/modern_train.csv
data/processed/splits/modern_test.csv
data/processed/splits/modern_split_70_30_seed0.csv
data/processed/splits/modern_class_counts_70_30_seed0.csv
```

当前 split：

```text
training samples: 1334
test samples: 573
classes: 32
```

### Step 3: MobileNetV2 Feature Extraction

脚本：

```text
src/mobilenet_features.py
```

功能：

- 读取 modern train/test CSV
- 加载 cropped leaf images
- resize 到 `224 x 224`
- 使用 pretrained MobileNetV2 提取 frozen features
- 保存 train/test feature CSV

使用 GPU 运行：

```powershell
python src\mobilenet_features.py --device cuda --batch-size 8
```

如果 CUDA out of memory，降低 batch size：

```powershell
python src\mobilenet_features.py --device cuda --batch-size 4
```

如果需要 CPU：

```powershell
python src\mobilenet_features.py --device cpu --batch-size 8
```

输出：

```text
data/processed/features/mobilenet_train_features.csv
data/processed/features/mobilenet_test_features.csv
```

### Step 4: Train Linear SVM and Evaluate

脚本：

```text
src/train_mobilenet_svm.py
```

功能：

- 读取 MobileNetV2 feature CSV
- 使用 `StandardScaler`
- 训练 hinge-loss linear SVM classifier
- 保存 trained model
- 保存 prediction CSV
- 保存 summary、classification report、confusion matrix、per-class accuracy
- 保存 easy correct examples 和 hard wrong examples

运行：

```powershell
python src\train_mobilenet_svm.py
```

模型输出：

```text
models/modern/mobilenet_linear_svm.joblib
models/modern/mobilenet_linear_svm_config.json
```

预测输出：

```text
data/processed/predictions/modern/mobilenet_svm_predictions.csv
```

Modern result tables：

```text
results/modern/tables/modern_training_summary.csv
results/modern/tables/classification_report_mobilenet_svm.csv
results/modern/tables/confusion_matrix_mobilenet_svm.csv
results/modern/tables/confusion_matrix_normalized_mobilenet_svm.csv
results/modern/tables/per_class_accuracy_mobilenet_svm.csv
results/modern/tables/easy_correct_examples_mobilenet_svm.csv
results/modern/tables/hard_wrong_examples_mobilenet_svm.csv
```

Modern result figures：

```text
results/modern/figures/confusion_matrix_mobilenet_svm.png
results/modern/figures/confusion_matrix_normalized_mobilenet_svm.png
results/modern/figures/per_class_accuracy_mobilenet_svm.png
results/modern/figures/overall_accuracy_mobilenet_svm.png
results/modern/figures/easy_correct_examples_mobilenet_svm.png
results/modern/figures/hard_wrong_examples_mobilenet_svm.png
```

Modern pipeline 完整运行顺序：

```powershell
python src\preprocessing.py
python src\split_modern.py
python src\mobilenet_features.py --device cuda --batch-size 8
python src\train_mobilenet_svm.py
```

如果 preprocessing 和 split 已经完成，则只需要运行：

```powershell
python src\mobilenet_features.py --device cuda --batch-size 8
python src\train_mobilenet_svm.py
```

当前 modern pipeline 结果：

```text
train accuracy: 1.0000
test accuracy: 0.9895
```

### Step 5: Modern Pipeline 额外验证

为了检查 `0.9895` 的 test accuracy 是否只是某一次 split 的偶然结果，本项目额外加入了 two validation scripts。

#### Multi-seed Validation

脚本：

```text
src/validate_modern_seeds.py
```

功能：

- 使用已经提取好的 MobileNetV2 feature CSV
- 将 train/test features 合并回完整 feature table
- 使用不同 random seed 重新做 stratified 70/30 split
- 每个 seed 重新训练 linear SVM classifier
- 比较不同 seed 下的 test accuracy
- 不覆盖原本的 `results/modern/` 和 `models/modern/`

运行：

```powershell
python src\validate_modern_seeds.py --seeds 0 1 2 3 4
```

输出：

```text
results/modern_validation/seeds/seed_validation_summary.csv
results/modern_validation/seeds/seed_validation_aggregate.csv
results/modern_validation/seeds/seed_0/
results/modern_validation/seeds/seed_1/
results/modern_validation/seeds/seed_2/
results/modern_validation/seeds/seed_3/
results/modern_validation/seeds/seed_4/
```

当前 multi-seed validation 结果：

```text
seed 0: 0.9913
seed 1: 0.9756
seed 2: 0.9825
seed 3: 0.9895
seed 4: 0.9843

mean test accuracy: 0.9846
std test accuracy: 0.0062
min test accuracy: 0.9756
max test accuracy: 0.9913
```

#### Stratified 5-Fold Cross-Validation

脚本：

```text
src/cross_validate_modern.py
```

功能：

- 使用完整 MobileNetV2 feature table
- 使用 `StratifiedKFold`
- 每个 fold 重新训练 linear SVM classifier
- 计算每个 fold 的 test accuracy
- 汇总 mean / std / min / max accuracy
- 不覆盖原本的 `results/modern/` 和 `models/modern/`

运行：

```powershell
python src\cross_validate_modern.py --folds 5 --seed 0
```

输出：

```text
results/modern_validation/cross_validation/cross_validation_summary.csv
results/modern_validation/cross_validation/cross_validation_aggregate.csv
results/modern_validation/cross_validation/fold_1/
results/modern_validation/cross_validation/fold_2/
results/modern_validation/cross_validation/fold_3/
results/modern_validation/cross_validation/fold_4/
results/modern_validation/cross_validation/fold_5/
```

当前 5-fold cross-validation 结果：

```text
fold 1: 0.9869
fold 2: 0.9738
fold 3: 0.9843
fold 4: 0.9869
fold 5: 0.9816

mean test accuracy: 0.9827
std test accuracy: 0.0054
min test accuracy: 0.9738
max test accuracy: 0.9869
```

验证结论：

```text
modern pipeline 的准确率在不同 random seeds 和 5-fold cross-validation 下都保持在约 97% 到 99%。
因此，原本 seed 0 split 上的 0.9895 test accuracy 不是明显的偶然结果。
```

---

## 6. Report Notes

### Classical Pipeline 报告要点

- 本任务是 32 类 leaf species classification。
- Classical pipeline 只使用 shape information，不使用 deep learning features。
- Hu moments 描述叶片区域的整体几何分布。
- Fourier descriptors 描述叶片边界形状。
- Classical feature vector 当前由 7 个 Hu moments 和 20 个 Fourier descriptors 组成，共 27 维。
- 使用 stratified 70/30 train/test split。
- 当前 classical baseline 使用 `random_state = 0`。
- GaussianNB 假设特征之间条件独立，这个假设对 Hu + Fourier descriptors 不一定完全成立。
- k-NN 对特征尺度敏感，因此训练时需要 standard scaler。
- k-NN 当前保存结果使用 `k = 5`。
- confusion matrix 可以显示哪些 species 容易混淆。
- easy correct / hard wrong examples 可以帮助解释 shape descriptor 的局限性。

### Modern Pipeline 报告要点

- Modern pipeline 使用 pretrained MobileNetV2 作为 frozen feature extractor。
- MobileNetV2 没有进行 fine-tuning，因此训练成本较低，结果也更容易复现。
- 每张 cropped leaf image 被 resize 到 `224 x 224`。
- MobileNetV2 为每张图生成 1280 维 feature vector。
- 最终分类器是 hinge-loss linear SVM classifier。
- Modern pipeline 和 classical pipeline 使用相同思想的 stratified 70/30 split。
- 当前 modern split 使用 `random_state = 0`。
- Modern result 明显高于 classical baseline，说明 pretrained deep visual features 对该数据集更有效。
- 当前 modern test accuracy 很高，应说明这是在 clean Flavia dataset 和 stratified random split 上得到的结果。
- Multi-seed validation 的 mean test accuracy 为 `0.9846`，说明结果在不同 random splits 下较稳定。
- 5-fold stratified cross-validation 的 mean test accuracy 为 `0.9827`，进一步支持 modern pipeline 的稳定性。
- 高准确率不一定代表对真实复杂场景完全泛化，因为 Flavia 图像背景较干净、叶片居中且类别内图像可能相似。
- 如果需要更严格验证，可以尝试 multiple random seeds 或 stratified cross-validation。
- batch size 只影响 MobileNetV2 feature extraction 的速度和 GPU memory，不应该明显改变最终结果。
