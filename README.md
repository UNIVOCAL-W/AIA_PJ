# AIA Topic A: Leaf Species Recognition

本项目是 Automatic Image Analysis 课程 Topic A：Leaf Species Recognition。目标是使用 Flavia leaf dataset，通过经典形状特征和分类器完成 32 类叶片 species 分类。

当前 classical pipeline 使用：

- binary leaf silhouette
- ordered boundary points
- Fourier descriptors
- Hu moments
- Gaussian Naive Bayes
- k-NN
- stratified train/test split
- accuracy、per-class accuracy、confusion matrix 评估

---

## 1. Project Structure

推荐项目文件夹结构如下：

```text
C:\AIA_workspace\
│
├── README.md
│
├── src\
│   ├── preprocessing.py
│   ├── features_hu.py
│   ├── features_fourier.py
│   ├── build_features.py
│   ├── split.py
│   ├── train.py
│   └── evaluate.py
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
│       ├── boundaries\
│       │   ├── 1001_boundary.npy
│       │   ├── 1002_boundary.npy
│       │   └── ...
│       │
│       ├── cropped_preview\
│       │   └── 每个类别保存若干张裁剪预览图
│       │
│       ├── contours_preview\
│       │   └── 每个类别保存若干张轮廓检测预览图
│       │
│       ├── features\
│       │   ├── hu_moments_features.csv
│       │   ├── fourier_features.csv
│       │   ├── classical_features.csv
│       │   ├── classical_feature_columns.txt
│       │   └── build_features_summary.txt
│       │
│       ├── splits\
│       │   ├── split_70_30_seed0.csv
│       │   ├── train_features_70_30_seed0.csv
│       │   ├── test_features_70_30_seed0.csv
│       │   ├── class_counts_70_30_seed0.csv
│       │   └── split_summary_70_30_seed0.txt
│       │
│       └── predictions\
│           └── classical\
│               ├── gaussian_nb_predictions.csv
│               └── knn_k5_predictions.csv
│
├── models\
│   └── classical\
│       ├── gaussian_nb.joblib
│       ├── knn_k5.joblib
│       ├── gaussian_nb_config.json
│       └── knn_k5_config.json
│
└── results\
    ├── tables\
    │   ├── classical_training_summary.csv
    │   ├── classical_evaluation_summary.csv
    │   ├── per_class_accuracy_gaussian_nb.csv
    │   ├── per_class_accuracy_knn_k5.csv
    │   ├── confusion_matrix_gaussian_nb.csv
    │   └── confusion_matrix_knn_k5.csv
    │
    └── figures\
        ├── confusion_matrix_gaussian_nb.png
        ├── confusion_matrix_knn_k5.png
        ├── confusion_matrix_normalized_gaussian_nb.png
        ├── confusion_matrix_normalized_knn_k5.png
        ├── per_class_accuracy_gaussian_nb.png
        ├── per_class_accuracy_knn_k5.png
        ├── easy_correct_examples_gaussian_nb.png
        ├── hard_wrong_examples_gaussian_nb.png
        ├── easy_correct_examples_knn_k5.png
        ├── hard_wrong_examples_knn_k5.png
        └── overall_accuracy_comparison.png
```

---

## 2. Environment Setup

建议使用独立 Python 环境：

```bash
conda create -n aia python=3.11
conda activate aia
pip install numpy pandas matplotlib opencv-python scikit-learn tqdm joblib
```

当前 classical pipeline 使用的是 `scikit-learn`，不需要 CUDA。CUDA 更适合后续 MobileNetV2 / PyTorch feature extraction 阶段。

---

## 3. Data Preparation

原始 Flavia 数据集图片放在：

```text
C:\AIA_workspace\data\raw\Leaves
```

图片文件名示例：

```text
1001.jpg
1002.jpg
...
3621.jpg
```

项目中使用图片编号范围生成 32 个 species label。每张图片通过 `image_id` 与 label 和特征表对齐。

---

## 4. Pipeline Overview

### Step 1: Preprocessing

脚本：

```text
src/preprocessing.py
```

功能：

- 读取原始 leaf image
- 转灰度图
- Otsu 二值化
- morphology open / close 去噪和补洞
- 提取最大外部轮廓
- 保存 binary mask
- 保存 ordered boundary points
- 每个类别保存若干张 crop 和 contour preview
- 保存 `preprocess_summary.csv`

运行：

```bash
python C:\AIA_workspace\src\preprocessing.py
```

输出：

```text
data/processed/masks/
data/processed/boundaries/
data/processed/cropped_preview/
data/processed/contours_preview/
data/processed/preprocess_summary.csv
```

---

### Step 2: Hu Moments Feature Extraction

脚本：

```text
src/features_hu.py
```

功能：

- 读取 `masks/`
- 对每张 binary leaf mask 计算 7 个 Hu moments
- 使用 log transform 稳定数值范围
- 保存 Hu moments feature table

运行：

```bash
python C:\AIA_workspace\src\features_hu.py
```

输出：

```text
data/processed/features/hu_moments_features.csv
```

---

### Step 3: Fourier Descriptors Feature Extraction

脚本：

```text
src/features_fourier.py
```

功能：

- 读取 `boundaries/` 中的 ordered boundary points
- 统一边界方向
- 对齐起始点
- 重采样到固定长度
- 中心化到 centroid
- 尺度归一化
- 将边界表示为 complex signal: `z = x + i y`
- 使用 FFT 计算 Fourier descriptors
- 保留 low-frequency coefficients

运行：

```bash
python C:\AIA_workspace\src\features_fourier.py --num-descriptors 20
```

输出：

```text
data/processed/features/fourier_features.csv
```

---

### Step 4: Build Classical Feature Table

脚本：

```text
src/build_features.py
```

功能：

- 读取 `hu_moments_features.csv`
- 读取 `fourier_features.csv`
- 按 `image_id` 合并
- 拼接成每张叶子的 classical feature vector

当前特征维度：

```text
7 Hu moments + 20 Fourier descriptors = 27 features
```

运行：

```bash
python C:\AIA_workspace\src\build_features.py
```

输出：

```text
data/processed/features/classical_features.csv
data/processed/features/classical_feature_columns.txt
data/processed/features/build_features_summary.txt
```

---

### Step 5: Train/Test Split

脚本：

```text
src/split_data.py
```

功能：

- 读取 `classical_features.csv`
- 使用 `train_test_split`
- `test_size = 0.3`
- `stratify = label`
- `random_state = 0`
- 保存固定 train/test split

运行：

```bash
python C:\AIA_workspace\src\split_data.py
```

输出：

```text
data/processed/splits/split_70_30_seed0.csv
data/processed/splits/train_features_70_30_seed0.csv
data/processed/splits/test_features_70_30_seed0.csv
data/processed/splits/class_counts_70_30_seed0.csv
data/processed/splits/split_summary_70_30_seed0.txt
```

说明：

- 任务仍然是 32 类分类
- train/test 是两个数据集合，不是二分类
- stratified split 保证每个类别在 train/test 中比例接近
- seed0 保证实验可复现

---

### Step 6: Train Classical Models

脚本：

```text
src/train_classical.py
```

功能：

- 读取 train/test feature CSV
- 使用 `classical_feature_columns.txt` 确定训练特征列
- 训练 Gaussian Naive Bayes
- 训练 k-NN，默认 `k = 5`
- 保存模型和 prediction CSV
- 使用 tqdm 显示训练进度

运行：

```bash
python C:\AIA_workspace\src\train_classical.py
```

或指定 k 值：

```bash
python C:\AIA_workspace\src\train_classical.py --knn-k 3
```

输出：

```text
models/classical/gaussian_nb.joblib
models/classical/knn_k5.joblib
data/processed/predictions/classical/gaussian_nb_predictions.csv
data/processed/predictions/classical/knn_k5_predictions.csv
results/tables/classical_training_summary.csv
```

说明：

- `.joblib` 是 scikit-learn 模型保存格式
- 不是 `.pth`，因为当前模型不是 PyTorch neural network
- GaussianNB 和 k-NN 没有 epoch，也没有 deep learning loss curve

---

### Step 7: Evaluation and Visualization

脚本：

```text
src/evaluate.py
```

功能：

- 读取 prediction CSV
- 计算 overall accuracy
- 计算 per-class accuracy
- 保存 classification report
- 保存 confusion matrix
- 绘制 confusion matrix
- 绘制 normalized confusion matrix
- 绘制 per-class accuracy bar plot
- 绘制 overall accuracy comparison
- 可视化 easy correct examples
- 可视化 hard wrong examples
- 使用 tqdm 显示评估和绘图进度

运行：

```bash
python C:\AIA_workspace\src\evaluate.py
```

输出：

```text
results/tables/classical_evaluation_summary.csv
results/tables/per_class_accuracy_gaussian_nb.csv
results/tables/per_class_accuracy_knn_k5.csv
results/tables/confusion_matrix_gaussian_nb.csv
results/tables/confusion_matrix_knn_k5.csv
results/figures/confusion_matrix_gaussian_nb.png
results/figures/confusion_matrix_knn_k5.png
results/figures/per_class_accuracy_gaussian_nb.png
results/figures/per_class_accuracy_knn_k5.png
results/figures/overall_accuracy_comparison.png
```

---

## 5. Recommended Running Order

完整 classical pipeline 运行顺序：

```bash
python C:\AIA_workspace\src\preprocessing.py
python C:\AIA_workspace\src\features_hu.py
python C:\AIA_workspace\src\features_fourier.py --num-descriptors 20
python C:\AIA_workspace\src\build_features.py
python C:\AIA_workspace\src\split_data.py
python C:\AIA_workspace\src\train_classical.py
python C:\AIA_workspace\src\evaluate.py
```

---

## 6. Current Project Status

已完成：

- Flavia 数据集准备
- label 映射表生成
- silhouette segmentation
- binary mask 保存
- ordered boundary points 保存
- Hu moments feature extraction
- Fourier descriptors feature extraction
- Hu + Fourier feature concatenation
- stratified train/test split 方案
- Gaussian Naive Bayes / k-NN 训练脚本
- evaluation + visualization 脚本

待完成：

- 运行 `train_classical.py` 并记录结果
- 运行 `evaluate.py` 并分析结果
- 在报告中解释 confusion matrix 和 per-class accuracy
- 分析 easy / hard examples
- 后续完成 modern comparison：MobileNetV2 frozen features + Linear SVM

---

## 7. Notes for Report

报告中可以重点说明：

- 本任务是 32 类 leaf species classification
- classical features 只使用 shape，不使用颜色或纹理
- Hu moments 描述叶片区域的整体几何分布
- Fourier descriptors 描述叶片边界形状
- 使用 stratified 70/30 train/test split
- 所有模型使用同一个 split 进行公平比较
- k-NN 对特征尺度敏感，因此训练时需要使用 scaler
- GaussianNB 假设特征条件独立，这可能不完全符合 Hu + Fourier 特征
- confusion matrix 可以显示哪些 species 容易混淆
- hard examples 可以说明只用 shape descriptor 的局限性

---

## 8. Modern Comparison Plan

后续 modern baseline 可以加入：

```text
MobileNetV2 frozen features + Linear SVM
```

建议新增文件：

```text
src/mobilenet_features.py
src/train_mobilenet_svm.py
```

流程：

```text
cropped leaf image
→ resize to 224 x 224
→ MobileNetV2 frozen feature extractor
→ feature vector
→ Linear SVM
→ same train/test split
→ compare with classical baseline
```
