import cv2
import numpy as np
from pathlib import Path
import csv


# =========================
# 1. 路径设置
# =========================

input_dir = Path(r"C:\AIA_workspace\data\raw\Leaves")
output_dir = Path(r"C:\AIA_workspace\data\processed")

mask_dir = output_dir / "masks"
crop_dir = output_dir / "cropped"
contour_dir = output_dir / "contours"

mask_dir.mkdir(parents=True, exist_ok=True)
crop_dir.mkdir(parents=True, exist_ok=True)
contour_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 单张图片预处理函数
# =========================

def preprocess_leaf_image(img_path, save=True):
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"[读取失败] {img_path}")
        return None

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊，减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu 二值化
    # Flavia 数据集通常是浅色背景 + 深色叶片，所以使用 THRESH_BINARY_INV
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 形态学处理：去除小噪声、填补小空洞
    kernel = np.ones((5, 5), np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找外部轮廓
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print(f"[未找到轮廓] {img_path}")
        return None

    # 默认最大轮廓是叶片
    largest_contour = max(contours, key=cv2.contourArea)

    contour_area = cv2.contourArea(largest_contour)

    if contour_area <= 0:
        print(f"[无效轮廓] {img_path}")
        return None

    # 生成干净的叶片 mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)

    # 获取叶片边界框
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 裁剪 RGB 图和 mask
    cropped_img = img_rgb[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    # 白色背景，只保留叶片区域
    white_background = np.ones_like(cropped_img) * 255

    cropped_leaf = np.where(
        cropped_mask[:, :, None] == 255,
        cropped_img,
        white_background
    )

    # 轮廓可视化
    contour_vis = img_rgb.copy()
    cv2.drawContours(contour_vis, [largest_contour], -1, (255, 0, 0), 3)

    if save:
        stem = img_path.stem

        mask_path = mask_dir / f"{stem}_mask.png"
        crop_path = crop_dir / f"{stem}_crop.png"
        contour_path = contour_dir / f"{stem}_contour.png"

        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(crop_path), cv2.cvtColor(cropped_leaf, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(contour_path), cv2.cvtColor(contour_vis, cv2.COLOR_RGB2BGR))

    return {
        "filename": img_path.name,
        "width": img.shape[1],
        "height": img.shape[0],
        "contour_area": contour_area,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h
    }


# =========================
# 3. 批量读取图片
# =========================

image_paths = []

for suffix in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
    image_paths.extend(input_dir.rglob(suffix))

image_paths = sorted(image_paths)

print("输入文件夹:", input_dir)
print("输出文件夹:", output_dir)
print("找到图片数量:", len(image_paths))

if len(image_paths) == 0:
    raise FileNotFoundError(f"没有在该路径下找到图片: {input_dir}")


# =========================
# 4. 批量处理图片
# =========================

records = []
failed_files = []

for i, img_path in enumerate(image_paths, start=1):
    print(f"[{i}/{len(image_paths)}] 正在处理: {img_path.name}")

    result = preprocess_leaf_image(img_path, save=True)

    if result is not None:
        records.append(result)
    else:
        failed_files.append(str(img_path))


# =========================
# 5. 保存处理记录
# =========================

summary_path = output_dir / "preprocess_summary.csv"

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "filename",
        "width",
        "height",
        "contour_area",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h"
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)


# 保存失败文件列表
failed_path = output_dir / "failed_files.txt"

with open(failed_path, "w", encoding="utf-8") as f:
    for file in failed_files:
        f.write(file + "\n")


# =========================
# 6. 输出结果
# =========================

print("\n处理完成")
print("成功处理图片数量:", len(records))
print("处理失败图片数量:", len(failed_files))
print("Mask 保存到:", mask_dir)
print("裁剪图片保存到:", crop_dir)
print("轮廓检查图保存到:", contour_dir)
print("处理记录保存到:", summary_path)
print("失败文件列表保存到:", failed_path)