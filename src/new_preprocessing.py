"""
Preprocessing for Topic A: Flavia Leaf Species Recognition

This script reads raw Flavia leaf images, segments each leaf into a binary silhouette,
extracts the ordered boundary points, and saves reproducible preprocessing metadata.

Default input:
    C:\\AIA_workspace\\data\\raw\\Leaves
Default output:
    C:\\AIA_workspace\\data\\processed

Outputs:
    masks/                  binary leaf masks, leaf=255, background=0
    boundaries/             ordered boundary points as .npy files, shape (N, 2), columns x,y
    cropped_preview/        optional cropped leaf preview images, by default 5 per species
    contours_preview/       optional contour visualization images, by default 5 per species
    preprocess_summary.csv  one row per successfully processed image
    failed_files.txt        failed image paths and error reasons

Notes:
    - CHAIN_APPROX_NONE is used so Fourier descriptors can later use dense ordered contours.
    - Preview images are saved by default for a small number of images per species to avoid
      slow processing from writing thousands of PNG files. Masks and boundaries are saved for every image.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# =========================
# 1. Default paths and options
# =========================

DEFAULT_INPUT_DIR = Path(r"C:\AIA_workspace\data\raw\Leaves")
DEFAULT_OUTPUT_DIR = Path(r"C:\AIA_workspace\data\processed")

IMAGE_SUFFIXES = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

# Smaller kernel preserves more boundary detail than 5x5, which is useful for Fourier descriptors.
MORPH_KERNEL_SIZE = 3
GAUSSIAN_KERNEL_SIZE = 5
MIN_CONTOUR_AREA = 100.0
PNG_COMPRESSION = 1  # 0 is fastest/largest, 9 is slowest/smallest.

# Save masks and boundaries for every image. Save preview images only for first N images per species.
DEFAULT_PREVIEWS_PER_CLASS = 5


# =========================
# 2. Flavia label mapping
# =========================

FLAVIA_RANGES = [
    (1001, 1059, 1, "Phyllostachys edulis (Carr.) Houz.", "pubescent bamboo"),
    (1060, 1122, 2, "Aesculus chinensis", "Chinese horse chestnut"),
    (1552, 1616, 3, "Berberis anhweiensis Ahrendt", "Anhui Barberry"),
    (1123, 1194, 4, "Cercis chinensis", "Chinese redbud"),
    (1195, 1267, 5, "Indigofera tinctoria L.", "true indigo"),
    (1268, 1323, 6, "Acer Palmatum", "Japanese maple"),
    (1324, 1385, 7, "Phoebe nanmu (Oliv.) Gamble", "Nanmu"),
    (1386, 1437, 8, "Kalopanax septemlobus (Thunb. ex A.Murr.) Koidz.", "castor aralia"),
    (1497, 1551, 9, "Cinnamomum japonicum Sieb.", "Chinese cinnamon"),
    (1438, 1496, 10, "Koelreuteria paniculata Laxm.", "goldenrain tree"),
    (2001, 2050, 11, "Ilex macrocarpa Oliv.", "Big-fruited Holly"),
    (2051, 2113, 12, "Pittosporum tobira (Thunb.) Ait. f.", "Japanese cheesewood"),
    # The source table has no label 13.
    (2114, 2165, 14, "Chimonanthus praecox L.", "wintersweet"),
    (2166, 2230, 15, "Cinnamomum camphora (L.) J. Presl", "camphortree"),
    (2231, 2290, 16, "Viburnum awabuki K.Koch", "Japan Arrowwood"),
    (2291, 2346, 17, "Osmanthus fragrans Lour.", "sweet osmanthus"),
    (2347, 2423, 18, "Cedrus deodara (Roxb.) G. Don", "deodar"),
    (2424, 2485, 19, "Ginkgo biloba L.", "ginkgo, maidenhair tree"),
    (2486, 2546, 20, "Lagerstroemia indica (L.) Pers.", "Crape myrtle, Crepe myrtle"),
    (2547, 2612, 21, "Nerium oleander L.", "oleander"),
    (2616, 2675, 22, "Podocarpus macrophyllus (Thunb.) Sweet", "yew plum pine"),
    (3001, 3055, 23, "Prunus serrulata Lindl. var. lannesiana auct.", "Japanese Flowering Cherry"),
    (3056, 3110, 24, "Ligustrum lucidum Ait. f.", "Glossy Privet"),
    (3111, 3175, 25, "Toona sinensis M. Roem.", "Chinese Toon"),
    (3176, 3229, 26, "Prunus persica (L.) Batsch", "peach"),
    (3230, 3281, 27, "Manglietia fordiana Oliv.", "Ford Woodlotus"),
    (3282, 3334, 28, "Acer buergerianum Miq.", "trident maple"),
    (3335, 3389, 29, "Mahonia bealei (Fortune) Carr.", "Beale's barberry"),
    (3390, 3446, 30, "Magnolia grandiflora L.", "southern magnolia"),
    (3447, 3510, 31, "Populus ×canadensis Moench", "Canadian poplar"),
    (3511, 3563, 32, "Liriodendron chinense (Hemsl.) Sarg.", "Chinese tulip tree"),
    (3566, 3621, 33, "Citrus reticulata Blanco", "tangerine"),
]


def get_flavia_label(image_id: str) -> Dict[str, str]:
    """Return label metadata from a Flavia 4-digit image id."""
    try:
        numeric_id = int(image_id)
    except ValueError:
        return {
            "label": "unknown",
            "scientific_name": "unknown",
            "common_name": "unknown",
        }

    for start, end, label, scientific_name, common_name in FLAVIA_RANGES:
        if start <= numeric_id <= end:
            return {
                "label": str(label),
                "scientific_name": scientific_name,
                "common_name": common_name,
            }

    return {
        "label": "unknown",
        "scientific_name": "unknown",
        "common_name": "unknown",
    }


# =========================
# 3. Data containers
# =========================

@dataclass
class OutputDirs:
    root: Path
    masks: Path
    boundaries: Path
    cropped_preview: Path
    contours_preview: Path

    @classmethod
    def create(cls, output_dir: Path) -> "OutputDirs":
        dirs = cls(
            root=output_dir,
            masks=output_dir / "masks",
            boundaries=output_dir / "boundaries",
            cropped_preview=output_dir / "cropped_preview",
            contours_preview=output_dir / "contours_preview",
        )
        for directory in [dirs.root, dirs.masks, dirs.boundaries, dirs.cropped_preview, dirs.contours_preview]:
            directory.mkdir(parents=True, exist_ok=True)
        return dirs


# =========================
# 4. Utility functions
# =========================

def find_image_paths(input_dir: Path) -> List[Path]:
    """Find all images recursively and return them in a stable sorted order."""
    image_paths: List[Path] = []
    for suffix in IMAGE_SUFFIXES:
        image_paths.extend(input_dir.rglob(suffix))
    return sorted(image_paths)


def read_image(img_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read image and return BGR, RGB, grayscale."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError("cv2.imread returned None. Check path or image file.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_rgb, gray


def segment_leaf(gray: np.ndarray) -> np.ndarray:
    """
    Segment leaf from light background.

    Returns a binary mask where leaf=255 and background=0.
    An automatic foreground-area check is included to correct accidental inversion.
    """
    blurred = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # If foreground occupies too much area, the threshold is probably inverted.
    foreground_ratio = np.count_nonzero(binary) / binary.size
    if foreground_ratio > 0.75:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def get_largest_ordered_contour(binary: np.ndarray) -> np.ndarray:
    """
    Extract the largest external contour with dense ordered points.

    Returns:
        contour_points: numpy array with shape (N, 2), columns are x,y.
    """
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,  # keep dense ordered boundary points for Fourier descriptors
    )

    if not contours:
        raise ValueError("no external contours found")

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < MIN_CONTOUR_AREA:
        raise ValueError(f"largest contour too small: area={area:.2f}")

    return largest_contour.reshape(-1, 2)


def contour_to_filled_mask(gray_shape: Tuple[int, int], contour_points: np.ndarray) -> np.ndarray:
    """Create a clean filled mask from ordered contour points."""
    mask = np.zeros(gray_shape, dtype=np.uint8)
    contour_cv = contour_points.reshape(-1, 1, 2).astype(np.int32)
    cv2.drawContours(mask, [contour_cv], -1, 255, thickness=-1)
    return mask


def make_cropped_leaf(img_rgb: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop leaf bounding box and replace background with white."""
    x, y, w, h = bbox
    cropped_img = img_rgb[y : y + h, x : x + w]
    cropped_mask = mask[y : y + h, x : x + w]
    white_background = np.ones_like(cropped_img) * 255
    cropped_leaf = np.where(cropped_mask[:, :, None] == 255, cropped_img, white_background)
    return cropped_leaf


def make_contour_visualization(img_rgb: np.ndarray, contour_points: np.ndarray) -> np.ndarray:
    """Draw the detected contour on the original image for manual checking."""
    contour_vis = img_rgb.copy()
    contour_cv = contour_points.reshape(-1, 1, 2).astype(np.int32)
    cv2.drawContours(contour_vis, [contour_cv], -1, (255, 0, 0), 3)
    return contour_vis


def save_png(path: Path, image: np.ndarray) -> None:
    """Save PNG with low compression for faster preprocessing."""
    cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])


# =========================
# 5. Main processing functions
# =========================

def preprocess_leaf_image(
    img_path: Path,
    out_dirs: OutputDirs,
    save_preview: bool = False,
) -> Dict[str, object]:
    """Preprocess one leaf image and save mask/boundary/optional preview images."""
    img_bgr, img_rgb, gray = read_image(img_path)

    binary = segment_leaf(gray)
    contour_points = get_largest_ordered_contour(binary)
    mask = contour_to_filled_mask(gray.shape, contour_points)

    contour_cv = contour_points.reshape(-1, 1, 2).astype(np.int32)
    contour_area = float(cv2.contourArea(contour_cv))
    x, y, w, h = cv2.boundingRect(contour_cv)

    image_id = img_path.stem
    label_info = get_flavia_label(image_id)

    mask_path = out_dirs.masks / f"{image_id}_mask.png"
    boundary_path = out_dirs.boundaries / f"{image_id}_boundary.npy"

    save_png(mask_path, mask)
    np.save(boundary_path, contour_points.astype(np.float32))

    crop_path = ""
    contour_preview_path = ""

    if save_preview:
        cropped_leaf = make_cropped_leaf(img_rgb, mask, (x, y, w, h))
        contour_vis = make_contour_visualization(img_rgb, contour_points)

        crop_path_obj = out_dirs.cropped_preview / f"{image_id}_crop.png"
        contour_path_obj = out_dirs.contours_preview / f"{image_id}_contour.png"

        save_png(crop_path_obj, cv2.cvtColor(cropped_leaf, cv2.COLOR_RGB2BGR))
        save_png(contour_path_obj, cv2.cvtColor(contour_vis, cv2.COLOR_RGB2BGR))

        crop_path = str(crop_path_obj)
        contour_preview_path = str(contour_path_obj)

    return {
        "filename": img_path.name,
        "image_id": image_id,
        "label": label_info["label"],
        "scientific_name": label_info["scientific_name"],
        "common_name": label_info["common_name"],
        "width": int(img_bgr.shape[1]),
        "height": int(img_bgr.shape[0]),
        "contour_area": contour_area,
        "num_boundary_points": int(len(contour_points)),
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(w),
        "bbox_h": int(h),
        "mask_path": str(mask_path),
        "boundary_path": str(boundary_path),
        "crop_preview_path": crop_path,
        "contour_preview_path": contour_preview_path,
    }


def write_summary_csv(summary_path: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        return

    fieldnames = list(records[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_failed_files(failed_path: Path, failed_files: List[Tuple[str, str]]) -> None:
    with open(failed_path, "w", encoding="utf-8") as f:
        for path, reason in failed_files:
            f.write(f"{path}\t{reason}\n")


def process_dataset(input_dir: Path, output_dir: Path, previews_per_class: int = DEFAULT_PREVIEWS_PER_CLASS) -> None:
    out_dirs = OutputDirs.create(output_dir)

    image_paths = find_image_paths(input_dir)

    print("输入文件夹:", input_dir)
    print("输出文件夹:", output_dir)
    print("找到图片数量:", len(image_paths))
    print("每类预览图保存数量:", previews_per_class)

    if not image_paths:
        raise FileNotFoundError(f"没有在该路径下找到图片: {input_dir}")

    records: List[Dict[str, object]] = []
    failed_files: List[Tuple[str, str]] = []
    preview_counts: Dict[str, int] = {}

    for i, img_path in enumerate(image_paths, start=1):
        label_info = get_flavia_label(img_path.stem)
        preview_key = label_info["label"]
        save_preview = preview_counts.get(preview_key, 0) < previews_per_class

        print(f"[{i}/{len(image_paths)}] 正在处理: {img_path.name} | label={preview_key} | save_preview={save_preview}")

        try:
            record = preprocess_leaf_image(img_path, out_dirs, save_preview=save_preview)
            records.append(record)

            if save_preview:
                preview_counts[preview_key] = preview_counts.get(preview_key, 0) + 1

        except Exception as exc:  # keep batch processing robust
            reason = str(exc)
            print(f"  [失败] {img_path.name}: {reason}")
            failed_files.append((str(img_path), reason))

    summary_path = output_dir / "preprocess_summary.csv"
    failed_path = output_dir / "failed_files.txt"

    write_summary_csv(summary_path, records)
    write_failed_files(failed_path, failed_files)

    unknown_labels = sum(1 for r in records if r.get("label") == "unknown")

    print("\n处理完成")
    print("成功处理图片数量:", len(records))
    print("处理失败图片数量:", len(failed_files))
    print("unknown label 数量:", unknown_labels)
    print("Mask 保存到:", out_dirs.masks)
    print("Boundary points 保存到:", out_dirs.boundaries)
    print("每类预览图实际保存数量:", dict(sorted(preview_counts.items(), key=lambda item: item[0])))
    print("裁剪预览图保存到:", out_dirs.cropped_preview)
    print("轮廓预览图保存到:", out_dirs.contours_preview)
    print("处理记录保存到:", summary_path)
    print("失败文件列表保存到:", failed_path)


# =========================
# 6. CLI entry point
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Flavia leaf images for Topic A.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing raw leaf images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for processed masks, boundaries, previews, and CSV summary.",
    )
    parser.add_argument(
        "--previews-per-class",
        type=int,
        default=DEFAULT_PREVIEWS_PER_CLASS,
        help="Save cropped/contour preview images for the first N successfully processed images of each species. Masks and boundaries are always saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(args.input_dir, args.output_dir, args.previews_per_class)


if __name__ == "__main__":
    main()
