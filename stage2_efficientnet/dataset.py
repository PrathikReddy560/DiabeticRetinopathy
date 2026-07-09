"""
dataset.py - Multi-source dataset loader for Diabetic Retinopathy grading.

Combines images from APTOS 2019, IDRiD, and (optionally) Messidor-2 into
unified train / validation / test splits with:
    - CLAHE + Ben Graham preprocessing
    - Albumentations-based training augmentations
    - WeightedRandomSampler for class-imbalanced training
"""

import os
import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter


# =========================================================================
# Paths (absolute, Windows)
# =========================================================================
DATASET_ROOT = r"c:\DiabeticRetinopathy\dataset"

# ----- APTOS 2019 -------------------------------------------------------
APTOS_TRAIN_CSV  = os.path.join(DATASET_ROOT, "train.csv")
APTOS_TEST_CSV   = os.path.join(DATASET_ROOT, "test.csv")
APTOS_VAL_CSV    = os.path.join(DATASET_ROOT, "valid.csv")
APTOS_TRAIN_IMGS = os.path.join(DATASET_ROOT, "train_images")
APTOS_TEST_IMGS  = os.path.join(DATASET_ROOT, "test_images")
APTOS_VAL_IMGS   = os.path.join(DATASET_ROOT, "val_images")

# ----- IDRiD -------------------------------------------------------------
IDRID_ROOT         = os.path.join(DATASET_ROOT, "IDRiD", "B. Disease Grading")
IDRID_TRAIN_CSV    = os.path.join(IDRID_ROOT, "2. Groundtruths",
                                  "a. IDRiD_Disease Grading_Training Labels.csv")
IDRID_TEST_CSV     = os.path.join(IDRID_ROOT, "2. Groundtruths",
                                  "b. IDRiD_Disease Grading_Testing Labels.csv")
IDRID_TRAIN_IMGS   = os.path.join(IDRID_ROOT, "1. Original Images",
                                  "a. Training Set")
IDRID_TEST_IMGS    = os.path.join(IDRID_ROOT, "1. Original Images",
                                  "b. Testing Set")

# ----- Messidor-2 (optional) --------------------------------------------
MESSIDOR2_ROOT = os.path.join(DATASET_ROOT, "messidor-2")
MESSIDOR2_CSV  = os.path.join(DATASET_ROOT, "messidor_data.csv")
MESSIDOR2_IMGS = os.path.join(DATASET_ROOT, "messidor-2", "messidor-2", "preprocess")

# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =========================================================================
# Preprocessing helpers
# =========================================================================

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
    L channel of the LAB colour space to enhance local contrast.

    Args:
        image: RGB uint8 image array (H, W, 3).

    Returns:
        CLAHE-enhanced RGB uint8 image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def ben_graham_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Ben Graham's preprocessing: subtract a Gaussian-blurred version of the
    image to remove large-scale illumination gradients, then rescale.

    Formula: result = 4 * image - 4 * GaussianBlur(image, sigma=10) + 128

    Args:
        image: RGB uint8 image array (H, W, 3).

    Returns:
        Preprocessed RGB uint8 image.
    """
    return cv2.addWeighted(
        image, 4,
        cv2.GaussianBlur(image, (0, 0), sigmaX=10), -4,
        128
    )


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline applied to every image before augmentation:
        1. CLAHE on LAB colour space
        2. Ben Graham normalisation
        3. Resize to 224×224

    Args:
        image: RGB uint8 image array.

    Returns:
        Preprocessed RGB uint8 image of shape (224, 224, 3).
    """
    image = apply_clahe(image)
    image = ben_graham_preprocessing(image)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    return image


# =========================================================================
# Augmentation transforms
# =========================================================================

def get_train_transforms() -> A.Compose:
    """
    Training augmentation pipeline using Albumentations v1.4+.

    Includes geometric, photometric, noise, and cutout augmentations
    followed by ImageNet normalisation and conversion to a PyTorch tensor.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-180, 180),
            p=0.7,
            mode=cv2.BORDER_CONSTANT,
            cval=0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.3
        ),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(10, 20),
            hole_width_range=(10, 20),
            p=0.3
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """
    Validation / test transform: normalise + convert to tensor only.
    No data augmentation.
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# =========================================================================
# Dataset class
# =========================================================================

class DRGradingDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy severity grading.

    Each sample consists of a retinal fundus image (preprocessed with CLAHE
    + Ben Graham normalisation, resized to 224×224) and an integer grade
    label in {0, 1, 2, 3, 4}.

    Args:
        image_paths:   List of absolute paths to fundus images.
        labels:        List of integer grade labels (0-4), same length as
                       image_paths.
        transform:     Albumentations Compose pipeline applied after
                       preprocessing (augmentation + normalisation).
        preprocessing: Whether to apply CLAHE + Ben Graham preprocessing.
                       Default: True.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[A.Compose] = None,
        preprocessing: bool = True,
    ):
        assert len(image_paths) == len(labels), (
            f"Mismatch: {len(image_paths)} images vs {len(labels)} labels"
        )
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load, preprocess, augment, and return a single sample.

        Returns:
            image: Float tensor of shape (3, 224, 224).
            label: Integer DR grade (0-4).
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 1. Read image with OpenCV (BGR) and convert to RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {img_path}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2-4. Apply preprocessing pipeline (CLAHE → Ben Graham → resize)
        if self.preprocessing:
            image = preprocess_image(image)

        # 5. Apply augmentation / normalisation transforms
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# =========================================================================
# Data-loading helpers for individual datasets
# =========================================================================

def _load_aptos(csv_path: str, img_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load APTOS 2019 split (train / valid / test).

    CSV columns: id_code, diagnosis
    Images:      {img_dir}/{id_code}.png

    Returns:
        Tuple of (image_paths, labels).
    """
    df = pd.read_csv(csv_path)
    image_paths: List[str] = []
    labels: List[int] = []

    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, f"{row['id_code']}.png")
        if os.path.isfile(img_path):
            image_paths.append(img_path)
            labels.append(int(row['diagnosis']))
        else:
            print(f"[APTOS] WARNING: image not found – {img_path}")

    return image_paths, labels


def _load_idrid(csv_path: str, img_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load IDRiD split (train / test).

    CSV columns: 'Image name', 'Retinopathy grade',
                 'Risk of macular edema ' (trailing space)
    NOTE: CSV may have trailing commas producing extra empty columns.
    Images:      {img_dir}/{Image name}.jpg

    Returns:
        Tuple of (image_paths, labels).
    """
    df = pd.read_csv(csv_path)

    # Clean up column names: strip whitespace and drop unnamed columns
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    image_paths: List[str] = []
    labels: List[int] = []

    for _, row in df.iterrows():
        img_name = str(row['Image name']).strip()
        img_path = os.path.join(img_dir, f"{img_name}.jpg")
        if os.path.isfile(img_path):
            image_paths.append(img_path)
            labels.append(int(row['Retinopathy grade']))
        else:
            print(f"[IDRiD] WARNING: image not found – {img_path}")

    return image_paths, labels


def _load_messidor2() -> Tuple[List[str], List[int]]:
    """
    Load Messidor-2 dataset.

    CSV:    dataset/messidor_data.csv  (columns: id_code, diagnosis)
    Images: dataset/messidor-2/messidor-2/preprocess/{id_code}
            (id_code already contains .png extension)

    Returns:
        Tuple of (image_paths, labels).
    """
    if not os.path.isfile(MESSIDOR2_CSV):
        print(f"[Messidor-2] CSV not found: {MESSIDOR2_CSV} – skipping.")
        return [], []

    if not os.path.isdir(MESSIDOR2_IMGS):
        print(f"[Messidor-2] Image dir not found: {MESSIDOR2_IMGS} – skipping.")
        return [], []

    df = pd.read_csv(MESSIDOR2_CSV)
    df.columns = df.columns.str.strip()

    image_paths: List[str] = []
    labels: List[int] = []
    skipped = 0

    for _, row in df.iterrows():
        img_name = str(row['id_code']).strip()
        grade = row['diagnosis']

        # Skip non-gradable or out-of-range
        if pd.isna(grade):
            continue
        grade = int(grade)
        if grade < 0 or grade > 4:
            continue

        # id_code already has .png extension
        img_path = os.path.join(MESSIDOR2_IMGS, img_name)
        if os.path.isfile(img_path):
            image_paths.append(img_path)
            labels.append(grade)
        else:
            # Try without extension (some CSVs have it, some don't)
            stem = os.path.splitext(img_name)[0]
            for ext in ('.png', '.jpg', '.tif', '.tiff'):
                alt = os.path.join(MESSIDOR2_IMGS, stem + ext)
                if os.path.isfile(alt):
                    image_paths.append(alt)
                    labels.append(grade)
                    break
            else:
                skipped += 1

    if skipped > 0:
        print(f"[Messidor-2] WARNING: {skipped} images not found on disk.")
    print(f"[Messidor-2] Loaded {len(image_paths)} images.")
    return image_paths, labels


# =========================================================================
# Statistics printer
# =========================================================================

def _print_split_stats(name: str, labels: List[int]) -> None:
    """Pretty-print the size and grade distribution for a data split."""
    counter = Counter(labels)
    total = len(labels)
    print(f"\n  {name}: {total} images")
    for grade in sorted(counter.keys()):
        count = counter[grade]
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"    Grade {grade}: {count:>5d}  ({pct:5.1f}%)")


# =========================================================================
# Main loader factory
# =========================================================================

def get_train_val_test_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train, validation, and test DataLoaders from all available
    datasets with class-balanced sampling for training.

    Splits:
        Train : APTOS train + IDRiD train + Messidor-2 (if available)
        Val   : APTOS valid.csv
        Test  : APTOS test + IDRiD test

    Args:
        batch_size:  Mini-batch size for all loaders.
        num_workers: Number of parallel data-loading workers.

    Returns:
        train_loader:        DataLoader with WeightedRandomSampler.
        val_loader:          DataLoader (no sampler, no augmentation).
        test_loader:         DataLoader (no sampler, no augmentation).
        class_weights_tensor: Tensor of shape (5,) with inverse-frequency
                              class weights (useful for loss weighting).
    """
    print("=" * 60)
    print("  Loading datasets for Stage 2 – EfficientNet-B0 DR Grader")
    print("=" * 60)

    # -----------------------------------------------------------------
    # 1. Collect all (path, label) pairs per split
    # -----------------------------------------------------------------

    # ── Training ──
    train_paths, train_labels = [], []

    # APTOS train
    p, l = _load_aptos(APTOS_TRAIN_CSV, APTOS_TRAIN_IMGS)
    print(f"[APTOS train]  {len(p)} images loaded.")
    train_paths.extend(p)
    train_labels.extend(l)

    # IDRiD train
    p, l = _load_idrid(IDRID_TRAIN_CSV, IDRID_TRAIN_IMGS)
    print(f"[IDRiD train]  {len(p)} images loaded.")
    train_paths.extend(p)
    train_labels.extend(l)

    # Messidor-2 (optional)
    p, l = _load_messidor2()
    if p:
        print(f"[Messidor-2]   {len(p)} images added to training set.")
        train_paths.extend(p)
        train_labels.extend(l)

    # ── Validation ──
    val_paths, val_labels = _load_aptos(APTOS_VAL_CSV, APTOS_VAL_IMGS)
    print(f"[APTOS valid]  {len(val_paths)} images loaded.")

    # ── Test ──
    test_paths, test_labels = [], []

    p, l = _load_aptos(APTOS_TEST_CSV, APTOS_TEST_IMGS)
    print(f"[APTOS test]   {len(p)} images loaded.")
    test_paths.extend(p)
    test_labels.extend(l)

    p, l = _load_idrid(IDRID_TEST_CSV, IDRID_TEST_IMGS)
    print(f"[IDRiD test]   {len(p)} images loaded.")
    test_paths.extend(p)
    test_labels.extend(l)

    # -----------------------------------------------------------------
    # 2. Print dataset statistics
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  Dataset Statistics")
    print("-" * 60)
    _print_split_stats("Train", train_labels)
    _print_split_stats("Val  ", val_labels)
    _print_split_stats("Test ", test_labels)
    print()

    # -----------------------------------------------------------------
    # 3. Compute class weights (inverse frequency)
    # -----------------------------------------------------------------
    counter = Counter(train_labels)
    num_classes = 5
    total_train = len(train_labels)

    # Weight for class c = total_samples / (num_classes * count_c)
    class_weights = []
    for c in range(num_classes):
        count = counter.get(c, 1)  # avoid division by zero
        class_weights.append(total_train / (num_classes * count))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print(f"  Class weights (inverse frequency):")
    for c in range(num_classes):
        print(f"    Grade {c}: {class_weights_tensor[c]:.4f}")
    print()

    # -----------------------------------------------------------------
    # 4. Build WeightedRandomSampler for training
    # -----------------------------------------------------------------
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # -----------------------------------------------------------------
    # 5. Create Datasets
    # -----------------------------------------------------------------
    train_dataset = DRGradingDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=get_train_transforms(),
        preprocessing=True,
    )
    val_dataset = DRGradingDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=get_val_transforms(),
        preprocessing=True,
    )
    test_dataset = DRGradingDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=get_val_transforms(),
        preprocessing=True,
    )

    # -----------------------------------------------------------------
    # 6. Create DataLoaders
    # -----------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,          # class-balanced sampling
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  DataLoaders ready  |  batch_size={batch_size}  |  "
          f"workers={num_workers}")
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val   batches: {len(val_loader)}")
    print(f"    Test  batches: {len(test_loader)}")
    print("=" * 60)

    return train_loader, val_loader, test_loader, class_weights_tensor


# =========================================================================
# Quick smoke test
# =========================================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader, cw = get_train_val_test_loaders(
        batch_size=8, num_workers=0
    )
    print(f"\nClass weights: {cw}")

    # Grab one batch
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")   # (8, 3, 224, 224)
        print(f"Batch labels:       {labels}")
        break
