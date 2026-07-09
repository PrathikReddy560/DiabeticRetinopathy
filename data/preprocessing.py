"""
Preprocessing pipeline v2 for retinal fundus images.
Enhanced augmentation: full rotation, elastic deform, stronger jitter.
Ben Graham-style preprocessing for better contrast.
"""

import cv2
import numpy as np
from torchvision import transforms


def crop_black_borders(image):
    """Crop the black background around the circular retina."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    mx, my = int(w * 0.02), int(h * 0.02)
    x, y = max(0, x - mx), max(0, y - my)
    w = min(image.shape[1] - x, w + 2 * mx)
    h = min(image.shape[0] - y, h + 2 * my)
    return image[y:y+h, x:x+w]


def apply_clahe(image):
    """Enhance contrast using CLAHE on the lightness channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def ben_graham_preprocess(image, sigma=10):
    """Ben Graham preprocessing: subtract local average color.
    Proven to improve fundus image quality for DR detection.
    Subtracts a Gaussian-blurred version to normalize lighting.
    """
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    result = cv2.addWeighted(image, 4, blur, -4, 128)
    return result


def load_and_preprocess(image_path, image_size=128):
    """Load a fundus image, crop borders, enhance contrast, resize."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load: {image_path}")

    image = crop_black_borders(image)
    image = cv2.resize(image, (256, 256))
    image = apply_clahe(image)
    image = ben_graham_preprocess(image, sigma=10)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_train_transforms(size=128):
    """Enhanced training transforms for fundus images.
    - Full 360° rotation (fundus images are circular)
    - Stronger color jitter
    - Random erasing (forces model to learn distributed features)
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),                    # full rotation
        transforms.RandomAffine(
            degrees=0, translate=(0.05, 0.05),             # small shifts
            scale=(0.95, 1.05),                            # mild zoom
        ),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,                  # stronger
            saturation=0.2, hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),  # small patches
    ])


def get_test_transforms(size=128):
    """Test transforms — no augmentation, same normalization."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


def get_tta_transforms(size=128):
    """Test-time augmentation transforms. Returns list of transforms.
    Score is averaged over all augmented views for more robust prediction.
    """
    base = [
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
    ]
    to_tensor = [
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ]
    return [
        # Original
        transforms.Compose(base + to_tensor),
        # Horizontal flip
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + to_tensor),
        # Vertical flip
        transforms.Compose(base + [transforms.RandomVerticalFlip(p=1.0)] + to_tensor),
        # 90° rotation
        transforms.Compose(base + [transforms.Lambda(lambda img: img.rotate(90))] + to_tensor),
        # 270° rotation
        transforms.Compose(base + [transforms.Lambda(lambda img: img.rotate(270))] + to_tensor),
    ]


def denormalize(tensor):
    """Convert [-1, 1] normalized tensor back to [0, 1] for display."""
    return tensor * 0.5 + 0.5
