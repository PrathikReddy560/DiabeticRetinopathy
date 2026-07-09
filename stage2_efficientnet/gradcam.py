"""
gradcam.py — Grad-CAM visualization for EfficientNet-B0 DR grader.

Generates class-discriminative heatmaps showing which retinal regions
the model focuses on when predicting each DR severity grade.
Targets the last convolutional layer ('conv_head') of EfficientNet-B0.

Usage:
    python gradcam.py
    python gradcam.py --checkpoint path/to/model.pth
    python gradcam.py --image path/to/fundus.png --grade 2
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import DRClassifier
from dataset import preprocess_image, get_train_val_test_loaders, IMAGENET_MEAN, IMAGENET_STD

# ── Constants ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(_SCRIPT_DIR, '..', 'checkpoints', 'efficientnet_best.pth')
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, '..', 'output', 'gradcam')

GRADE_NAMES = [
    'Grade 0 (No DR)', 'Grade 1 (Mild)', 'Grade 2 (Moderate)',
    'Grade 3 (Severe)', 'Grade 4 (Proliferative)',
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════
#  Grad-CAM core
# ══════════════════════════════════════════════════════════════════

class GradCAM:
    """Grad-CAM: Gradient-weighted Class Activation Mapping.

    Hooks into a target convolutional layer to capture:
      - Forward activations (feature maps)
      - Backward gradients (importance weights)

    Args:
        model: The DRClassifier model.
        target_layer: The nn.Module to hook (e.g., model.backbone.conv_head).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        # Register hooks on the target layer
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Save feature map activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Save gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate a Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, 3, 224, 224).
            target_class: Grade to explain (0-4). If None, uses predicted class.

        Returns:
            cam: Normalized heatmap as numpy array (224, 224), values in [0, 1].
            predicted_class: The class the model predicted.
        """
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)
        predicted_class = logits.argmax(dim=1).item()

        if target_class is None:
            target_class = predicted_class

        # Backward pass from the target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]        # (C, H, W)
        activations = self.activations[0]    # (C, H, W)

        # Global average pooling of gradients → per-channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=DEVICE)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only keep positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input dimensions (224×224)
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

        return cam, predicted_class


# ══════════════════════════════════════════════════════════════════
#  Overlay utilities
# ══════════════════════════════════════════════════════════════════

def create_overlay(original_image, heatmap, alpha=0.4):
    """Blend a Grad-CAM heatmap with the original image.

    Args:
        original_image: RGB uint8 numpy array (H, W, 3).
        heatmap: Float numpy array (H, W) in [0, 1].
        alpha: Heatmap opacity (0 = invisible, 1 = fully opaque).

    Returns:
        overlay: Blended uint8 image (H, W, 3).
    """
    # Convert heatmap to colour (JET colourmap)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Resize original to match
    original_resized = cv2.resize(original_image, (224, 224))

    # Blend
    overlay = ((1 - alpha) * original_resized.astype(np.float32) +
               alpha * heatmap_colored.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def _tensor_to_image(tensor):
    """Convert a normalised tensor back to a displayable uint8 RGB image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # C,H,W → H,W,C
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ══════════════════════════════════════════════════════════════════
#  Generate samples for all grades
# ══════════════════════════════════════════════════════════════════

def generate_gradcam_samples(checkpoint_path=CHECKPOINT_PATH, output_dir=OUTPUT_DIR,
                              samples_per_grade=5):
    """Generate Grad-CAM visualizations for sample images from each grade.

    Saves:
      - Individual images: grade_{g}_sample_{i}.png
      - Summary grid: gradcam_summary.png
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = DRClassifier(pretrained=False)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[✓] Model loaded from {checkpoint_path}")

    # Target the last conv layer of EfficientNet-B0
    target_layer = model.backbone.conv_head
    grad_cam = GradCAM(model, target_layer)

    # Get test data
    _, _, test_loader, _ = get_train_val_test_loaders(batch_size=1, num_workers=0)
    print(f"[✓] Test set: {len(test_loader.dataset)} images")

    # Collect correctly classified samples per grade
    grade_samples = {g: [] for g in range(5)}  # {grade: [(img_tensor, label), ...]}

    print("\nCollecting correctly classified samples...")
    for img_tensor, label in test_loader:
        label_val = label.item()
        if len(grade_samples[label_val]) >= samples_per_grade:
            # Check if we have enough for all grades
            if all(len(v) >= samples_per_grade for v in grade_samples.values()):
                break
            continue

        # Quick forward check — only keep if correctly classified
        with torch.no_grad():
            pred = model(img_tensor.to(DEVICE)).argmax(dim=1).item()
        if pred == label_val:
            grade_samples[label_val].append(img_tensor)

    # Generate Grad-CAMs
    print("\nGenerating Grad-CAM heatmaps...")
    fig_rows = []

    for g in range(5):
        samples = grade_samples[g]
        if not samples:
            print(f"  Grade {g}: no correctly classified samples found")
            continue

        row_images = []
        for i, img_tensor in enumerate(samples):
            # Generate heatmap
            cam, pred = grad_cam.generate(img_tensor, target_class=g)

            # Get displayable original image
            original = _tensor_to_image(img_tensor[0])

            # Create overlay
            overlay = create_overlay(original, cam, alpha=0.4)

            # Create heatmap image (JET)
            heatmap_vis = cv2.applyColorMap(
                (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_vis = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)

            # Save individual: original | heatmap | overlay
            fig_ind, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(original)
            axes[0].set_title('Original', fontsize=10)
            axes[0].axis('off')
            axes[1].imshow(heatmap_vis)
            axes[1].set_title('Grad-CAM Heatmap', fontsize=10)
            axes[1].axis('off')
            axes[2].imshow(overlay)
            axes[2].set_title(f'Overlay (Pred: {GRADE_NAMES[pred]})', fontsize=10)
            axes[2].axis('off')
            fig_ind.suptitle(f'{GRADE_NAMES[g]} — Sample {i+1}', fontsize=12,
                            fontweight='bold')
            fig_ind.tight_layout()

            save_path = os.path.join(output_dir, f'grade_{g}_sample_{i+1}.png')
            fig_ind.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig_ind)

            row_images.append(overlay)

        fig_rows.append((g, row_images))
        print(f"  Grade {g}: {len(samples)} samples saved")

    # Create summary grid
    if fig_rows:
        n_rows = len(fig_rows)
        n_cols = max(len(imgs) for _, imgs in fig_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for row_idx, (grade, images) in enumerate(fig_rows):
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx < len(images):
                    ax.imshow(images[col_idx])
                    if col_idx == 0:
                        ax.set_ylabel(GRADE_NAMES[grade], fontsize=10,
                                     fontweight='bold', rotation=0, labelpad=80,
                                     va='center')
                ax.axis('off')

        fig.suptitle('Grad-CAM Summary — DR Severity Grading', fontsize=16,
                    fontweight='bold', y=1.02)
        fig.tight_layout()
        summary_path = os.path.join(output_dir, 'gradcam_summary.png')
        fig.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n[✓] Summary grid saved → {summary_path}")

    print(f"[✓] All Grad-CAM outputs saved to {output_dir}")


# ══════════════════════════════════════════════════════════════════
#  Single-image Grad-CAM
# ══════════════════════════════════════════════════════════════════

def gradcam_single_image(image_path, checkpoint_path=CHECKPOINT_PATH,
                          target_class=None, output_path=None):
    """Generate Grad-CAM for a single fundus image.

    Args:
        image_path: Path to the fundus image.
        checkpoint_path: Path to the model checkpoint.
        target_class: Grade to explain (0-4). None = predicted class.
        output_path: Where to save the output. None = auto-generate.

    Returns:
        overlay: The blended overlay image (numpy array).
        predicted_grade: The model's prediction.
        heatmap: The raw Grad-CAM heatmap.
    """
    # Load model
    model = DRClassifier(pretrained=False)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    target_layer = model.backbone.conv_head
    grad_cam = GradCAM(model, target_layer)

    # Preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preprocessed = preprocess_image(img_rgb)

    # Convert to tensor
    img_float = preprocessed.astype(np.float32) / 255.0
    img_norm = (img_float - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    # Generate Grad-CAM
    heatmap, predicted_grade = grad_cam.generate(input_tensor, target_class)
    overlay = create_overlay(preprocessed, heatmap, alpha=0.4)

    # Save if path provided
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f'gradcam_{basename}.png')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(preprocessed)
    axes[0].set_title('Original (preprocessed)', fontsize=11)
    axes[0].axis('off')

    heatmap_vis = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_vis = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)
    axes[1].imshow(heatmap_vis)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay — {GRADE_NAMES[predicted_grade]}', fontsize=11)
    axes[2].axis('off')

    fig.suptitle(f'Predicted: {GRADE_NAMES[predicted_grade]}', fontsize=14,
                fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Grad-CAM saved → {output_path}")

    return overlay, predicted_grade, heatmap


# ══════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grad-CAM for DR grading')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to a single fundus image')
    parser.add_argument('--grade', type=int, default=None,
                       help='Target grade class (0-4), default=predicted')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for single image')
    args = parser.parse_args()

    if args.image:
        gradcam_single_image(args.image, args.checkpoint, args.grade, args.output)
    else:
        generate_gradcam_samples(args.checkpoint)
