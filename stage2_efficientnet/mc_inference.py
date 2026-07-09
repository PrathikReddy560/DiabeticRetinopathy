"""
mc_inference.py — Monte Carlo Dropout inference for uncertainty estimation.

Runs the EfficientNet-B0 DR grader multiple times with dropout active
to estimate epistemic uncertainty. Flags low-confidence predictions
for manual review.

Usage:
    python mc_inference.py --image path/to/fundus.png
    python mc_inference.py --image path/to/fundus.png --passes 50
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import DRClassifier
from dataset import preprocess_image, IMAGENET_MEAN, IMAGENET_STD

# ── Constants ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(_SCRIPT_DIR, '..', 'checkpoints', 'efficientnet_best.pth')

MC_PASSES = 30
UNCERTAINTY_THRESHOLD = 0.15

GRADE_NAMES = [
    'Grade 0 (No DR)',
    'Grade 1 (Mild NPDR)',
    'Grade 2 (Moderate NPDR)',
    'Grade 3 (Severe NPDR)',
    'Grade 4 (Proliferative DR)',
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════
#  Image preprocessing (matches dataset.py pipeline)
# ══════════════════════════════════════════════════════════════════

def preprocess_single_image(image_path):
    """Load and preprocess a single fundus image for inference.

    Applies: CLAHE → Ben Graham normalization → resize 224×224 → ImageNet norm.

    Args:
        image_path: Absolute path to the fundus image.

    Returns:
        input_tensor: Float tensor of shape (1, 3, 224, 224), ready for the model.
        original_image: The preprocessed RGB image (before normalization) for display.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # CLAHE + Ben Graham + resize
    preprocessed = preprocess_image(img)

    # Convert to tensor with ImageNet normalization
    img_float = preprocessed.astype(np.float32) / 255.0
    img_norm = (img_float - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    return input_tensor, preprocessed


# ══════════════════════════════════════════════════════════════════
#  MC Dropout inference core
# ══════════════════════════════════════════════════════════════════

def mc_inference(model, image_tensor, num_passes=MC_PASSES):
    """Run Monte Carlo Dropout inference.

    Performs multiple stochastic forward passes with dropout active,
    then aggregates predictions to estimate uncertainty.

    Args:
        model: DRClassifier model (will be set to eval + MC dropout mode).
        image_tensor: Preprocessed image tensor (1, 3, 224, 224).
        num_passes: Number of stochastic forward passes.

    Returns:
        dict with keys:
            mean_probs      — np.ndarray (5,) average probability per grade
            std_probs       — np.ndarray (5,) standard deviation per grade
            final_grade     — int, predicted grade (argmax of mean_probs)
            confidence_pct  — float, confidence percentage
            uncertainty_flag — bool, True if std > UNCERTAINTY_THRESHOLD
            all_probs       — np.ndarray (num_passes, 5) raw probability matrix
    """
    model.eval()
    model.enable_mc_dropout()
    image_tensor = image_tensor.to(DEVICE)

    all_probs = []

    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (5,)
            all_probs.append(probs)

    all_probs = np.array(all_probs)  # (num_passes, 5)

    # Aggregate statistics
    mean_probs = all_probs.mean(axis=0)      # (5,)
    std_probs = all_probs.std(axis=0)        # (5,)
    final_grade = int(np.argmax(mean_probs))
    confidence_pct = float((1.0 - std_probs[final_grade]) * 100)
    uncertainty_flag = bool(std_probs[final_grade] > UNCERTAINTY_THRESHOLD)

    return {
        'mean_probs': mean_probs,
        'std_probs': std_probs,
        'final_grade': final_grade,
        'confidence_pct': confidence_pct,
        'uncertainty_flag': uncertainty_flag,
        'all_probs': all_probs,
    }


# ══════════════════════════════════════════════════════════════════
#  Single-image inference with formatted output
# ══════════════════════════════════════════════════════════════════

def run_single_inference(image_path, checkpoint_path=CHECKPOINT_PATH,
                          num_passes=MC_PASSES):
    """Run full MC Dropout inference on a single image and print results.

    Args:
        image_path: Path to the fundus image.
        checkpoint_path: Path to the saved model checkpoint.
        num_passes: Number of MC Dropout forward passes.

    Returns:
        result: dict from mc_inference()
    """
    # Load model
    model = DRClassifier(pretrained=False)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(DEVICE)

    # Preprocess
    image_tensor, _ = preprocess_single_image(image_path)

    # Run MC inference
    result = mc_inference(model, image_tensor, num_passes)

    # Pretty print results
    filename = os.path.basename(image_path)
    grade = result['final_grade']
    confidence = result['confidence_pct']
    mean_probs = result['mean_probs']
    std_probs = result['std_probs']
    uncertain = result['uncertainty_flag']

    banner = "=" * 60

    print(f"\n{banner}")
    print(f"  DR Severity Assessment")
    print(f"{banner}")
    print(f"  Image: {filename}")
    print(f"  MC Passes: {num_passes}")
    print(f"  Predicted Grade: {GRADE_NAMES[grade]}")
    print(f"  Confidence: {confidence:.1f}%")
    print()
    print(f"  Grade Probabilities:")
    for g in range(5):
        marker = "  ← predicted" if g == grade else ""
        print(f"    {GRADE_NAMES[g]:>30s}: {mean_probs[g]*100:5.1f}% "
              f"(±{std_probs[g]*100:4.1f}%){marker}")
    print()

    if uncertain:
        print(f"  ⚠ Status: LOW CONFIDENCE — Refer for Manual Review")
        print(f"    (predicted class std = {std_probs[grade]:.3f} > "
              f"threshold {UNCERTAINTY_THRESHOLD})")
    else:
        print(f"  ✓ Status: HIGH CONFIDENCE")

    print(f"{banner}\n")

    return result


# ══════════════════════════════════════════════════════════════════
#  Batch inference (for integration with the full pipeline)
# ══════════════════════════════════════════════════════════════════

def batch_mc_inference(model, image_tensors, num_passes=MC_PASSES):
    """Run MC Dropout on a batch of images.

    Args:
        model: DRClassifier (already loaded and on device).
        image_tensors: Tensor of shape (B, 3, 224, 224).
        num_passes: Number of MC forward passes.

    Returns:
        List of result dicts, one per image in the batch.
    """
    model.eval()
    model.enable_mc_dropout()
    image_tensors = image_tensors.to(DEVICE)
    batch_size = image_tensors.size(0)

    all_pass_probs = []  # (num_passes, B, 5)

    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(image_tensors)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_pass_probs.append(probs)

    stacked = np.stack(all_pass_probs, axis=0)  # (num_passes, B, 5)

    results = []
    for i in range(batch_size):
        sample_probs = stacked[:, i, :]  # (num_passes, 5)
        mean_probs = sample_probs.mean(axis=0)
        std_probs = sample_probs.std(axis=0)
        final_grade = int(np.argmax(mean_probs))
        confidence_pct = float((1.0 - std_probs[final_grade]) * 100)
        uncertainty_flag = bool(std_probs[final_grade] > UNCERTAINTY_THRESHOLD)

        results.append({
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'final_grade': final_grade,
            'confidence_pct': confidence_pct,
            'uncertainty_flag': uncertainty_flag,
        })

    return results


# ══════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MC Dropout inference for DR severity grading'
    )
    parser.add_argument('--image', type=str, required=True,
                       help='Path to a fundus image')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                       help='Path to model checkpoint')
    parser.add_argument('--passes', type=int, default=MC_PASSES,
                       help=f'Number of MC Dropout passes (default: {MC_PASSES})')
    args = parser.parse_args()

    run_single_inference(args.image, args.checkpoint, args.passes)
