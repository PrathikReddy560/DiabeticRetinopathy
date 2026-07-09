"""
Single-image inference — the script that will run on the Raspberry Pi.
Usage: python inference.py --image path/to/retina.png
"""

import os, sys, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models.ganomaly import Generator
from data.preprocessing import load_and_preprocess, get_test_transforms, denormalize
from utils import anomaly_score


def load_model(ckpt=None, device=None):
    """Load trained Generator and calibrated threshold."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ckpt or os.path.join(config.CHECKPOINT_DIR, "best.pth")

    G = Generator(config.CHANNELS, config.LATENT_DIM, config.FEATURE_MAPS).to(device)
    G.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)['G'])
    G.eval()

    # Load calibrated threshold
    thr_path = os.path.join(config.CHECKPOINT_DIR, "threshold.txt")
    threshold = float(open(thr_path).read().strip()) if os.path.exists(thr_path) else config.ANOMALY_THRESHOLD
    return G, threshold


def predict(image_path, G, threshold, device=None):
    """Run GANomaly on a single image. Returns result dict."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = load_and_preprocess(image_path, config.IMAGE_SIZE)
    tensor = get_test_transforms(config.IMAGE_SIZE)(img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon, z, z_hat = G(tensor)
        score = anomaly_score(z, z_hat).item()

    is_dr = score >= threshold
    return {
        'score': score,
        'threshold': threshold,
        'is_anomalous': is_dr,
        'verdict': "⚠ DR SUSPECTED" if is_dr else "✓ HEALTHY",
        'action': "→ Refer to Stage 2 for grading" if is_dr else "→ Routine follow-up in 12 months",
        'original': tensor.cpu(),
        'reconstructed': recon.cpu(),
    }


def show_result(result, save_path=None):
    """Visualize original, reconstruction, and error map."""
    orig = denormalize(result['original'][0]).permute(1, 2, 0).numpy().clip(0, 1)
    recon = denormalize(result['reconstructed'][0]).permute(1, 2, 0).numpy().clip(0, 1)
    error = np.abs(orig - recon)
    color = '#F44336' if result['is_anomalous'] else '#4CAF50'

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='#1a1a2e')
    for ax, img, title in zip(axes, [orig, recon, error],
                               ['Original', 'Reconstructed', 'Error Map']):
        ax.imshow(img if title != 'Error Map' else error, cmap='hot' if title == 'Error Map' else None)
        ax.set_title(title, color='white', fontweight='bold')
        ax.axis('off')

    fig.suptitle(f"{result['verdict']}  |  Score: {result['score']:.4f}  |  {result['action']}",
                 color=color, fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = save_path or os.path.join(config.OUTPUT_DIR, "inference_result.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--save', default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, thr = load_model(args.checkpoint, device)
    result = predict(args.image, G, thr, device)

    print(f"\n  {result['verdict']}")
    print(f"  Score: {result['score']:.4f} (threshold: {result['threshold']:.4f})")
    print(f"  {result['action']}\n")

    show_result(result, args.save)
