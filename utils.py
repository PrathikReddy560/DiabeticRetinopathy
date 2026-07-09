"""
Utility functions v2: weight init, SSIM loss, combined anomaly scoring,
visualization, metrics, and test-time augmentation.
"""

import os, json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, f1_score)
from data.preprocessing import denormalize


# ── Weight Initialization ────────────────────────────────────────

def init_weights(model):
    """Kaiming initialization (better for LeakyReLU networks)."""
    for m in model.modules():
        name = m.__class__.__name__
        if 'Conv' in name and hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0.2, nonlinearity='leaky_relu')
        elif 'BatchNorm' in name and hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)


# ── SSIM Loss ────────────────────────────────────────────────────

def _gaussian_kernel(size=11, sigma=1.5, channels=3):
    """Create a 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    g = g / g.sum()
    return g.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)


def ssim(x, y, window_size=11, sigma=1.5, val_range=2.0):
    """Compute SSIM between two image tensors (both in [-1,1]).
    Returns: scalar SSIM value (higher = more similar, max = 1.0).
    """
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    channels = x.size(1)

    kernel = _gaussian_kernel(window_size, sigma, channels).to(x.device)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2 = F.conv2d(x ** 2, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y ** 2, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_xy

    num   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denom = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / denom).mean()


def ssim_loss(x, y):
    """SSIM-based loss: 1 - SSIM (lower = better reconstruction)."""
    return 1.0 - ssim(x, y)


# ── Anomaly Scoring ──────────────────────────────────────────────

def anomaly_score(z, z_hat):
    """L2 distance between original and reconstructed latent vectors."""
    return torch.mean((z.view(z.size(0), -1) - z_hat.view(z_hat.size(0), -1)) ** 2, dim=1)


def combined_anomaly_score(z, z_hat, x, x_hat, w_lat=0.0, w_recon=0.0, w_ssim=1.0):
    """Combined anomaly score from 3 signals for more robust detection.
    - Latent distance: how much the bottleneck encoding changed
    - Reconstruction L1: pixel-level difference
    - SSIM error: structural difference
    (Currently configured for SSIM-only as it empirically yields the best AUC for DR)
    """
    # 1. Latent distance
    lat = torch.mean((z.view(z.size(0), -1) - z_hat.view(z_hat.size(0), -1)) ** 2, dim=1)

    # 2. Reconstruction L1 (per-image mean)
    recon = torch.mean(torch.abs(x - x_hat).view(x.size(0), -1), dim=1)

    # 3. SSIM error (per-image)
    ssim_err = torch.zeros(x.size(0), device=x.device)
    for i in range(x.size(0)):
        ssim_err[i] = 1.0 - ssim(x[i:i+1], x_hat[i:i+1])

    # Use raw weighted sum — no per-batch normalization!
    # Threshold calibration handles absolute scale differences.
    return w_lat * lat + w_recon * recon + w_ssim * ssim_err


# ── Gradient Penalty (WGAN-GP) ───────────────────────────────────

def gradient_penalty(D, real, fake, device):
    """Compute gradient penalty for training stability."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    pred, _ = D(interp)

    grads = torch.autograd.grad(
        outputs=pred, inputs=interp,
        grad_outputs=torch.ones_like(pred),
        create_graph=True, retain_graph=True
    )[0]
    grads = grads.view(grads.size(0), -1)
    penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


# ── Threshold Finding ────────────────────────────────────────────

def find_threshold(labels, scores):
    """Find optimal threshold using Youden's J statistic. Returns (threshold, auc)."""
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_stat = tpr - fpr
    best = np.argmax(j_stat)
    threshold = thresholds[best]
    print(f"[Threshold] AUC: {auc:.4f} | Threshold: {threshold:.4f} | "
          f"TPR: {tpr[best]:.4f} | FPR: {fpr[best]:.4f} | J: {j_stat[best]:.4f}")
    return threshold, auc


def find_threshold_at_sensitivity(labels, scores, target_sens=0.85):
    """Find threshold that achieves target sensitivity with max specificity."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.where(tpr >= target_sens)[0]
    if len(valid) == 0:
        return thresholds[0], roc_auc_score(labels, scores)
    # Among valid thresholds, pick the one with lowest FPR (highest specificity)
    best_idx = valid[np.argmin(fpr[valid])]
    return thresholds[best_idx], roc_auc_score(labels, scores)


# ── Visualization ────────────────────────────────────────────────

def save_reconstructions(original, reconstructed, epoch, output_dir, n=8):
    """Save original vs reconstructed comparison grid."""
    n = min(n, original.size(0))
    orig = denormalize(original[:n].cpu())
    recon = denormalize(reconstructed[:n].cpu())
    error = torch.abs(orig - recon)

    grid = make_grid(torch.cat([orig, recon, error]), nrow=n, padding=2)
    plt.figure(figsize=(n * 2, 6))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    ep_str = f"{epoch:03d}" if isinstance(epoch, int) else str(epoch)
    plt.title(f"Epoch {epoch} — Original / Reconstructed / Error")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"recon_epoch_{ep_str}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_losses(history, output_dir):
    """Plot training loss curves."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    keys = [('g_loss', '#2196F3', 'Generator Total'),
            ('d_loss', '#F44336', 'Discriminator'),
            ('con', '#4CAF50', 'Reconstruction (L1)'),
            ('lat', '#9C27B0', 'Latent (L2)'),
            ('ssim', '#FF9800', 'SSIM Loss'),
            ('val_auc', '#00BCD4', 'Validation AUC')]
    for ax, (key, color, title) in zip(axes.flat, keys):
        if key in history and history[key]:
            ax.plot(history[key], color=color, linewidth=1.5)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_losses_v2.png"), dpi=150)
    plt.close()


def plot_results(labels, scores, threshold, output_dir):
    """Plot score distribution and ROC curve."""
    labels, scores = np.array(labels), np.array(scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Score distribution
    ax1.hist(scores[labels == 0], bins=50, alpha=0.7, label='Healthy', color='#4CAF50', density=True)
    ax1.hist(scores[labels == 1], bins=50, alpha=0.7, label='DR', color='#F44336', density=True)
    ax1.axvline(threshold, color='orange', ls='--', lw=2, label=f'Threshold={threshold:.3f}')
    ax1.set_xlabel('Anomaly Score')
    ax1.set_title('Score Distribution')
    ax1.legend()

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ax2.plot(fpr, tpr, color='#2196F3', lw=2, label=f'AUC={auc:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_results_v2.png"), dpi=150)
    plt.close()


def evaluate(labels, predictions, scores):
    """Print classification report and medical metrics."""
    labels, predictions = np.array(labels), np.array(predictions)
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    auc = roc_auc_score(labels, scores)

    print("\n" + classification_report(labels, predictions,
          target_names=['Healthy', 'DR Present']))
    print(f"  Sensitivity: {sens:.4f}  |  Specificity: {spec:.4f}")
    print(f"  F1: {f1_score(labels, predictions):.4f}  |  AUC: {auc:.4f}")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn} (missed!)  TN: {tn}")

    return {'sensitivity': sens, 'specificity': spec, 'auc': auc,
            'f1': f1_score(labels, predictions), 'cm': cm.tolist(),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}
