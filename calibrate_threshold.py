"""
GANomaly v2 — Threshold Calibration Script.
- Youden's J statistic optimization
- Constrained: max specificity at target sensitivity
- Saves threshold for the Flask app
"""

import os, sys, argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, f1_score)

import config
from models.ganomaly_v2 import Generator
from data.dataset import prepare_datasets, get_dataloaders
from utils import combined_anomaly_score, find_threshold, find_threshold_at_sensitivity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensitivity', type=float, default=0.85,
                        help='Target sensitivity (default: 0.85)')
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_v2.pth")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    G = Generator(
        cfg.get('channels', config.CHANNELS),
        cfg.get('latent_dim', config.LATENT_DIM),
        cfg.get('feature_maps', config.FEATURE_MAPS)
    ).to(device)
    G.load_state_dict(ckpt['G'])
    G.eval()

    # Load test data
    idrid_cfg = {'train_csv': config.IDRID_TRAIN_CSV, 'train_img': config.IDRID_TRAIN_IMG,
                 'test_csv': config.IDRID_TEST_CSV, 'test_img': config.IDRID_TEST_IMG}
    _, val_ds, test_ds = prepare_datasets(
        config.IMAGES_DIR, config.LABELS_CSV, config.IMAGE_SIZE,
        config.TRAIN_SPLIT, config.TEST_NORMAL_COUNT,
        normal_folder=config.NORMAL_DIR, idrid_config=idrid_cfg)
    _, _, test_loader = get_dataloaders(val_ds, val_ds, test_ds, config.BATCH_SIZE)

    # Collect scores
    all_labels, all_scores, all_grades = [], [], []
    with torch.no_grad():
        for images, labels, grades in tqdm(test_loader, desc="Scoring"):
            images = images.to(device)
            x_hat, z, z_hat = G(images)
            scores = combined_anomaly_score(z, z_hat, images, x_hat)
            all_labels.extend(labels.numpy())
            all_scores.extend(scores.cpu().numpy())
            all_grades.extend(grades.numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_grades = np.array(all_grades)

    # ── Youden's J threshold ──
    threshold_j, auc = find_threshold(all_labels, all_scores)

    # ── Sensitivity-constrained threshold ──
    target = args.sensitivity
    threshold_s, _ = find_threshold_at_sensitivity(all_labels, all_scores, target_sens=target)
    predictions = (all_scores >= threshold_s).astype(int)

    cm = confusion_matrix(all_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    # Read old threshold
    thresh_path = os.path.join(config.CHECKPOINT_DIR, "threshold_v2.txt")
    old_thresh = "N/A"
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            old_thresh = f.read().strip()

    print(f"\n{'='*55}")
    print(f"  Target sensitivity:  {target*100:.0f}%")
    print(f"  Youden J threshold:  {threshold_j:.4f}")
    print(f"  New threshold:       {threshold_s:.4f}  (was {old_thresh})")
    print(f"  Actual sensitivity:  {sens:.4f} ({sens*100:.1f}%)")
    print(f"  Actual specificity:  {spec:.4f} ({spec*100:.1f}%)")
    print(f"{'='*55}")

    print(classification_report(all_labels, predictions,
          target_names=['Healthy', 'DR Present']))
    print(f"  Sensitivity: {sens:.4f}  |  Specificity: {spec:.4f}")
    print(f"  F1: {f1_score(all_labels, predictions):.4f}  |  AUC: {auc:.4f}")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn} (missed!)  TN: {tn}")

    # Per-grade
    print(f"\nPer-Grade Detection:")
    for g in sorted(np.unique(all_grades)):
        mask = all_grades == g
        g_scores = all_scores[mask]
        detected = np.sum(g_scores >= threshold_s)
        total = len(g_scores)
        print(f"  Grade {int(g)}: {np.mean(g_scores):.4f} avg | "
              f"detected {detected}/{total} ({detected/total*100:.0f}%)")

    # Save threshold
    with open(thresh_path, 'w') as f:
        f.write(str(threshold_s))
    print(f"\nNew threshold saved to {thresh_path}")


if __name__ == "__main__":
    main()
