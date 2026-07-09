"""
GANomaly v2 — Comprehensive Evaluation Script.
- Full test set evaluation with combined anomaly score
- Test-Time Augmentation (TTA)
- ROC curve, score distribution, confusion matrix plots
- Per-grade detection breakdown
- All metrics saved to results.json
"""

import os, sys, json, argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, f1_score, ConfusionMatrixDisplay)

import config
from models.ganomaly_v2 import Generator
from data.dataset import prepare_datasets, get_dataloaders
from data.preprocessing import load_and_preprocess, get_tta_transforms, denormalize
from utils import anomaly_score, combined_anomaly_score, find_threshold, find_threshold_at_sensitivity


def load_model(device):
    """Load best v2 model checkpoint."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_v2.pth")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    G = Generator(
        cfg.get('channels', config.CHANNELS),
        cfg.get('latent_dim', config.LATENT_DIM),
        cfg.get('feature_maps', config.FEATURE_MAPS)
    ).to(device)
    G.load_state_dict(ckpt['G'])
    G.eval()
    print(f"Loaded model from epoch {ckpt.get('epoch', '?')} "
          f"(val_score: {ckpt.get('val_score', '?'):.4f})")
    return G


def evaluate_test_set(G, test_loader, device, use_combined=True):
    """Run evaluation on the full test set. Returns labels, scores, grades."""
    all_labels, all_scores, all_grades = [], [], []

    with torch.no_grad():
        for images, labels, grades in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            x_hat, z, z_hat = G(images)

            if use_combined:
                scores = combined_anomaly_score(z, z_hat, images, x_hat)
            else:
                scores = anomaly_score(z, z_hat)

            all_labels.extend(labels.numpy())
            all_scores.extend(scores.cpu().numpy())
            all_grades.extend(grades.numpy())

    return np.array(all_labels), np.array(all_scores), np.array(all_grades)





def evaluate_with_tta(G, test_ds, device, num_augments=5):
    """Test-Time Augmentation: average scores over augmented views."""
    tta_transforms = get_tta_transforms(config.IMAGE_SIZE)
    all_labels, all_scores, all_grades = [], [], []

    print(f"Running TTA with {len(tta_transforms)} augmented views...")

    for idx in tqdm(range(len(test_ds)), desc="TTA Eval"):
        img_path = test_ds.image_paths[idx]
        grade = test_ds.labels[idx]
        label = 0 if grade == 0 else 1

        # Load raw image
        raw_img = load_and_preprocess(img_path, config.IMAGE_SIZE)

        # Score each augmented view
        view_scores = []
        for tfm in tta_transforms:
            tensor = tfm(raw_img).unsqueeze(0).to(device)
            with torch.no_grad():
                x_hat, z, z_hat = G(tensor)
                sc = combined_anomaly_score(z, z_hat, tensor, x_hat)
                view_scores.append(sc.item())

        # Average all views
        avg_score = np.mean(view_scores)
        all_labels.append(label)
        all_scores.append(avg_score)
        all_grades.append(grade)

    return np.array(all_labels), np.array(all_scores), np.array(all_grades)


def plot_roc_curve(labels, scores, output_dir):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color='#1565C0', lw=2.5, label=f'GANomaly v2 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1)
    plt.fill_between(fpr, tpr, alpha=0.15, color='#42A5F5')

    # Mark operating points
    j_stat = tpr - fpr
    best_j = np.argmax(j_stat)
    plt.plot(fpr[best_j], tpr[best_j], 'ro', ms=10,
             label=f'Youden J ({fpr[best_j]:.2f}, {tpr[best_j]:.2f})')

    # 85% sensitivity line
    sens_idx = np.argmin(np.abs(tpr - 0.85))
    plt.plot(fpr[sens_idx], tpr[sens_idx], 'g^', ms=10,
             label=f'85% Sens ({fpr[sens_idx]:.2f}, {tpr[sens_idx]:.2f})')

    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    plt.title(f'ROC Curve — GANomaly v2 DR Screening\nAUC = {auc:.4f}', fontsize=14)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve_v2.png"), dpi=200)
    plt.close()
    print(f"  ROC curve saved: output/roc_curve_v2.png")


def plot_score_distribution(labels, scores, grades, threshold, output_dir):
    """Plot anomaly score distributions for healthy vs each DR grade."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Binary: Healthy vs DR
    ax1.hist(scores[labels == 0], bins=60, alpha=0.7, label='Healthy',
             color='#4CAF50', density=True, edgecolor='none')
    ax1.hist(scores[labels == 1], bins=60, alpha=0.7, label='DR (all grades)',
             color='#F44336', density=True, edgecolor='none')
    ax1.axvline(threshold, color='#FF9800', ls='--', lw=2.5,
                label=f'Threshold = {threshold:.4f}')
    ax1.set_xlabel('Combined Anomaly Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Score Distribution: Healthy vs DR', fontsize=13)
    ax1.legend(fontsize=11)

    # Per-grade
    colors = ['#4CAF50', '#8BC34A', '#FF9800', '#F44336', '#B71C1C']
    grade_names = ['Grade 0\n(Healthy)', 'Grade 1\n(Mild)', 'Grade 2\n(Moderate)',
                   'Grade 3\n(Severe)', 'Grade 4\n(Proliferative)']
    grade_data = [scores[grades == g] for g in range(5)]

    parts = ax2.violinplot([d for d in grade_data if len(d) > 0],
                            positions=[i for i, d in enumerate(grade_data) if len(d) > 0],
                            showmeans=True, showmedians=True)
    ax2.axhline(threshold, color='#FF9800', ls='--', lw=2, label=f'Threshold')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(grade_names, fontsize=10)
    ax2.set_ylabel('Anomaly Score', fontsize=12)
    ax2.set_title('Score Distribution by DR Grade', fontsize=13)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distribution_v2.png"), dpi=200)
    plt.close()
    print(f"  Score distribution saved: output/score_distribution_v2.png")


def plot_confusion_matrix(labels, predictions, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Healthy', 'DR Present'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix — GANomaly v2', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_v2.png"), dpi=200)
    plt.close()
    print(f"  Confusion matrix saved: output/confusion_matrix_v2.png")


def per_grade_analysis(scores, grades, threshold):
    """Print detection rate for each DR grade."""
    print(f"\nPer-Grade Detection:")
    results = {}
    for g in sorted(np.unique(grades)):
        mask = grades == g
        g_scores = scores[mask]
        detected = np.sum(g_scores >= threshold)
        total = len(g_scores)
        rate = detected / total if total > 0 else 0
        print(f"  Grade {int(g)}: {np.mean(g_scores):.4f} avg | "
              f"detected {detected}/{total} ({rate*100:.0f}%)")
        results[f"grade_{int(g)}"] = {
            'mean_score': float(np.mean(g_scores)),
            'detected': int(detected),
            'total': int(total),
            'detection_rate': float(rate)
        }
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate GANomaly v2')
    parser.add_argument('--tta', action='store_true', help='Enable Test-Time Augmentation')
    parser.add_argument('--combined', action='store_true', default=True,
                        help='Use combined anomaly score (default: True)')
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # Load model
    G = load_model(device)

    # Load data
    idrid_cfg = {'train_csv': config.IDRID_TRAIN_CSV, 'train_img': config.IDRID_TRAIN_IMG,
                 'test_csv': config.IDRID_TEST_CSV, 'test_img': config.IDRID_TEST_IMG}
    _, val_ds, test_ds = prepare_datasets(
        config.IMAGES_DIR, config.LABELS_CSV, config.IMAGE_SIZE,
        config.TRAIN_SPLIT, config.TEST_NORMAL_COUNT,
        normal_folder=config.NORMAL_DIR, idrid_config=idrid_cfg)
    _, _, test_loader = get_dataloaders(val_ds, val_ds, test_ds, config.BATCH_SIZE)

    # ── Evaluate ──
    if args.tta:
        labels, scores, grades = evaluate_with_tta(G, test_ds, device)
    else:
        labels, scores, grades = evaluate_test_set(G, test_loader, device,
                                                    use_combined=args.combined)

    # ── Thresholds ──
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — GANomaly v2")
    print(f"{'='*60}")

    # Youden's J optimal threshold
    threshold_j, auc = find_threshold(labels, scores)
    preds_j = (scores >= threshold_j).astype(int)

    # 85% sensitivity threshold
    threshold_85, _ = find_threshold_at_sensitivity(labels, scores, target_sens=0.85)
    preds_85 = (scores >= threshold_85).astype(int)

    # ── Metrics at Youden's J ──
    print(f"\n--- At Youden's J Threshold ({threshold_j:.4f}) ---")
    cm_j = confusion_matrix(labels, preds_j)
    tn, fp, fn, tp = cm_j.ravel()
    sens_j = tp / (tp + fn) if (tp + fn) else 0
    spec_j = tn / (tn + fp) if (tn + fp) else 0
    f1_j = f1_score(labels, preds_j)
    print(classification_report(labels, preds_j, target_names=['Healthy', 'DR Present']))
    print(f"  Sensitivity: {sens_j:.4f}  |  Specificity: {spec_j:.4f}")
    print(f"  F1: {f1_j:.4f}  |  AUC: {auc:.4f}")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn} (missed!)  TN: {tn}")

    # ── Metrics at 85% Sensitivity ──
    print(f"\n--- At 85% Sensitivity Threshold ({threshold_85:.4f}) ---")
    cm_85 = confusion_matrix(labels, preds_85)
    tn2, fp2, fn2, tp2 = cm_85.ravel()
    sens_85 = tp2 / (tp2 + fn2) if (tp2 + fn2) else 0
    spec_85 = tn2 / (tn2 + fp2) if (tn2 + fp2) else 0
    f1_85 = f1_score(labels, preds_85)
    print(f"  Sensitivity: {sens_85:.4f}  |  Specificity: {spec_85:.4f}")
    print(f"  F1: {f1_85:.4f}  |  AUC: {auc:.4f}")
    print(f"  TP: {tp2}  FP: {fp2}  FN: {fn2} (missed!)  TN: {tn2}")

    # ── Per-Grade ──
    grade_results = per_grade_analysis(scores, grades, threshold_j)

    # ── Plots ──
    print(f"\nGenerating plots...")
    plot_roc_curve(labels, scores, config.OUTPUT_DIR)
    plot_score_distribution(labels, scores, grades, threshold_j, config.OUTPUT_DIR)
    plot_confusion_matrix(labels, preds_j, config.OUTPUT_DIR)

    # ── Save results.json ──
    results = {
        'model': 'GANomaly_v2',
        'image_size': config.IMAGE_SIZE,
        'test_samples': int(len(labels)),
        'auc': float(auc),
        'youden_j': {
            'threshold': float(threshold_j),
            'sensitivity': float(sens_j),
            'specificity': float(spec_j),
            'f1': float(f1_j),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        },
        'sensitivity_85': {
            'threshold': float(threshold_85),
            'sensitivity': float(sens_85),
            'specificity': float(spec_85),
            'f1': float(f1_85),
            'tp': int(tp2), 'fp': int(fp2), 'fn': int(fn2), 'tn': int(tn2),
        },
        'per_grade': grade_results,
        'tta_enabled': args.tta,
        'combined_score': args.combined,
    }

    results_path = os.path.join(config.OUTPUT_DIR, "results_v2.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  All metrics saved: {results_path}")

    # ── Save threshold for app.py ──
    thresh_path = os.path.join(config.CHECKPOINT_DIR, "threshold_v2.txt")
    with open(thresh_path, 'w') as f:
        f.write(str(threshold_85))
    print(f"  Threshold saved: {thresh_path}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  AUC:          {auc:.4f}")
    print(f"  Sens@Youden:  {sens_j:.4f}  |  Spec@Youden:  {spec_j:.4f}")
    print(f"  Sens@85%:     {sens_85:.4f}  |  Spec@85%:     {spec_85:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
