"""
evaluate_efficientnet.py
========================
Complete evaluation script for the trained EfficientNet-B0 DR severity grader.

Performs two evaluation modes:
  1. Deterministic — standard forward pass with dropout OFF, computing accuracy,
     QWK, per-class & macro AUC, confusion matrix, and ROC curves.
  2. MC Dropout   — 30 stochastic forward passes with dropout ON, producing
     uncertainty-adjusted accuracy and per-grade confidence statistics.

Outputs:
  • Console report with all metrics
  • confusion_matrix.png  (seaborn heatmap)
  • roc_curves.png        (one curve per DR grade)

Usage:
    python evaluate_efficientnet.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # headless backend — safe for servers / CI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
)

# ── Local imports ────────────────────────────────────────────────────────────
from dataset import get_train_val_test_loaders
from model import DRClassifier

# ── Constants ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.path.join(_SCRIPT_DIR, "..", "checkpoints", "efficientnet_best.pth")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "output", "stage2")
MC_PASSES = 30
GRADE_NAMES = [
    "Grade 0 (No DR)",
    "Grade 1 (Mild)",
    "Grade 2 (Moderate)",
    "Grade 3 (Severe)",
    "Grade 4 (Proliferative)",
]
NUM_CLASSES = len(GRADE_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: load model from checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model(checkpoint_path: str) -> DRClassifier:
    """Instantiate DRClassifier and load the best checkpoint weights."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first or verify the path."
        )

    model = DRClassifier(pretrained=False)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Support both raw state-dicts and wrapped checkpoints
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.to(DEVICE)
    print(f"[✓] Model loaded from {checkpoint_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Deterministic evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_predictions(model, loader):
    """Run a single deterministic forward pass over the data loader.

    Returns:
        all_labels : np.ndarray of true grade labels
        all_preds  : np.ndarray of predicted grade labels
        all_probs  : np.ndarray (N, NUM_CLASSES) of softmax probabilities
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        images = batch[0].to(DEVICE)
        # The dataset may return (image, binary_label, grade) or (image, grade)
        # We always want the DR severity grade (0-4).
        labels = batch[-1]  # grade is always the last element

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs, axis=0),
    )


def _print_deterministic_metrics(labels, preds, probs):
    """Compute and print deterministic evaluation metrics."""
    # --- sklearn classification report ---
    print("\n" + classification_report(
        labels, preds, target_names=GRADE_NAMES, digits=4, zero_division=0,
    ))

    # --- Overall accuracy ---
    acc = accuracy_score(labels, preds)

    # --- Quadratic Weighted Kappa ---
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")

    # --- Macro AUC (one-vs-rest) ---
    try:
        macro_auc = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro",
        )
    except ValueError:
        macro_auc = float("nan")

    # --- Per-grade AUC (one-vs-rest) ---
    per_grade_auc = []
    for c in range(NUM_CLASSES):
        binary = (labels == c).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            per_grade_auc.append(float("nan"))
        else:
            per_grade_auc.append(roc_auc_score(binary, probs[:, c]))

    return acc, qwk, macro_auc, per_grade_auc


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def _save_confusion_matrix(labels, preds, output_dir):
    """Save an annotated confusion matrix heatmap as PNG."""
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=GRADE_NAMES,
        yticklabels=GRADE_NAMES,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("DR Grade Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Grade", fontsize=12)
    ax.set_ylabel("True Grade", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Confusion matrix saved → {path}")


def _save_roc_curves(labels, probs, per_grade_auc, output_dir):
    """Plot one-vs-rest ROC curves for all 5 grades and save as PNG."""
    from sklearn.metrics import roc_curve as sk_roc_curve

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for c in range(NUM_CLASSES):
        binary = (labels == c).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue  # skip grades with no true samples
        fpr, tpr, _ = sk_roc_curve(binary, probs[:, c])
        auc_val = per_grade_auc[c]
        ax.plot(fpr, tpr, color=colors[c], lw=2,
                label=f"{GRADE_NAMES[c]}  (AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — DR Grade Classification", fontsize=14,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] ROC curves saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MC Dropout evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _mc_dropout_evaluation(model, loader, num_passes: int = MC_PASSES):
    """Run *num_passes* stochastic forward passes with dropout active.

    Returns:
        labels    : np.ndarray (N,) — true grades
        mean_probs: np.ndarray (N, C) — averaged softmax probabilities
        std_probs : np.ndarray (N, C) — per-class standard deviation
    """
    model.eval()
    model.enable_mc_dropout()  # re-activate dropout while keeping BN in eval

    all_labels = []
    # Accumulate per-pass probabilities for each sample
    all_pass_probs = []  # list of (N_batch, C) arrays, one list per pass

    # First pass: also collect labels
    for pass_idx in range(num_passes):
        pass_probs = []
        pass_labels = []
        for batch in loader:
            images = batch[0].to(DEVICE)
            labels_batch = batch[-1]

            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pass_probs.append(probs)

            if pass_idx == 0:
                pass_labels.append(labels_batch.numpy())

        all_pass_probs.append(np.concatenate(pass_probs, axis=0))
        if pass_idx == 0:
            all_labels = np.concatenate(pass_labels)

        if (pass_idx + 1) % 10 == 0 or pass_idx == 0:
            print(f"  MC pass {pass_idx + 1}/{num_passes} completed")

    # Stack: (num_passes, N, C)
    stacked = np.stack(all_pass_probs, axis=0)
    mean_probs = stacked.mean(axis=0)   # (N, C)
    std_probs = stacked.std(axis=0)     # (N, C)

    return all_labels, mean_probs, std_probs


def _print_mc_dropout_metrics(labels, mean_probs, std_probs):
    """Compute and print MC Dropout uncertainty metrics."""
    mc_preds = np.argmax(mean_probs, axis=1)
    mc_acc = accuracy_score(labels, mc_preds)

    # Uncertainty per sample: std of the predicted class
    pred_std = std_probs[np.arange(len(mc_preds)), mc_preds]
    low_conf_mask = pred_std > 0.15
    n_low = low_conf_mask.sum()

    # Uncertainty-adjusted accuracy: accuracy on high-confidence samples only
    high_conf_mask = ~low_conf_mask
    if high_conf_mask.sum() > 0:
        ua_acc = accuracy_score(labels[high_conf_mask], mc_preds[high_conf_mask])
    else:
        ua_acc = 0.0

    # Per-grade stats
    print("\n  Per-grade MC Dropout statistics:")
    print(f"  {'Grade':<30s}  {'Mean Conf':>10s}  {'Uncertain %':>11s}")
    print("  " + "-" * 55)
    for g in range(NUM_CLASSES):
        mask_g = (labels == g)
        if mask_g.sum() == 0:
            continue
        # Mean confidence = mean of max probability for this grade's samples
        mean_conf = mean_probs[mask_g].max(axis=1).mean()
        unc_rate = (pred_std[mask_g] > 0.15).mean() * 100
        print(f"  {GRADE_NAMES[g]:<30s}  {mean_conf:>9.4f}  {unc_rate:>10.1f}%")

    return ua_acc, n_low, len(labels)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main evaluate() entry point
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate():
    """Full evaluation pipeline: deterministic + MC Dropout."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────────
    model = _load_model(CHECKPOINT_PATH)

    # ── 2. Get test data loader ──────────────────────────────────────────────
    _, _, test_loader, _ = get_train_val_test_loaders()
    print(f"[✓] Test set loaded: {len(test_loader.dataset)} samples\n")

    # ── 3. Deterministic evaluation ──────────────────────────────────────────
    labels, preds, probs = _collect_predictions(model, test_loader)
    acc, qwk, macro_auc, per_grade_auc = _print_deterministic_metrics(
        labels, preds, probs
    )

    # Save visualizations
    _save_confusion_matrix(labels, preds, OUTPUT_DIR)
    _save_roc_curves(labels, probs, per_grade_auc, OUTPUT_DIR)

    # ── 4. MC Dropout evaluation ─────────────────────────────────────────────
    print("\n[MC Dropout] Running stochastic evaluation …")
    mc_labels, mc_mean, mc_std = _mc_dropout_evaluation(model, test_loader)
    ua_acc, n_low, n_total = _print_mc_dropout_metrics(mc_labels, mc_mean, mc_std)

    # ── 5. Summary report ────────────────────────────────────────────────────
    auc_str = "[" + ", ".join(f"{a:.2f}" for a in per_grade_auc) + "]"
    banner = "=" * 60
    print(f"""
{banner}
  EVALUATION RESULTS — EfficientNet-B0 DR Grader
{banner}
[Deterministic]
  Accuracy:       {acc:.4f}
  QWK:            {qwk:.4f}
  Macro AUC:      {macro_auc:.4f}
  Per-grade AUC:  {auc_str}

[MC Dropout - {MC_PASSES} passes]
  Uncertainty-adjusted Accuracy: {ua_acc:.4f}
  Low confidence samples: {n_low}/{n_total} ({n_low / n_total * 100:.1f}%)
{banner}
""")

    # ── 6. Save metrics to JSON for downstream use ───────────────────────────
    import json
    metrics = {
        "accuracy": float(acc),
        "qwk": float(qwk),
        "macro_auc": float(macro_auc),
        "per_grade_auc": [float(a) for a in per_grade_auc],
        "mc_dropout_passes": MC_PASSES,
        "uncertainty_adjusted_accuracy": float(ua_acc),
        "low_confidence_count": int(n_low),
        "total_samples": int(n_total),
    }
    json_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] Metrics saved → {json_path}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    evaluate()
