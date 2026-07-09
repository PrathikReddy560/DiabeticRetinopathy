"""
train_efficientnet.py (v2)
===========================
Enhanced training script for EfficientNet-B0 DR severity grader.

Improvements over v1:
  - Focal Loss (replaces CrossEntropy) — focuses on hard examples
  - Label Smoothing (0.1) — prevents overconfident predictions
  - Mixup augmentation — blends images for better generalization
  - Higher Phase 2 LR (3e-5 vs 1e-5) — backbone actually learns
  - Full backbone unfreeze in Phase 2 — all layers fine-tuned
  - 80 total epochs (20 Phase 1 + 60 Phase 2) — more time to converge
  - Patience increased to 15

Metrics tracked every epoch:
  - Focal loss (class-weighted)
  - Accuracy
  - Quadratic Weighted Kappa (QWK)  — primary checkpoint metric
  - Macro AUC (one-vs-rest)
  - Per-class F1 scores (grades 0-4)

Mixed-precision training (AMP) is used when a CUDA device is available.
"""

import os
import sys
import json
import time
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import cohen_kappa_score, roc_auc_score, f1_score

# ---------------------------------------------------------------------------
# Sibling-module imports
# ---------------------------------------------------------------------------
from dataset import DRGradingDataset, get_train_val_test_loaders
from model import DRClassifier

# ===================================================================
# Training configuration (v2 — improved)
# ===================================================================
BATCH_SIZE = 32
NUM_WORKERS = 4

PHASE1_EPOCHS = 20          # Epochs with frozen backbone
PHASE2_EPOCHS = 60          # Epochs with fine-tuning (was 30)
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS  # 80

PHASE1_LR = 1e-3            # Learning rate for Phase 1 (head only)
PHASE2_LR = 3e-5            # Learning rate for Phase 2 (was 1e-5)

PATIENCE = 15               # Early-stopping patience (was 10)

# Focal Loss parameters
FOCAL_GAMMA = 2.0           # Focus on hard examples
LABEL_SMOOTHING = 0.1       # Prevent overconfident predictions

# Mixup parameters
MIXUP_ALPHA = 0.2           # Beta distribution alpha for mixup
MIXUP_PROB = 0.5            # Probability of applying mixup per batch

# Paths — relative to the project root (one level above this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, '..', 'checkpoints')
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, '..', 'output', 'stage2')


# ===================================================================
# Focal Loss implementation
# ===================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Reduces the contribution of easy examples and focuses training
    on hard, misclassified samples. Crucial for imbalanced DR grading
    where Grades 1 and 3 are rare.

    L_focal = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        weight: Per-class weight tensor (inverse frequency).
        gamma: Focusing parameter. Higher = more focus on hard examples.
        label_smoothing: Smooths one-hot targets to prevent overconfidence.
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw model output
            targets: (B,) integer class labels
        """
        num_classes = logits.size(1)

        # Apply label smoothing to targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1),
                                    1.0 - self.label_smoothing)

        # Compute log-softmax
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Combine: -alpha * focal_weight * smooth_target * log(p)
        loss = -focal_weight * smooth_targets * log_probs

        # Apply class weights
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

        return loss.sum(dim=1).mean()


# ===================================================================
# Mixup augmentation
# ===================================================================

def mixup_data(x, y, alpha=0.2):
    """
    Apply Mixup augmentation: blend two random images together.

    This creates virtual training examples that lie between real samples,
    which dramatically improves generalization on small datasets.

    Args:
        x: Input batch (B, C, H, W)
        y: Label batch (B,)
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        mixed_x, y_a, y_b, lam
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup: weighted sum of losses for both targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===================================================================
# Helper utilities
# ===================================================================

def _setup_device():
    """Select the best available device (CUDA → CPU) and log it."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[WARNING] CUDA not available — training on CPU (will be slow).")
    return device


def _ensure_dirs():
    """Create checkpoint and output directories if they don't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================================
# Metric computation
# ===================================================================

def compute_metrics(y_true, y_pred, y_probs):
    """Compute all validation metrics.

    Args:
        y_true:  np.ndarray of ground-truth labels (N,)
        y_pred:  np.ndarray of predicted labels (N,)
        y_probs: np.ndarray of softmax probabilities (N, 5)

    Returns:
        dict with keys: accuracy, qwk, auc, f1_per_class
    """
    accuracy = (y_pred == y_true).sum() / len(y_true)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')

    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4])

    return {
        'accuracy': float(accuracy),
        'qwk': float(qwk),
        'auc': float(auc),
        'f1_per_class': f1_per_class.tolist(),
    }


# ===================================================================
# Single training epoch (with Mixup)
# ===================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    use_mixup=True):
    """Run one training epoch with mixed-precision and optional Mixup.

    Returns:
        avg_loss: mean training loss over all batches
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # --- Mixup augmentation ---
        apply_mixup = use_mixup and np.random.random() < MIXUP_PROB

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            if apply_mixup:
                mixed_images, y_a, y_b, lam = mixup_data(
                    images, labels, alpha=MIXUP_ALPHA
                )
                logits = model(mixed_images)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

        # --- Backward pass with gradient scaling ---
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    avg_loss = running_loss / max(num_samples, 1)
    return avg_loss


# ===================================================================
# Validation
# ===================================================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation and return loss + all metric arrays."""
    model.eval()
    running_loss = 0.0
    num_samples = 0

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / max(num_samples, 1)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_pred, y_probs)
    return avg_loss, metrics


# ===================================================================
# Main training routine
# ===================================================================

def train():
    """Execute the full two-phase training pipeline (v2)."""

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    device = _setup_device()
    _ensure_dirs()

    print("=" * 70)
    print(" Diabetic Retinopathy — EfficientNet-B0 Training (v2)")
    print("=" * 70)
    print(f" Focal Loss (gamma={FOCAL_GAMMA}) + Label Smoothing ({LABEL_SMOOTHING})")
    print(f" Mixup (alpha={MIXUP_ALPHA}, p={MIXUP_PROB})")
    print(f" Phase 1: {PHASE1_EPOCHS} epochs, LR={PHASE1_LR}")
    print(f" Phase 2: {PHASE2_EPOCHS} epochs, LR={PHASE2_LR} (full unfreeze)")
    print(f" Patience: {PATIENCE}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 2. Data loaders
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, class_weights = get_train_val_test_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Move class weights to device
    class_weights_tensor = class_weights.to(device)

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = DRClassifier()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] DRClassifier: {total_params:,} total parameters")

    # ------------------------------------------------------------------
    # 4. Loss function — Focal Loss with class weights + label smoothing
    # ------------------------------------------------------------------
    criterion = FocalLoss(
        weight=class_weights_tensor,
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
    )

    # ------------------------------------------------------------------
    # 5. Mixed-precision scaler
    # ------------------------------------------------------------------
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ------------------------------------------------------------------
    # 6. Training history storage
    # ------------------------------------------------------------------
    history = {
        'train_loss': [], 'val_loss': [],
        'accuracy': [], 'qwk': [], 'auc': [],
        'f1_per_class': [], 'phase': [], 'lr': [],
    }

    best_qwk = -1.0
    best_epoch = -1
    patience_counter = 0
    start_time = time.time()

    # ==================================================================
    # PHASE 1 — Frozen backbone (epochs 1 .. PHASE1_EPOCHS)
    # ==================================================================
    print("\n" + "─" * 70)
    print(f" Phase 1: Frozen backbone training (epochs 1-{PHASE1_EPOCHS})")
    print("─" * 70)

    model.freeze_backbone()

    optimizer = torch.optim.Adam(
        model.grade_head.parameters(),
        lr=PHASE1_LR,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2,
    )

    for epoch in range(1, TOTAL_EPOCHS + 1):

        # --- Phase transition ---
        if epoch == PHASE1_EPOCHS + 1:
            print("\n" + "═" * 70)
            print(f" Phase 2: Full fine-tuning (epochs {PHASE1_EPOCHS + 1}-{TOTAL_EPOCHS})")
            print(f" Unfreezing ENTIRE backbone, LR → {PHASE2_LR}")
            print("═" * 70)

            # Unfreeze ALL backbone layers (not just 30)
            for param in model.backbone.parameters():
                param.requires_grad = True

            # New optimizer for all trainable params
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=PHASE2_LR,
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2,
            )

            # Reset early-stopping for new phase
            patience_counter = 0

        phase_label = "Phase 1" if epoch <= PHASE1_EPOCHS else "Phase 2"

        # --- Train (with Mixup in Phase 2 only) ---
        use_mixup = (epoch > PHASE1_EPOCHS)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_mixup=use_mixup,
        )

        # --- Validate ---
        val_loss, metrics = validate(model, val_loader, criterion, device)

        # Step LR scheduler
        scheduler.step(epoch)

        # --- Record history ---
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(metrics['accuracy'])
        history['qwk'].append(metrics['qwk'])
        history['auc'].append(metrics['auc'])
        history['f1_per_class'].append(metrics['f1_per_class'])
        history['phase'].append(phase_label)
        history['lr'].append(current_lr)

        # --- Pretty-print epoch summary ---
        f1_str = [f"{v:.2f}" for v in metrics['f1_per_class']]
        saved_marker = ""

        # --- Checkpointing (best QWK) ---
        if metrics['qwk'] > best_qwk:
            best_qwk = metrics['qwk']
            best_epoch = epoch
            patience_counter = 0

            ckpt_path = os.path.join(CHECKPOINT_DIR, 'efficientnet_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_qwk': best_qwk,
                'class_weights': class_weights,
            }, ckpt_path)
            saved_marker = " ★ saved!"
        else:
            patience_counter += 1

        print(
            f"[Epoch {epoch}/{TOTAL_EPOCHS} | {phase_label}] "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"QWK: {metrics['qwk']:.4f} | "
            f"AUC: {metrics['auc']:.4f}"
            f"{saved_marker}"
        )
        print(f"  F1 per grade: [{', '.join(f1_str)}]")

        # --- Early stopping ---
        if patience_counter >= PATIENCE:
            print(
                f"\n[EARLY STOP] No QWK improvement for {PATIENCE} epochs. "
                f"Stopping at epoch {epoch}."
            )
            break

    # ==================================================================
    # Training summary
    # ==================================================================
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    print("\n" + "=" * 70)
    print(" Training Complete (v2)")
    print("=" * 70)
    print(f"  Best QWK  : {best_qwk:.4f}")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Total Time: {elapsed_str}")
    print(f"  Checkpoint: {os.path.join(CHECKPOINT_DIR, 'efficientnet_best.pth')}")

    # Save training history
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History   : {history_path}")
    print("=" * 70)


# ===================================================================
# Entry point
# ===================================================================
if __name__ == '__main__':
    train()
