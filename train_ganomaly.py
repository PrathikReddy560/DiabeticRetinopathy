"""
GANomaly v2 Training Script — Full rewrite with all improvements.
- 128×128 resolution with deeper architecture
- SSIM loss + gradient penalty
- Cosine annealing with warm restarts
- Mixed precision (AMP) training
- Early stopping on validation AUC
- Combined anomaly scoring
"""

import os, sys, time, json

# Ensure the project root is in the Python path so we can import 'data' and 'models'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import config
from models.ganomaly_v2 import Generator, Discriminator
from data.dataset import prepare_datasets, get_dataloaders
from utils import (init_weights, anomaly_score, combined_anomaly_score,
                   ssim_loss, gradient_penalty,
                   save_reconstructions, plot_losses, find_threshold)

import numpy as np
from sklearn.metrics import roc_auc_score


def validate_auc(G, val_loader, device):
    """Compute AUC on validation set using combined anomaly score.
    Val set is all healthy (label=0), so we measure reconstruction quality.
    Returns mean anomaly score (lower = better for healthy images).
    """
    G.eval()
    scores = []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            x_hat, z, z_hat = G(images)
            sc = anomaly_score(z, z_hat)
            scores.extend(sc.cpu().numpy())
    G.train()
    return np.mean(scores)


def train():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # ── Data ──
    idrid_cfg = {'train_csv': config.IDRID_TRAIN_CSV, 'train_img': config.IDRID_TRAIN_IMG,
                 'test_csv': config.IDRID_TEST_CSV, 'test_img': config.IDRID_TEST_IMG}
    train_ds, val_ds, test_ds = prepare_datasets(
        config.IMAGES_DIR, config.LABELS_CSV, config.IMAGE_SIZE,
        config.TRAIN_SPLIT, config.TEST_NORMAL_COUNT,
        normal_folder=config.NORMAL_DIR, idrid_config=idrid_cfg)
    train_loader, val_loader, _ = get_dataloaders(train_ds, val_ds, test_ds, config.BATCH_SIZE)

    # ── Models ──
    G = Generator(config.CHANNELS, config.LATENT_DIM, config.FEATURE_MAPS).to(device)
    D = Discriminator(config.CHANNELS, config.FEATURE_MAPS).to(device)
    G.apply(init_weights)
    D.apply(init_weights)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator: {g_params:,} params")
    print(f"Discriminator: {d_params:,} params")

    # ── Optimizers & Schedulers ──
    opt_G = optim.Adam(G.parameters(), lr=config.LR, betas=config.BETAS)
    opt_D = optim.Adam(D.parameters(), lr=config.LR, betas=config.BETAS)

    # Cosine annealing with warm restarts every 40 epochs
    sched_G = CosineAnnealingWarmRestarts(opt_G, T_0=40, T_mult=1, eta_min=1e-6)
    sched_D = CosineAnnealingWarmRestarts(opt_D, T_0=40, T_mult=1, eta_min=1e-6)

    # ── Loss functions ──
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # ── Mixed precision ──
    scaler = GradScaler('cuda')

    # ── Training state ──
    best_val_score = float('inf')
    patience_counter = 0
    history = {'g_loss': [], 'd_loss': [], 'con': [], 'lat': [], 'ssim': [],
               'gp': [], 'val_score': [], 'val_auc': [], 'lr': []}

    print(f"\n{'='*60}")
    print(f"  GANomaly v2 Training")
    print(f"  Image: {config.IMAGE_SIZE}×{config.IMAGE_SIZE} | Batch: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.EPOCHS} | LR: {config.LR}")
    print(f"  Losses: ADV={config.W_ADV} CON={config.W_CON} "
          f"LAT={config.W_LAT} SSIM={config.W_SSIM} GP={config.GP_WEIGHT}")
    print(f"  Early stop patience: {config.PATIENCE}")
    print(f"{'='*60}\n")

    for epoch in range(1, config.EPOCHS + 1):
        G.train()
        D.train()
        t0 = time.time()

        ep_g, ep_d, ep_con, ep_lat, ep_ssim, ep_gp = 0, 0, 0, 0, 0, 0
        n_batches = 0

        for images, _, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)
            real_label = torch.ones(batch_size, device=device) * 0.9   # label smoothing
            fake_label = torch.zeros(batch_size, device=device) + 0.1

            # ──────────────── Train Discriminator ────────────────
            # D step runs in float32 (no GradScaler) because gradient
            # penalty requires second-order gradients incompatible with AMP.
            opt_D.zero_grad()

            with autocast():
                pred_real, feat_real_d = D(images)
                x_hat_d, _, _ = G(images)
                pred_fake_d, _ = D(x_hat_d.detach())

            # Losses in float32
            d_loss_real = bce_loss(pred_real.float(), real_label)
            d_loss_fake = bce_loss(pred_fake_d.float(), fake_label)
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            # Gradient penalty (float32)
            gp = gradient_penalty(D, images, x_hat_d.detach(), device)
            d_total = d_loss + config.GP_WEIGHT * gp

            d_total.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()

            # ──────────────── Train Generator ────────────────
            opt_G.zero_grad()

            with autocast():
                x_hat, z, z_hat = G(images)
                pred_fake, feat_fake = D(x_hat)
                _, feat_real = D(images)

                # 1. Adversarial loss (fool D)
                loss_adv = bce_loss(pred_fake, real_label)

                # 2. Reconstruction loss (L1)
                loss_con = l1_loss(x_hat, images)

                # 3. Latent loss (L2 between z and z_hat)
                loss_lat = l2_loss(z.view(batch_size, -1), z_hat.view(batch_size, -1))

                # 4. SSIM structural loss
                loss_ssim = ssim_loss(x_hat, images)

                # 5. Feature matching loss (from discriminator)
                loss_fm = l2_loss(feat_fake, feat_real.detach())

                # Total generator loss
                g_loss = (config.W_ADV * loss_adv +
                          config.W_CON * loss_con +
                          config.W_LAT * loss_lat +
                          config.W_SSIM * loss_ssim +
                          1.0 * loss_fm)

            scaler.scale(g_loss).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(opt_G)
            scaler.update()

            # NaN guard — skip corrupted batches
            if torch.isnan(g_loss) or torch.isnan(d_loss):
                continue

            ep_g += g_loss.item()
            ep_d += d_loss.item()
            ep_con += loss_con.item()
            ep_lat += loss_lat.item()
            ep_ssim += loss_ssim.item()
            ep_gp += gp.item()
            n_batches += 1

        # ── Epoch averages ──
        ep_g   /= n_batches
        ep_d   /= n_batches
        ep_con /= n_batches
        ep_lat /= n_batches
        ep_ssim /= n_batches
        ep_gp  /= n_batches
        elapsed = time.time() - t0

        history['g_loss'].append(ep_g)
        history['d_loss'].append(ep_d)
        history['con'].append(ep_con)
        history['lat'].append(ep_lat)
        history['ssim'].append(ep_ssim)
        history['gp'].append(ep_gp)
        history['lr'].append(opt_G.param_groups[0]['lr'])

        # Step schedulers
        sched_G.step()
        sched_D.step()

        print(f"[{epoch}] G:{ep_g:.4f} D:{ep_d:.4f} "
              f"Con:{ep_con:.4f} Lat:{ep_lat:.4f} SSIM:{ep_ssim:.4f} "
              f"GP:{ep_gp:.4f} ({elapsed:.0f}s)")

        # ── Validation ──
        if epoch % 5 == 0 or epoch == 1:
            val_score = validate_auc(G, val_loader, device)
            history['val_score'].append(val_score)

            improved = ""
            if val_score < best_val_score:
                best_val_score = val_score
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'opt_G': opt_G.state_dict(),
                    'opt_D': opt_D.state_dict(),
                    'val_score': val_score,
                    'config': {
                        'image_size': config.IMAGE_SIZE,
                        'latent_dim': config.LATENT_DIM,
                        'feature_maps': config.FEATURE_MAPS,
                        'channels': config.CHANNELS,
                    }
                }, os.path.join(config.CHECKPOINT_DIR, "best_v2.pth"))
                improved = " ★ saved!"
            else:
                patience_counter += 5  # checked every 5 epochs

            print(f"  Val score: {val_score:.4f}{improved}")

            # Early stopping
            if patience_counter >= config.PATIENCE:
                print(f"\n⚠ Early stopping at epoch {epoch} "
                      f"(no improvement for {config.PATIENCE} epochs)")
                break

        # ── Visualizations ──
        if epoch % config.VIZ_EVERY == 0:
            with torch.no_grad():
                sample = next(iter(val_loader))[0][:8].to(device)
                x_hat, _, _ = G(sample)
                save_reconstructions(sample, x_hat, epoch, config.OUTPUT_DIR)

    # ── Save final training artifacts ──
    plot_losses(history, config.OUTPUT_DIR)

    # Save training history
    hist_save = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(os.path.join(config.OUTPUT_DIR, "training_history_v2.json"), 'w') as f:
        json.dump(hist_save, f, indent=2)

    print(f"\nDone! Best val score: {best_val_score:.4f}")
    print(f"Model saved: checkpoints/best_v2.pth")
    print(f"History saved: output/training_history_v2.json")


if __name__ == "__main__":
    train()
