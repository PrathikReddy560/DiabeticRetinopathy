"""
GANomaly Evaluation — computes anomaly scores, finds optimal threshold,
prints medical metrics, and generates plots.
"""

import os, sys, argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models.ganomaly import Generator
from data.dataset import prepare_datasets, get_dataloaders
from utils import anomaly_score, find_threshold, plot_results, evaluate, save_reconstructions


def test(ckpt=None):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = ckpt or os.path.join(config.CHECKPOINT_DIR, "best.pth")

    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}\nTrain first with train_ganomaly.py")
        return

    # Data
    idrid_cfg = {'train_csv': config.IDRID_TRAIN_CSV, 'train_img': config.IDRID_TRAIN_IMG,
                 'test_csv': config.IDRID_TEST_CSV, 'test_img': config.IDRID_TEST_IMG}
    _, val_ds, test_ds = prepare_datasets(
        config.IMAGES_DIR, config.LABELS_CSV, config.IMAGE_SIZE,
        config.TRAIN_SPLIT, config.TEST_NORMAL_COUNT,
        normal_folder=config.NORMAL_DIR, idrid_config=idrid_cfg)
    _, val_loader, test_loader = get_dataloaders(val_ds, val_ds, test_ds, config.BATCH_SIZE)

    # Load model
    G = Generator(config.CHANNELS, config.LATENT_DIM, config.FEATURE_MAPS).to(device)
    G.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)['G'])
    G.eval()

    # Compute test scores
    scores, labels, grades = [], [], []
    sample_normal, sample_dr = None, None

    with torch.no_grad():
        for imgs, lbl, grd in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            recon, z, z_hat = G(imgs)
            scores.extend(anomaly_score(z, z_hat).cpu().numpy())
            labels.extend(lbl.numpy())
            grades.extend(grd.numpy())

            # Grab samples for visualization
            if sample_normal is None and (lbl == 0).any():
                idx = (lbl == 0).nonzero(as_tuple=True)[0][:4]
                sample_normal = (imgs[idx].cpu(), recon[idx].cpu())
            if sample_dr is None and (lbl == 1).any():
                idx = (lbl == 1).nonzero(as_tuple=True)[0][:4]
                sample_dr = (imgs[idx].cpu(), recon[idx].cpu())

    scores, labels, grades = np.array(scores), np.array(labels), np.array(grades)

    # Find threshold & evaluate
    threshold, auc = find_threshold(labels, scores)
    preds = (scores >= threshold).astype(int)
    metrics = evaluate(labels, preds, scores)

    # Per-grade breakdown
    print("\nPer-Grade Detection:")
    for g in range(5):
        mask = grades == g
        if mask.any():
            det = preds[mask].sum()
            print(f"  Grade {g}: {scores[mask].mean():.4f} avg | "
                  f"detected {det}/{mask.sum()} ({det/mask.sum()*100:.0f}%)")

    # Plots
    plot_results(labels, scores, threshold, config.OUTPUT_DIR)
    if sample_normal and sample_dr:
        real = torch.cat([sample_normal[0], sample_dr[0]])
        recon = torch.cat([sample_normal[1], sample_dr[1]])
        save_reconstructions(real, recon, "test", config.OUTPUT_DIR)

    # Save threshold
    with open(os.path.join(config.CHECKPOINT_DIR, "threshold.txt"), 'w') as f:
        f.write(f"{threshold}\n")

    print(f"\nAUC: {auc:.4f} | Sens: {metrics['sensitivity']:.4f} | "
          f"Spec: {metrics['specificity']:.4f} | Threshold: {threshold:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default=None)
    test(p.parse_args().checkpoint)
