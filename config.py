"""Configuration for GANomaly v2 — update DATASET paths before running."""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Check if running on Kaggle
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

if IS_KAGGLE:
    # On Kaggle, datasets are mounted in /kaggle/input
    # Based on user output, the dataset is uploaded under 'datasets'
    DATASET_DIR = "/kaggle/input/datasets"
    
    # Check if it's nested (e.g. /kaggle/input/datasets/dataset/train.csv) due to zipping
    if not os.path.exists(os.path.join(DATASET_DIR, "train.csv")) and os.path.exists(os.path.join(DATASET_DIR, "dataset", "train.csv")):
        DATASET_DIR = os.path.join(DATASET_DIR, "dataset")
        
    IMAGES_DIR   = os.path.join(DATASET_DIR, "train_images")
    LABELS_CSV   = os.path.join(DATASET_DIR, "train.csv")
    NORMAL_DIR   = os.path.join(DATASET_DIR, "normal") 
    
    # IDRiD dataset
    IDRID_DIR       = os.path.join(DATASET_DIR, "IDRiD", "B. Disease Grading")
    IDRID_TRAIN_IMG = os.path.join(IDRID_DIR, "1. Original Images", "a. Training Set")
    IDRID_TEST_IMG  = os.path.join(IDRID_DIR, "1. Original Images", "b. Testing Set")
    IDRID_TRAIN_CSV = os.path.join(IDRID_DIR, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
    IDRID_TEST_CSV  = os.path.join(IDRID_DIR, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")

    # Outputs on Kaggle must go to /kaggle/working
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    OUTPUT_DIR     = "/kaggle/working/output"
else:
    # ── Paths (Local) ─────────────────────────────────────────────
    DATASET_DIR  = os.path.join(ROOT, "dataset")
    IMAGES_DIR   = os.path.join(DATASET_DIR, "train_images")   # APTOS images
    LABELS_CSV   = os.path.join(DATASET_DIR, "train.csv")       # APTOS labels
    NORMAL_DIR   = os.path.join(DATASET_DIR, "normal")           # ODIR normal eyes

    # IDRiD dataset (3rd camera source)
    IDRID_DIR       = os.path.join(DATASET_DIR, "IDRiD", "B. Disease Grading")
    IDRID_TRAIN_IMG = os.path.join(IDRID_DIR, "1. Original Images", "a. Training Set")
    IDRID_TEST_IMG  = os.path.join(IDRID_DIR, "1. Original Images", "b. Testing Set")
    IDRID_TRAIN_CSV = os.path.join(IDRID_DIR, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
    IDRID_TEST_CSV  = os.path.join(IDRID_DIR, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")

    CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
    OUTPUT_DIR     = os.path.join(ROOT, "output")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model (v2) ───────────────────────────────────────
IMAGE_SIZE   = 128        # was 64 — captures microaneurysms
CHANNELS     = 3
LATENT_DIM   = 100
FEATURE_MAPS = 64

# ── Training ─────────────────────────────────────────
EPOCHS        = 200       # was 100
BATCH_SIZE    = 16        # was 32, reduced for 128×128 VRAM
LR            = 1e-4      # was 2e-4, more stable with GP
BETAS         = (0.5, 0.999)

# Loss weights: L = w_adv·L_adv + w_con·L_recon + w_lat·L_latent + w_ssim·L_ssim
W_ADV  = 1.0
W_CON  = 50.0
W_LAT  = 1.0
W_SSIM = 10.0             # NEW: SSIM structural loss

# Gradient penalty weight (WGAN-GP style)
GP_WEIGHT = 10.0

# Early stopping patience (epochs without val AUC improvement)
PATIENCE = 30

# ── Dataset split ────────────────────────────────────
TRAIN_SPLIT       = 0.85
TEST_NORMAL_COUNT = 150
ANOMALY_THRESHOLD = 0.5   # overridden after calibration

# ── Device & Logging ─────────────────────────────────
DEVICE             = "cuda" if __import__('torch').cuda.is_available() else "cpu"
SAVE_EVERY         = 25   # epochs
VIZ_EVERY          = 10   # epochs
