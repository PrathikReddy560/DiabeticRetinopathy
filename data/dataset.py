"""
Dataset loader for GANomaly training.
Train: APTOS Grade 0 + ODIR Normal + IDRiD Grade 0 COMBINED (3-camera robustness)
Test: Full APTOS + IDRiD (all grades)
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import load_and_preprocess, get_train_transforms, get_test_transforms


class RetinalDataset(Dataset):
    def __init__(self, image_paths, labels, image_size=64, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_and_preprocess(self.image_paths[idx], self.image_size)
        grade = self.labels[idx]
        label = 0 if grade == 0 else 1  # binary: healthy vs DR

        if self.transform:
            image = self.transform(image)

        return image, label, grade


def _build_image_map(directory):
    """Recursively find all images in a directory and map basename -> full_path."""
    img_map = {}
    if os.path.exists(directory):
        for root, _, files in os.walk(directory):
            for f in files:
                basename, ext = os.path.splitext(f)
                if ext.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
                    img_map[basename] = os.path.join(root, f)
    return img_map


def _load_idrid(csv_path, img_dir):
    """Load IDRiD dataset: returns (paths, grades) for existing images."""
    if not os.path.exists(csv_path):
        return [], []

    df = pd.read_csv(csv_path)
    # Clean column names (IDRiD CSV has trailing spaces)
    df.columns = [c.strip() for c in df.columns]
    
    idrid_img_map = _build_image_map(img_dir)

    paths, grades = [], []
    for _, row in df.iterrows():
        name = str(row['Image name']).strip()
        grade = int(row['Retinopathy grade'])
        
        img_path = idrid_img_map.get(name)
        if img_path:
            paths.append(img_path)
            grades.append(grade)

    return paths, grades


def prepare_datasets(images_dir, labels_csv, image_size=64,
                     train_split=0.85, test_normal_count=150,
                     normal_folder=None, idrid_config=None):
    """
    Train/val: APTOS Grade 0 + ODIR Normal + IDRiD Grade 0 COMBINED.
    Test: Full APTOS + IDRiD (all grades).
    """
    # ── Source 1: APTOS ──
    df = pd.read_csv(labels_csv)
    
    aptos_img_map = _build_image_map(images_dir)
    df['image_path'] = df['id_code'].astype(str).apply(lambda x: aptos_img_map.get(x, ""))
    df = df[df['image_path'] != ""].reset_index(drop=True)

    print(f"[APTOS]  {len(df)} images found. Grade distribution:")
    print(df['diagnosis'].value_counts().sort_index().to_string())

    # APTOS Grade 0 for training
    aptos_paths, aptos_labels = [], []
    if len(df) > 0 and 0 in df['diagnosis'].values:
        aptos_healthy = df[df['diagnosis'] == 0].sample(frac=1, random_state=42).reset_index(drop=True)
        aptos_paths = aptos_healthy['image_path'].tolist()
        aptos_labels = [0] * len(aptos_paths)
    print(f"         → {len(aptos_paths)} healthy (Grade 0)")

    # ── Source 2: ODIR Normal folder ──
    odir_paths, odir_labels = [], []
    if normal_folder and os.path.isdir(normal_folder):
        odir_img_map = _build_image_map(normal_folder)
        odir_paths = list(odir_img_map.values())
        odir_labels = [0] * len(odir_paths)
        print(f"[ODIR]   {len(odir_paths)} normal images")
    else:
        print(f"[ODIR]   0 normal images")

    # ── Source 3: IDRiD ──
    idrid_healthy_paths, idrid_healthy_labels = [], []
    idrid_test_paths, idrid_test_labels = [], []

    if idrid_config:
        # Training set
        tr_paths, tr_grades = _load_idrid(idrid_config['train_csv'], idrid_config['train_img'])
        idrid_healthy_paths = [p for p, g in zip(tr_paths, tr_grades) if g == 0]
        idrid_healthy_labels = [0] * len(idrid_healthy_paths)

        # Testing set
        te_paths, te_grades = _load_idrid(idrid_config['test_csv'], idrid_config['test_img'])

        # All IDRiD images go to test set
        idrid_test_paths = tr_paths + te_paths
        idrid_test_labels = tr_grades + te_grades

        print(f"[IDRiD]  {len(tr_paths)} train + {len(te_paths)} test images found")
        print(f"         → {len(idrid_healthy_paths)} healthy (Grade 0) for training")
    else:
        print("[IDRiD]  0 train + 0 test images found")

    # ── Combine ALL healthy images for training ──
    all_healthy_paths = aptos_paths + odir_paths + idrid_healthy_paths
    all_healthy_labels = aptos_labels + odir_labels + idrid_healthy_labels

    # Shuffle combined data
    combined = list(zip(all_healthy_paths, all_healthy_labels))
    np.random.seed(42)
    np.random.shuffle(combined)
    all_healthy_paths, all_healthy_labels = zip(*combined)
    all_healthy_paths, all_healthy_labels = list(all_healthy_paths), list(all_healthy_labels)

    print(f"\n[Combined] {len(all_healthy_paths)} total healthy images for training")

    # ── Split into train/val ──
    split_idx = int(len(all_healthy_paths) * train_split)
    train_paths = all_healthy_paths[:split_idx]
    train_labels = all_healthy_labels[:split_idx]
    val_paths = all_healthy_paths[split_idx:]
    val_labels = all_healthy_labels[split_idx:]

    # ── Test data: ALL APTOS + ODIR sample + IDRiD ──
    test_paths = df['image_path'].tolist()
    test_labels = df['diagnosis'].tolist()

    # Add ODIR normals sample to test
    if odir_paths:
        np.random.seed(99)
        odir_test_sample = np.random.choice(odir_paths, size=min(200, len(odir_paths)), replace=False).tolist()
        test_paths.extend(odir_test_sample)
        test_labels.extend([0] * len(odir_test_sample))

    # Add IDRiD to test
    if idrid_test_paths:
        test_paths.extend(idrid_test_paths)
        test_labels.extend(idrid_test_labels)

    print(f"[Split]  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    train_ds = RetinalDataset(train_paths, train_labels, image_size, get_train_transforms(image_size))
    val_ds = RetinalDataset(val_paths, val_labels, image_size, get_test_transforms(image_size))
    test_ds = RetinalDataset(test_paths, test_labels, image_size, get_test_transforms(image_size))

    return train_ds, val_ds, test_ds


def get_dataloaders(train_ds, val_ds, test_ds, batch_size=32):
    """Wrap datasets in DataLoaders."""
    import torch
    cuda = torch.cuda.is_available()
    make = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                          num_workers=0 if not cuda else 2,
                                          pin_memory=cuda, drop_last=shuffle)
    return make(train_ds, True), make(val_ds, False), make(test_ds, False)
