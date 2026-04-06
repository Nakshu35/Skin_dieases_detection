# =============================================================
# dataset.py  —  Load, preprocess, and serve all three datasets
# =============================================================
#
# What this file does step by step:
#   1. Reads each CSV (HAM10000, PAD-UFES-20, Dermaco-In)
#   2. Maps labels to a unified disease class index
#   3. Encodes metadata (age, sex, location, skin_type) into numbers
#   4. Applies image augmentations for training
#   5. Returns (image_tensor, metadata_tensor, label) for each sample
#
# NOTE: Column names below match standard dataset formats.
#       If your CSV has different column names, change the
#       COLUMN_MAP dictionaries at the top.

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

import config   # our config file


# ── How each dataset labels diseases → unified class name ─────
# Add more mappings if your dataset uses different label strings

HAM_LABEL_MAP = {
    "mel":   "melanoma",
    "bcc":   "basal_cell_carcinoma",
    "scc":   "squamous_cell_carcinoma",
    "akiec": "actinic_keratosis",
    "df":    "dermatofibroma",
    "nv":    "nevus",
    "vasc":  "vascular_lesion",
}

PAD_LABEL_MAP = {
    "MEL":  "melanoma",
    "BCC":  "basal_cell_carcinoma",
    "SCC":  "squamous_cell_carcinoma",
    "ACK":  "actinic_keratosis",
    "SEK":  "dermatofibroma",       # seborrheic keratosis → closest match
    "NEV":  "nevus",
}

DERM_LABEL_MAP = {
    # Adjust these to match your Dermaco-In label strings
    "melanoma":              "melanoma",
    "basal_cell_carcinoma":  "basal_cell_carcinoma",
    "squamous_cell_carcinoma": "squamous_cell_carcinoma",
    "actinic_keratosis":     "actinic_keratosis",
    "dermatofibroma":        "dermatofibroma",
    "nevus":                 "nevus",
    "vascular_lesion":       "vascular_lesion",
}

# Class name → integer index (built from config.DISEASE_CLASSES list)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(config.DISEASE_CLASSES)}


# =============================================================
# Helper: encode metadata into numbers
# =============================================================

def encode_sex(value):
    """Male=1, Female=0, Unknown=-1"""
    if pd.isna(value):
        return -1.0
    v = str(value).lower().strip()
    if v in ("male", "m", "1"):
        return 1.0
    if v in ("female", "f", "0"):
        return 0.0
    return -1.0


def encode_location(value, location_vocab):
    """Map body location string to integer index. Unknown → -1"""
    if pd.isna(value):
        return -1.0
    v = str(value).lower().strip()
    return float(location_vocab.get(v, -1))


def normalize_age(value, max_age=100.0):
    """Scale age to 0-1 range. Missing → -1"""
    if pd.isna(value):
        return -1.0
    return float(value) / max_age


def encode_skin_type(value):
    """Fitzpatrick skin type I-VI → 0-5. Missing → -1"""
    if pd.isna(value):
        return -1.0
    try:
        return float(int(value) - 1) / 5.0   # normalise to 0-1
    except (ValueError, TypeError):
        return -1.0


# Build a shared location vocabulary from all datasets so the
# same integer always means the same body site
LOCATION_VOCAB = {
    "scalp": 0, "face": 1, "ear": 2, "neck": 3,
    "chest": 4, "abdomen": 5, "back": 6,
    "upper extremity": 7, "lower extremity": 8,
    "acral": 9, "genital": 10, "oral/genital": 11,
    "hand": 12, "foot": 13, "trunk": 14,
    "forearm": 15, "arm": 16, "leg": 17,
}


# =============================================================
# Core Dataset Class
# =============================================================

class SkinDiseaseDataset(Dataset):
    """
    A single unified dataset class that works for all three datasets.

    Parameters
    ----------
    dataframe  : pd.DataFrame  — rows from the CSV (already filtered for split)
    img_dir    : str           — folder where images live
    label_map  : dict          — maps raw label string → unified class name
    transform  : torchvision transform — applied to each image
    is_train   : bool          — True = apply augmentations
    """

    def __init__(self, dataframe, img_dir, label_map, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.img_dir   = img_dir
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── 1. Load image ──────────────────────────────────────
        img_name = str(row["image_id"])
        # Try both .jpg and .png extensions
        img_path = os.path.join(self.img_dir, img_name + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, img_name + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, img_name)   # already has extension

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # ── 2. Build metadata vector ───────────────────────────
        age      = normalize_age(row.get("age", np.nan))
        sex      = encode_sex(row.get("sex", np.nan))
        location = encode_location(row.get("localization", row.get("region", np.nan)),
                                   LOCATION_VOCAB)
        skin     = encode_skin_type(row.get("skin_type", np.nan))

        metadata = torch.tensor([age, sex, location, skin], dtype=torch.float32)

        # ── 3. Get label ───────────────────────────────────────
        raw_label     = str(row["dx"]).strip()
        unified_label = self.label_map.get(raw_label, None)

        if unified_label is None:
            # Skip unknown labels by defaulting to nevus (least harmful fallback)
            # In practice, clean your CSVs to avoid this
            unified_label = "nevus"

        label = CLASS_TO_IDX[unified_label]

        return image, metadata, label


# =============================================================
# Transform Pipelines
# =============================================================

def get_train_transform():
    """Augmentations for training — makes model more robust"""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


def get_val_transform():
    """No augmentations for validation/test — only resize & normalize"""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


# =============================================================
# Data Loading Functions
# =============================================================

def load_ham10000(split="pretrain"):
    """
    Returns a SkinDiseaseDataset for HAM10000.
    HAM10000 is ONLY used in pretraining (20% sample).
    """
    df = pd.read_csv(config.HAM10000_CSV)

    # HAM10000 column for label is 'dx', image id is 'image_id' — standard format
    # Use 20% for pretraining, rest discarded (as per plan)
    df_sample, _ = train_test_split(df, test_size=0.80,
                                    random_state=config.SEED,
                                    stratify=df["dx"])
    transform = get_train_transform() if split == "pretrain" else get_val_transform()
    return SkinDiseaseDataset(df_sample, config.HAM10000_IMG_DIR,
                              HAM_LABEL_MAP, transform)


def load_pad(split="pretrain"):
    """
    Returns train / val / test splits for PAD-UFES-20.
    Split ratios: 70% train, 20% val, 10% test (from config plan).
    """
    df = pd.read_csv(config.PAD_CSV)

    # PAD-UFES-20 uses 'diagnostic' column for labels and 'img_id' for image name
    # Rename to our standard column names
    df = df.rename(columns={"diagnostic": "dx", "img_id": "image_id",
                             "age": "age", "gender": "sex",
                             "region": "localization"})

    train_df, temp_df = train_test_split(df, test_size=0.30,
                                         random_state=config.SEED,
                                         stratify=df["dx"])
    val_df, test_df   = train_test_split(temp_df, test_size=0.33,
                                          random_state=config.SEED,
                                          stratify=temp_df["dx"])
    # test_size=0.33 of 30% ≈ 10% of total

    splits = {"train": train_df, "val": val_df, "test": test_df}
    transform = get_train_transform() if split == "train" else get_val_transform()
    return SkinDiseaseDataset(splits[split], config.PAD_IMG_DIR,
                              PAD_LABEL_MAP, transform)


def load_derm(split="train"):
    """
    Returns train / val / test splits for Dermaco-In (main dataset).
    """
    df = pd.read_csv(config.DERM_CSV)

    train_df, temp_df = train_test_split(df, test_size=0.30,
                                          random_state=config.SEED,
                                          stratify=df["dx"])
    val_df, test_df   = train_test_split(temp_df, test_size=0.33,
                                          random_state=config.SEED,
                                          stratify=temp_df["dx"])

    splits = {"train": train_df, "val": val_df, "test": test_df}
    transform = get_train_transform() if split == "train" else get_val_transform()
    return SkinDiseaseDataset(splits[split], config.DERM_IMG_DIR,
                              DERM_LABEL_MAP, transform)


# =============================================================
# Build DataLoaders for both training stages
# =============================================================

def get_pretrain_loaders():
    """
    Stage 1 — Pretraining
    Combined: HAM10000 (20%) + PAD train (20%) + Derm train (20%)
    """
    ham_ds  = load_ham10000(split="pretrain")
    pad_ds  = load_pad(split="train")        # already 70%, we'll just use it
    derm_ds = load_derm(split="train")

    combined = ConcatDataset([ham_ds, pad_ds, derm_ds])

    loader = DataLoader(combined,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=True)
    print(f"[Pretrain] Total samples: {len(combined)}")
    return loader


def get_finetune_loaders():
    """
    Stage 2 — Fine-tuning
    Train = PAD (70%) + Derm (70%)
    Val   = PAD (20%) + Derm (20%)
    Test  = PAD (10%) + Derm (10%)
    """
    train_ds = ConcatDataset([load_pad("train"), load_derm("train")])
    val_ds   = ConcatDataset([load_pad("val"),   load_derm("val")])
    test_ds  = ConcatDataset([load_pad("test"),  load_derm("test")])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)

    print(f"[Finetune] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
