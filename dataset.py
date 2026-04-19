# =============================================================
# dataset.py  —  Cross-Dataset Data Loading
# =============================================================
#
# KEY DESIGN DECISIONS vs. PREVIOUS VERSION:
#
#   1. TWO SEPARATE DATASET CLASSES
#      PADDataset   → Stage 1, binary labels (cancer / non-cancer)
#      DermacoDataset → Stage 2, multi-class from Dermaco-In dx column
#      Keeping them separate makes the metadata encoding logic for each
#      dataset explicit and auditable, rather than branching inside one
#      class with a mode flag. Both inherit from a shared _BaseDataset
#      that handles image loading + transform dispatch.
#
#   2. METADATA ENCODING IS DATASET-SPECIFIC
#      PAD-UFES-20 uses 'gender' / 'region' / 'fitspatrick'.
#      Dermaco-In uses 'sex' / 'localization' / 'skin_type'.
#      Each dataset's encode_metadata() reads its own column names from
#      config.*_COL constants, so a schema change is a one-line config edit.
#
#   3. STRATIFIED SPLITS ARE INDEPENDENT PER DATASET
#      PAD is split for Stage 1 (only train + val needed; no test held-out
#      because we evaluate the final model on Dermaco-In test only).
#      Dermaco-In is split 70/20/10 with the 10% test sealed immediately.
#
#   4. DOMAIN SHIFT AUGMENTATION
#      PAD images are clinical macro photos; Dermaco-In may be dermoscopy.
#      The augmentation pipelines are identical in structure but the
#      intensity can be tuned per-dataset via the `aug_strength` argument.
#      Strong augmentation on PAD prevents the model from overfitting PAD's
#      specific photographic style before transferring to Dermaco-In.
#
#   5. CLASS WEIGHTING RETURNS DATASET-SPECIFIC TENSORS
#      get_class_weights() is a standalone function that works for both
#      binary (PAD) and multi-class (Dermaco) by accepting num_classes as
#      a parameter. No global state is modified.

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBU_OK = True
except ImportError:
    _ALBU_OK = False

import config


# =============================================================
# Shared metadata encoding primitives
# =============================================================

def _encode_sex(value) -> float:
    """
    Male → 1.0 | Female → 0.0 | Unknown → -1.0
    Handles both 'MALE'/'FEMALE' strings and 'M'/'F' abbreviations.
    """
    if pd.isna(value):
        return -1.0
    v = str(value).strip().upper()
    return 1.0 if v in ("MALE", "M", "1") else (0.0 if v in ("FEMALE", "F", "0") else -1.0)


def _encode_region(value, vocab: dict) -> float:
    if pd.isna(value):
        return -1.0
    return float(vocab.get(str(value).strip().lower(), -1))


def _normalize_age(value, max_age: float = 100.0) -> float:
    if pd.isna(value):
        return -1.0
    return min(float(value) / max_age, 1.0)


def _encode_skin_type(value) -> float:
    """Fitzpatrick I-VI → 0.0–1.0 | Missing → -1.0"""
    if pd.isna(value):
        return -1.0
    try:
        return (float(int(value)) - 1.0) / 5.0
    except (ValueError, TypeError):
        return -1.0


# Shared region vocabulary (covers both PAD and Dermaco-In region labels)
REGION_VOCAB = {
    # PAD-UFES-20 region values
    "face": 0, "neck": 1, "scalp": 2, "ear": 3,
    "chest": 4, "back": 5, "abdomen": 6, "trunk": 7,
    "arm": 8, "forearm": 9, "hand": 10, "leg": 11,
    "foot": 12, "acral": 13, "genital": 14,
    # Dermaco-In / HAM10000 localization values
    "upper extremity": 8, "lower extremity": 11,
    "oral/genital": 14, "head/neck": 1,
}


# =============================================================
# Augmentation pipelines
# =============================================================

def _build_albu_train(strong: bool = True) -> "A.Compose":
    """
    Albumentations training pipeline.
    strong=True is used for PAD to prevent style-specific overfitting.
    strong=False (milder) for Dermaco-In to avoid destroying dermoscopy features.
    """
    spatial = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=30, border_mode=0, p=0.6),
    ]
    color = [
        A.ColorJitter(brightness=0.25, contrast=0.25,
                      saturation=0.25, hue=0.08, p=0.7),
        A.ToGray(p=0.05),
    ]
    noise_blur = [
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.3 if strong else 0.15),
        A.GaussNoise(var_limit=(10, 50), p=0.2 if strong else 0.1),
    ]
    distortion = [
        A.OneOf([
            A.ElasticTransform(alpha=60, sigma=6, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=1.0),
        ], p=0.3 if strong else 0.1),
    ]
    dropout = [
        A.CoarseDropout(max_holes=8, max_height=24, max_width=24,
                        fill_value=0, p=0.25 if strong else 0.1),
    ]
    return A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        *spatial, *color, *noise_blur, *distortion, *dropout,
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ])


def _build_tv_train() -> transforms.Compose:
    """Torchvision fallback when Albumentations is not installed."""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.25, contrast=0.25,
                               saturation=0.25, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


def _build_val_transform() -> transforms.Compose:
    """Deterministic: resize + normalize only. Used for val AND test."""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


def get_train_transform(dataset: str = "pad") -> object:
    """
    Returns the appropriate augmentation pipeline for a dataset.
    dataset: "pad" → strong augmentation
             "derm" → standard augmentation
    """
    strong = (dataset == "pad")
    if config.USE_ALBUMENTATIONS and _ALBU_OK:
        return _build_albu_train(strong=strong)
    return _build_tv_train()


def get_val_transform() -> transforms.Compose:
    return _build_val_transform()


# =============================================================
# Base Dataset class (shared image loading logic)
# =============================================================

class _BaseDataset(Dataset):
    """
    Internal base class. Handles image loading and transform dispatch.
    Subclasses implement _get_img_path() and _get_label_and_meta().
    """

    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform
        self._use_albu = (config.USE_ALBUMENTATIONS and _ALBU_OK
                          and isinstance(transform, A.Compose))

    def __len__(self) -> int:
        return len(self.df)

    def _apply_transform(self, pil_img: Image.Image) -> torch.Tensor:
        if self._use_albu:
            arr = np.array(pil_img, dtype=np.uint8)
            return self.transform(image=arr)["image"]
        if self.transform:
            return self.transform(pil_img)
        return transforms.ToTensor()(pil_img)

    def _load_image(self, img_path: str) -> Image.Image:
        """Load image, trying common extensions if base path doesn't exist."""
        if os.path.exists(img_path):
            return Image.open(img_path).convert("RGB")
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = img_path + ext
            if os.path.exists(candidate):
                return Image.open(candidate).convert("RGB")
        raise FileNotFoundError(f"Image not found: {img_path}")

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img    = self._load_image(self._get_img_path(row))
        image  = self._apply_transform(img)
        meta, label = self._get_label_and_meta(row)
        return image, meta, label

    def _get_img_path(self, row) -> str:
        raise NotImplementedError

    def _get_label_and_meta(self, row):
        raise NotImplementedError


# =============================================================
# PADDataset  (Stage 1 — binary)
# =============================================================

class PADDataset(_BaseDataset):
    """
    PAD-UFES-20 dataset for Stage 1 binary pretraining.

    Label encoding:
        dx in config.PAD_CANCER_LABELS → 1 (cancer)
        else                           → 0 (non-cancer)

    Metadata: age (normalized), sex (encoded), region (encoded),
              Fitzpatrick skin type (encoded)
    """

    def __init__(self, dataframe: pd.DataFrame, transform):
        super().__init__(dataframe, transform)

    def _get_img_path(self, row) -> str:
        return os.path.join(config.PAD_IMG_DIR,
                            str(row[config.PAD_IMG_ID_COL]))

    def _get_label_and_meta(self, row):
        # Binary label
        dx    = str(row[config.PAD_LABEL_COL]).strip()
        label = 1 if dx in config.PAD_CANCER_LABELS else 0

        # Metadata
        age    = _normalize_age(row.get(config.PAD_AGE_COL, np.nan))
        sex    = _encode_sex(row.get(config.PAD_SEX_COL, np.nan))
        region = _encode_region(row.get(config.PAD_REGION_COL, np.nan),
                                REGION_VOCAB)
        skin   = _encode_skin_type(row.get(config.PAD_SKINTYPE_COL, np.nan))
        meta   = torch.tensor([age, sex, region, skin], dtype=torch.float32)

        return meta, label


# =============================================================
# DermacoDataset  (Stage 2 — multi-class)
# =============================================================

class DermacoDataset(_BaseDataset):
    """
    Dermaco-In dataset for Stage 2 multi-class fine-tuning.

    Label encoding:
        dx → config.DERM_CLASS_TO_IDX[dx]   (built dynamically from CSV)

    Metadata: age, sex, region, skin_type
              (uses DERM_*_COL column names from config)
    """

    def __init__(self, dataframe: pd.DataFrame, transform):
        super().__init__(dataframe, transform)
        # Validate label col exists
        if config.DERM_LABEL_COL not in dataframe.columns:
            raise ValueError(
                f"Column '{config.DERM_LABEL_COL}' not found. "
                f"Available: {list(dataframe.columns)}"
            )

    def _get_img_path(self, row) -> str:
        return os.path.join(config.DERM_IMG_DIR,
                            str(row[config.DERM_IMG_ID_COL]))

    def _get_label_and_meta(self, row):
        dx    = str(row[config.DERM_LABEL_COL]).strip()
        label = config.DERM_CLASS_TO_IDX.get(dx, 0)   # 0 fallback; log if hit

        age    = _normalize_age(row.get(config.DERM_AGE_COL, np.nan))
        sex    = _encode_sex(row.get(config.DERM_SEX_COL, np.nan))
        region = _encode_region(row.get(config.DERM_REGION_COL, np.nan),
                                REGION_VOCAB)
        # skin_type column may not exist in Dermaco-In
        if config.DERM_SKINTYPE_COL and config.DERM_SKINTYPE_COL in row.index:
            skin = _encode_skin_type(row[config.DERM_SKINTYPE_COL])
        else:
            skin = -1.0   # impute as missing

        meta = torch.tensor([age, sex, region, skin], dtype=torch.float32)
        return meta, label


# =============================================================
# Stratified split helpers
# =============================================================

def make_pad_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads PAD-UFES-20 CSV and returns (train_df, val_df).

    NOTE: No test split for PAD. The final model is evaluated on
    Dermaco-In test only — evaluating on PAD test would measure
    Stage 1 binary performance, not the cross-dataset goal.

    Stratified on the BINARY label (cancer/non-cancer) so each split
    has the same cancer prevalence as the full dataset.
    """
    df = pd.read_csv(config.PAD_CSV)
    _check_columns(df, [config.PAD_LABEL_COL, config.PAD_IMG_ID_COL],
                   dataset="PAD-UFES-20")

    # Binary stratification label
    bin_labels = (df[config.PAD_LABEL_COL]
                  .apply(lambda dx: 1 if dx in config.PAD_CANCER_LABELS else 0))

    # 70% train, 30% val (no test — Stage 1 is a pretraining step)
    train_df, val_df = train_test_split(
        df,
        test_size    = 1.0 - config.TRAIN_RATIO - config.TEST_RATIO,
        random_state = config.SEED,
        stratify     = bin_labels,
    )
    print(f"[PAD splits]  Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_derm_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads Dermaco-In CSV and returns (train_df, val_df, test_df).

    test_df is the 10% held-out set — sealed immediately, never used
    during Stage 2 training or any hyperparameter tuning.

    Stratified on the multi-class dx label.
    """
    df = pd.read_csv(config.DERM_CSV)
    _check_columns(df, [config.DERM_LABEL_COL, config.DERM_IMG_ID_COL],
                   dataset="Dermaco-In")

    train_df, temp_df = train_test_split(
        df,
        test_size    = 1.0 - config.TRAIN_RATIO,   # 30%
        random_state = config.SEED,
        stratify     = df[config.DERM_LABEL_COL],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size    = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state = config.SEED,
        stratify     = temp_df[config.DERM_LABEL_COL],
    )
    print(f"[Derm splits] Train: {len(train_df)} | "
          f"Val: {len(val_df)} | Test: {len(test_df)}")
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def _check_columns(df: pd.DataFrame, required: list, dataset: str):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset} CSV missing columns: {missing}. "
            f"Available: {sorted(df.columns.tolist())}"
        )


# =============================================================
# Class weighting utilities (dataset-agnostic)
# =============================================================

def get_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for FocalLoss alpha.
    Normalized so that weights sum to num_classes (balanced scale).
    """
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)   # avoid div-by-zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: list[int],
                          num_classes: int) -> WeightedRandomSampler:
    """
    WeightedRandomSampler for balanced training batches.
    Each sample's draw probability ∝ 1 / class_frequency.

    Prefer this over class-weighted loss alone for severe imbalance:
    sampler ensures balanced mini-batches; weighted loss still focuses
    on hard examples within each batch.
    """
    counts     = np.bincount(labels, minlength=num_classes).astype(float)
    counts     = np.where(counts == 0, 1.0, counts)
    class_wts  = 1.0 / counts
    sample_wts = [class_wts[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights     = torch.tensor(sample_wts, dtype=torch.float64),
        num_samples = len(sample_wts),
        replacement = True,
    )


# =============================================================
# DataLoader factories
# =============================================================

def _extract_labels(df: pd.DataFrame, mode: str) -> list[int]:
    """Extract integer labels from a DataFrame for sampler construction."""
    if mode == "binary":
        return [1 if dx in config.PAD_CANCER_LABELS else 0
                for dx in df[config.PAD_LABEL_COL].astype(str).str.strip()]
    else:
        return [config.DERM_CLASS_TO_IDX.get(str(dx).strip(), 0)
                for dx in df[config.DERM_LABEL_COL].astype(str)]


def get_pad_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Stage 1 DataLoaders for PAD-UFES-20 binary classification.

    Returns
    -------
    train_loader  : DataLoader (WeightedRandomSampler for class balance)
    val_loader    : DataLoader (no shuffle)
    class_weights : torch.Tensor (2,) for FocalLoss alpha
    """
    train_labels = _extract_labels(train_df, "binary")

    train_ds = PADDataset(train_df, get_train_transform(dataset="pad"))
    val_ds   = PADDataset(val_df,   get_val_transform())

    sampler  = make_weighted_sampler(train_labels, num_classes=2)
    wts      = get_class_weights(train_labels, num_classes=2)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    print(f"[PAD Loaders]  Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, wts


def get_derm_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Stage 2 DataLoaders for Dermaco-In multi-class classification.

    Returns
    -------
    train_loader  : DataLoader (WeightedRandomSampler)
    val_loader    : DataLoader
    class_weights : torch.Tensor (DERM_NUM_CLASSES,) for FocalLoss alpha
    """
    train_labels = _extract_labels(train_df, "multiclass")

    train_ds = DermacoDataset(train_df, get_train_transform(dataset="derm"))
    val_ds   = DermacoDataset(val_df,   get_val_transform())

    sampler  = make_weighted_sampler(train_labels, config.DERM_NUM_CLASSES)
    wts      = get_class_weights(train_labels, config.DERM_NUM_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    print(f"[Derm Loaders] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, wts


def get_derm_test_loader(test_df: pd.DataFrame) -> DataLoader:
    """
    Stage 3: test DataLoader for Dermaco-In held-out test set.
    Called ONLY in evaluate.py — never during training.
    """
    test_ds = DermacoDataset(test_df, get_val_transform())
    loader  = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                         shuffle=False, num_workers=config.NUM_WORKERS,
                         pin_memory=True)
    print(f"[Derm Test]    Samples: {len(test_ds)}")
    return loader
