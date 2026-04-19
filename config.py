# =============================================================
# config.py  —  Cross-Dataset Transfer Learning Pipeline
# =============================================================
#
# KEY DESIGN DECISIONS vs. PREVIOUS VERSION:
#
#   1. TWO SEPARATE DATASET NAMESPACES
#      Everything PAD-related is prefixed PAD_*, everything Dermaco-In
#      is prefixed DERM_*. There is zero mixing of their configs so you
#      can tune each dataset independently without risk of collision.
#
#   2. DYNAMIC CLASS LOADING PER DATASET
#      load_pad_classes() populates PAD_* globals (binary: 2 classes).
#      load_derm_classes() populates DERM_* globals (multi-class: N classes).
#      Both must be called in train.py before any model construction.
#
#   3. METADATA DIMENSION MISMATCH HANDLING
#      PAD-UFES-20 has 4 metadata features; Dermaco-In may have a different
#      schema. META_INPUT_DIM_PAD and META_INPUT_DIM_DERM allow the
#      MetadataEncoder to be rebuilt with the correct input dim for each
#      stage rather than hardcoding 4.
#
#   4. CROSS-DATASET WEIGHT TRANSFER SAFETY
#      Because the metadata MLP input dim may differ between datasets,
#      TRANSFER_META_MLP controls whether the metadata encoder weights
#      are transferred (True) or re-initialized (False) in Stage 2.
#      If the two datasets have different metadata schemas, set False.
#
#   5. SEPARATE CHECKPOINT NAMES
#      pretrained_pad_binary.pth   ← Stage 1 output (PAD binary)
#      best_dermaco_model.pth      ← Stage 2 output (Dermaco multi-class)

import os
import pandas as pd


# ── Directory layout ──────────────────────────────────────────
DATA_ROOT      = "data"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"

# ── PAD-UFES-20 paths ─────────────────────────────────────────
PAD_IMG_DIR = os.path.join(DATA_ROOT, "PAD-UFES-20", "images")
PAD_CSV     = os.path.join(DATA_ROOT, "PAD-UFES-20", "metadata.csv")

# PAD-UFES-20 column names (adjust if your CSV differs)
PAD_LABEL_COL    = "dx"
PAD_IMG_ID_COL   = "img_id"
PAD_AGE_COL      = "age"
PAD_SEX_COL      = "gender"
PAD_REGION_COL   = "region"
PAD_SKINTYPE_COL = "fitspatrick"   # Fitzpatrick scale column

# Which PAD dx values are malignant → label=1 in binary task
# PAD-UFES-20 dx values: MEL, BCC, SCC, ACK, NEV, SEK
PAD_CANCER_LABELS = {"MEL", "BCC", "SCC", "ACK"}

# ── Dermaco-In paths ──────────────────────────────────────────
DERM_IMG_DIR = os.path.join(DATA_ROOT, "Dermaco-In", "images")
DERM_CSV     = os.path.join(DATA_ROOT, "Dermaco-In", "metadata.csv")

# Dermaco-In column names (adjust if your CSV differs)
DERM_LABEL_COL    = "dx"
DERM_IMG_ID_COL   = "image_id"    # change to "img_id" if needed
DERM_AGE_COL      = "age"
DERM_SEX_COL      = "sex"
DERM_REGION_COL   = "localization"
DERM_SKINTYPE_COL = "skin_type"   # set None if column absent

# ── PAD dynamic class globals (populated at runtime) ──────────
PAD_DISEASE_CLASSES: list[str] = []    # ["non_cancer", "cancer"] for binary
PAD_NUM_CLASSES:     int       = 0
PAD_CLASS_TO_IDX:    dict      = {}
PAD_IDX_TO_CLASS:    dict      = {}

# ── Dermaco-In dynamic class globals (populated at runtime) ───
DERM_DISEASE_CLASSES: list[str] = []
DERM_NUM_CLASSES:     int       = 0
DERM_CLASS_TO_IDX:    dict      = {}
DERM_IDX_TO_CLASS:    dict      = {}


def load_pad_classes() -> None:
    """
    Populates PAD_* class globals.
    For Stage 1 binary task we always force exactly 2 classes:
        0 → non_cancer
        1 → cancer
    This is intentionally NOT derived from the CSV dx column because
    the binary grouping is defined by PAD_CANCER_LABELS, not by the
    raw label set.
    """
    global PAD_DISEASE_CLASSES, PAD_NUM_CLASSES, PAD_CLASS_TO_IDX, PAD_IDX_TO_CLASS
    PAD_DISEASE_CLASSES = ["non_cancer", "cancer"]
    PAD_NUM_CLASSES     = 2
    PAD_CLASS_TO_IDX    = {"non_cancer": 0, "cancer": 1}
    PAD_IDX_TO_CLASS    = {0: "non_cancer", 1: "cancer"}
    print(f"[Config/PAD]  Binary classes: {PAD_DISEASE_CLASSES}")


def load_derm_classes(csv_path: str = DERM_CSV,
                      label_col: str = DERM_LABEL_COL) -> None:
    """
    Reads unique dx values from Dermaco-In CSV to build the multi-class
    label mapping. Sorted for reproducibility across runs.

    Must be called BEFORE constructing the Stage 2 model.
    """
    global DERM_DISEASE_CLASSES, DERM_NUM_CLASSES, DERM_CLASS_TO_IDX, DERM_IDX_TO_CLASS
    df     = pd.read_csv(csv_path, usecols=[label_col])
    unique = sorted(df[label_col].dropna().astype(str).unique().tolist())
    DERM_DISEASE_CLASSES = unique
    DERM_NUM_CLASSES     = len(unique)
    DERM_CLASS_TO_IDX    = {name: idx for idx, name in enumerate(unique)}
    DERM_IDX_TO_CLASS    = {idx: name for idx, name in enumerate(unique)}
    print(f"[Config/Derm] {DERM_NUM_CLASSES} classes: {DERM_DISEASE_CLASSES}")


# ── Image settings (shared) ───────────────────────────────────
IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]   # ImageNet stats (both datasets use these)
STD      = [0.229, 0.224, 0.225]

# ── Metadata dimensions ───────────────────────────────────────
# PAD-UFES-20: age, sex, region, skin_type  → 4 features
# Dermaco-In:  age, sex, region, skin_type  → 4 features
# If Dermaco-In lacks skin_type set DERM_SKINTYPE_COL = None and
# change META_INPUT_DIM_DERM to 3.
META_INPUT_DIM_PAD  = 4
META_INPUT_DIM_DERM = 4    # adjust if Dermaco-In has different metadata columns

# Whether to transfer the metadata encoder weights from Stage 1 to Stage 2.
# True  → transfer (only valid when META_INPUT_DIM_PAD == META_INPUT_DIM_DERM)
# False → re-initialize for Dermaco-In (safer if schemas differ)
TRANSFER_META_MLP = (META_INPUT_DIM_PAD == META_INPUT_DIM_DERM)

# ── Split ratios (applied per-dataset) ───────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

# ── Stage 1 — PAD Binary Pretraining ─────────────────────────
PAD_PRETRAIN_EPOCHS    = 90         # 80–100 range; cosine scheduler adapts
PAD_PRETRAIN_LR        = 3e-4       # backbone is frozen → higher LR is fine
PAD_PRETRAIN_WD        = 1e-4
PAD_PRETRAIN_SCHEDULER = "cosine"   # "cosine" | "plateau"
PAD_FOCAL_GAMMA        = 2.5        # slightly higher → more focus on hard samples
                                    # (PAD is moderately imbalanced)

# ── Stage 2 — Dermaco-In Multi-class Fine-tuning ─────────────
DERM_FINETUNE_EPOCHS     = 130      # 120–150 range
DERM_FINETUNE_LR         = 3e-5     # small LR; backbone partially unfrozen
DERM_FINETUNE_WD         = 5e-5
DERM_FINETUNE_SCHEDULER  = "cosine"
DERM_EARLY_STOP_PATIENCE = 45       # 40–55 range
DERM_LABEL_SMOOTHING     = 0.1      # helps with Dermaco-In's long tail
DERM_FOCAL_GAMMA         = 2.0

# Progressive unfreezing schedule for Stage 2
# Layer groups: stem(0) → layer1(1) → layer2(2) → layer3(3) → layer4(4)
UNFREEZE_START_GROUPS = 1           # start with only layer4 unfrozen
UNFREEZE_EVERY        = 25          # unfreeze one more group every N epochs
UNFREEZE_MAX_GROUPS   = 4           # stop at layer1 (don't touch stem)

# ── Shared training settings ──────────────────────────────────
BATCH_SIZE   = 32
NUM_WORKERS  = 4
SEED         = 42

# ── Advanced features ─────────────────────────────────────────
USE_AMP             = True    # Automatic Mixed Precision (fp16/fp32)
GRAD_CLIP_NORM      = 1.0     # set None to disable
USE_ALBUMENTATIONS  = True    # Albumentations augmentation pipeline

# ── Focal Loss shared gamma (overridden per-stage above) ──────
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None   # set to tensor to override auto-computed class weights

# ── Checkpoint filenames ──────────────────────────────────────
PAD_PRETRAIN_CKPT  = os.path.join(CHECKPOINT_DIR, "pretrained_pad_binary.pth")
DERM_FINETUNE_CKPT = os.path.join(CHECKPOINT_DIR, "best_dermaco_model.pth")
