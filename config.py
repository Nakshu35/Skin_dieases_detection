# =============================================================
# config.py  —  All settings live here. Change values here only.
# =============================================================

import os

# ── Paths ─────────────────────────────────────────────────────
DATA_ROOT        = "data"                          # root folder that holds all datasets
HAM10000_IMG_DIR = os.path.join(DATA_ROOT, "HAM10000", "images")
HAM10000_CSV     = os.path.join(DATA_ROOT, "HAM10000", "HAM10000_metadata.csv")

PAD_IMG_DIR      = os.path.join(DATA_ROOT, "PAD-UFES-20", "images")
PAD_CSV          = os.path.join(DATA_ROOT, "PAD-UFES-20", "metadata.csv")

DERM_IMG_DIR     = os.path.join(DATA_ROOT, "Dermaco-In", "images")
DERM_CSV         = os.path.join(DATA_ROOT, "Dermaco-In", "metadata.csv")

CHECKPOINT_DIR   = "checkpoints"                  # saved model weights go here
RESULTS_DIR      = "results"                      # plots, metrics go here

# ── Disease classes (unified across all 3 datasets) ──────────
DISEASE_CLASSES = [
    "melanoma",       # 0
    "basal_cell_carcinoma",  # 1
    "squamous_cell_carcinoma",  # 2
    "actinic_keratosis",     # 3
    "dermatofibroma",        # 4
    "nevus",                 # 5
    "vascular_lesion",       # 6
]
NUM_CLASSES = len(DISEASE_CLASSES)

# ── Severity weights (0.0 = benign, 1.0 = most dangerous) ────
#    You can adjust these based on clinical knowledge
SEVERITY_WEIGHTS = {
    "melanoma":                  1.0,
    "basal_cell_carcinoma":      0.9,
    "squamous_cell_carcinoma":   0.8,
    "actinic_keratosis":         0.6,
    "dermatofibroma":            0.2,
    "nevus":                     0.1,
    "vascular_lesion":           0.3,
}

# ── Metadata feature columns ─────────────────────────────────
#    These are the columns we expect in every CSV after preprocessing
META_COLS = ["age_norm", "sex_enc", "location_enc", "skin_type_enc"]
META_INPUT_DIM = len(META_COLS)    # 4 features going into MLP

# ── Image settings ────────────────────────────────────────────
IMG_SIZE    = 224          # ResNet50 expects 224×224
MEAN        = [0.485, 0.456, 0.406]   # ImageNet mean (used because we use pretrained weights)
STD         = [0.229, 0.224, 0.225]   # ImageNet std

# ── Training hyperparameters ─────────────────────────────────
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4            # L2 regularisation

PRETRAIN_EPOCHS = 15              # Stage 1 — general feature learning
FINETUNE_EPOCHS = 25              # Stage 2 — task-specific learning

# ── Focal Loss hyperparameters ───────────────────────────────
FOCAL_GAMMA  = 2.0    # focuses on hard examples (2.0 is standard)
FOCAL_ALPHA  = None   # set to a list of per-class weights if you want, else None

# ── Misc ─────────────────────────────────────────────────────
SEED         = 42
DEVICE       = "cuda"     # change to "cpu" if no GPU
NUM_WORKERS  = 4          # dataloader parallel workers
