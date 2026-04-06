# Skin Disease Detection Pipeline
### Multimodal Deep Learning — ResNet50 + Metadata MLP + Focal Loss

---

## Project Structure

```
skin_disease_pipeline/
│
├── config.py          ← ALL settings (paths, hyperparameters, class names)
├── dataset.py         ← Dataset loading, preprocessing, DataLoaders
├── model.py           ← Full model architecture
├── losses.py          ← Focal Loss implementation
├── trainer.py         ← Training + validation loops
├── evaluate.py        ← Metrics, confusion matrix, severity scoring
├── gradcam.py         ← Grad-CAM visualization
├── train.py           ← Main script — run this to train
├── predict.py         ← Run on a single image
└── requirements.txt   ← Python dependencies
```

---

## Architecture

```
INPUT
  Image (224×224×3)          Metadata (age, sex, location, skin_type)
       │                                         │
       ▼                                         ▼
  ResNet50 backbone                    MLP (4 → 64 → 128)
  (pretrained ImageNet)
       │                                         │
       ▼                                         ▼
  Projection Layer                        128-dim vector
  (2048 → 512)
       │                                         │
       └─────────────── Concat ─────────────────┘
                            │
                      640-dim vector
                            │
                    Fusion MLP (640 → 256 → NUM_CLASSES)
                            │
                      Softmax Output
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Disease Class              Severity Score
        (e.g. Melanoma)          (probability × weight)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Organize your data
```
data/
├── HAM10000/
│   ├── images/          ← all .jpg images
│   └── HAM10000_metadata.csv
│
├── PAD-UFES-20/
│   ├── images/
│   └── metadata.csv
│
└── Dermaco-In/
    ├── images/
    └── metadata.csv
```

### 3. CSV format expected

Each CSV needs these columns (rename yours to match):

| Column        | Description                        |
|---------------|------------------------------------|
| `image_id`    | filename without extension         |
| `dx`          | disease label string               |
| `age`         | patient age (number)               |
| `sex`         | "male" / "female"                  |
| `localization`| body location string               |
| `skin_type`   | Fitzpatrick type 1-6 (optional)    |

---

## Training

```bash
# Full pipeline: pretrain → finetune → evaluate → grad-cam
python train.py
```

Training stages:
- **Stage 1 (Pretraining):** 15 epochs, frozen ResNet, learns general skin features
- **Stage 2 (Fine-tuning):** 25 epochs, unfrozen ResNet, task-specific learning

Outputs saved to:
- `checkpoints/best_pretrain.pth`
- `checkpoints/best_finetune.pth`
- `results/Pretraining_history.png`
- `results/Fine-tuning_history.png`
- `results/confusion_matrix.png`
- `results/gradcam/`

---

## Predict on a single image

```bash
python predict.py \
  --image   path/to/lesion.jpg \
  --age     45 \
  --sex     male \
  --location back \
  --skin_type 2
```

Example output:
```
═════════════════════════════════════════════
   SKIN DISEASE PREDICTION REPORT
═════════════════════════════════════════════
  Predicted Disease : Basal Cell Carcinoma
  Confidence        : 87.3%
  Severity Score    : 0.784 / 1.000

  All Probabilities:
    melanoma                       0.031  █
    basal_cell_carcinoma           0.873  ██████████████████████████
    squamous_cell_carcinoma        0.042  █
    actinic_keratosis              0.021
    dermatofibroma                 0.015
    nevus                          0.012
    vascular_lesion                0.006

  Recommendation: Urgent dermatology referral recommended.
═════════════════════════════════════════════
```

---

## Key Design Choices Explained

### Why Focal Loss?
Skin disease datasets are imbalanced — common conditions (nevus) have 10-50× more samples than rare cancers (melanoma). Standard cross-entropy learns to just predict the majority class.

Focal Loss adds a `(1-p)^γ` factor:
- Easy samples (model already confident) → near-zero gradient contribution
- Hard samples (rare diseases, uncertain predictions) → full gradient contribution

This forces the model to actually learn rare diseases.

### Why two-stage training?
- **Stage 1 (frozen backbone):** Prevents destroying pretrained ImageNet features early on. Only the new layers learn.
- **Stage 2 (unfrozen backbone):** Fine-tunes all layers end-to-end for skin disease specifics.

### Why metadata?
Images alone miss clinical context. A dark lesion on a 70-year-old's back has very different risk than the same lesion on a 20-year-old's arm. Metadata captures this.

### Why Grad-CAM?
Medical AI without explainability is not clinically acceptable. Grad-CAM shows which skin region the model used for its decision — letting a doctor verify or override it.

---

## Evaluation Metrics

| Metric    | Why it matters                                      |
|-----------|-----------------------------------------------------|
| Accuracy  | Overall % correct                                   |
| F1-macro  | Balanced across classes — important with imbalance  |
| Precision | How many predicted positives are actually positive  |
| Recall    | How many actual diseases we correctly detected      |
| AUC-ROC   | Standard metric in clinical ML papers               |

---

## Files Quick Reference

| File         | What to change                              |
|--------------|---------------------------------------------|
| `config.py`  | paths, class names, hyperparameters          |
| `dataset.py` | `LABEL_MAP` dicts if your CSV labels differ  |
| `model.py`   | architecture dims if you want to experiment  |
| `losses.py`  | gamma value for focal loss tuning            |
| `train.py`   | which stages to run                          |
