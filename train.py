# =============================================================
# train.py  —  Cross-Dataset Transfer Learning Pipeline
# =============================================================
#
# EXECUTION FLOW:
#
#   python train.py
#
#   ┌──────────────────────────────────────────────────────────────┐
#   │  STEP 0  Bootstrap                                            │
#   │    • Set random seed                                          │
#   │    • Detect device (cuda / cpu)                               │
#   │    • config.load_pad_classes()   → PAD_NUM_CLASSES = 2        │
#   │    • config.load_derm_classes()  → DERM_NUM_CLASSES = N       │
#   │    • make_pad_splits()           → train_df_pad, val_df_pad   │
#   │    • make_derm_splits()          → train/val/test (sealed)    │
#   └─────────────────────────────────┬────────────────────────────┘
#                                     │
#   ┌─────────────────────────────────▼────────────────────────────┐
#   │  STAGE 1  PAD-UFES-20 Binary Pretraining                      │
#   │    • Build SkinDiseaseModel(num_classes=2, freeze=True)       │
#   │    • get_pad_loaders() → train + val with WeightedSampler     │
#   │    • FocalLoss(gamma=2.5) + class weights                     │
#   │    • AdamW + CosineAnnealingLR                                │
#   │    • 90 epochs, no early stopping                             │
#   │    • Saves: pretrained_pad_binary.pth                         │
#   └─────────────────────────────────┬────────────────────────────┘
#                                     │
#                    transfer_weights_cross_dataset()
#                    (backbone + projection + fusion MLP transferred;
#                     head discarded; metadata MLP transferred if
#                     input dims match)
#                                     │
#   ┌─────────────────────────────────▼────────────────────────────┐
#   │  STAGE 2  Dermaco-In Multi-class Fine-tuning                  │
#   │    • Replace head: 128 → DERM_NUM_CLASSES                     │
#   │    • get_derm_loaders() → train + val with WeightedSampler    │
#   │    • FocalLoss(gamma=2.0) + label_smoothing=0.1               │
#   │    • AdamW (differential LR: backbone 0.1×, head 1×)         │
#   │    • Domain-adaptive BN (frozen first 10 epochs)              │
#   │    • Progressive unfreezing (layer4 → ... → layer1)           │
#   │    • CosineAnnealingLR + EarlyStopping(patience=45)           │
#   │    • 130 epochs max                                            │
#   │    • Saves: best_dermaco_model.pth                            │
#   └─────────────────────────────────┬────────────────────────────┘
#                                     │
#              TemperatureScaler.calibrate(derm_val_loader)
#                                     │
#   ┌─────────────────────────────────▼────────────────────────────┐
#   │  STAGE 3  Dermaco-In Test Evaluation (10% held-out only)      │
#   │    • TTA (10 augmented passes)                                │
#   │    • Classification report → results/                         │
#   │    • Confusion matrix      → results/confusion_matrix.png     │
#   │    • ROC curves            → results/roc_curves.png           │
#   │    • Metrics JSON          → results/test_metrics.json        │
#   └──────────────────────────────────────────────────────────────┘

import os
import random
import torch
import numpy as np

import config
from dataset  import (make_pad_splits, make_derm_splits,
                      get_pad_loaders, get_derm_loaders, get_derm_test_loader)
from model    import (SkinDiseaseModel, transfer_weights_cross_dataset,
                      load_stage2_final_weights, TemperatureScaler)
from trainer  import run_stage
from evaluate import (run_full_evaluation, plot_training_history)


# =============================================================
# Reproducibility
# =============================================================

def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic mode — slight performance cost but exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================
# Main
# =============================================================

def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Setup] Device: {device}")
    if device.type == "cuda":
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR,    exist_ok=True)

    # ── STEP 0: Bootstrap class registries ──────────────────
    # Must happen before ANY model construction or dataset loading.
    # load_pad_classes() always produces exactly 2 classes (binary).
    # load_derm_classes() reads Dermaco-In CSV to count unique dx values.
    config.load_pad_classes()
    config.load_derm_classes()

    # ── STEP 1: Create splits ────────────────────────────────
    # PAD splits: train 80% / val 20% (no test — binary pretraining only)
    # Derm splits: train 70% / val 20% / test 10% (test sealed immediately)
    pad_train_df,  pad_val_df                        = make_pad_splits()
    derm_train_df, derm_val_df, derm_test_df         = make_derm_splits()
    # derm_test_df is now sealed — never used until Stage 3.

    # ──────────────────────────────────────────────────────────
    # STAGE 1 — PAD-UFES-20 Binary Pretraining
    # ──────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  STAGE 1 — PAD-UFES-20 Binary Pretraining")
    print("═"*65)

    pad_train_loader, pad_val_loader, pad_class_wts = get_pad_loaders(
        pad_train_df, pad_val_df
    )

    # Build Stage 1 model: binary head, frozen backbone, PAD metadata dim
    model = SkinDiseaseModel(
        num_classes     = config.PAD_NUM_CLASSES,   # 2
        meta_input_dim  = config.META_INPUT_DIM_PAD,  # 4
        freeze_backbone = True,
    ).to(device)

    s1_history, s1_ckpt = run_stage(
        model        = model,
        train_loader = pad_train_loader,
        val_loader   = pad_val_loader,
        stage        = "stage1",
        device       = device,
        class_weights= pad_class_wts,
        label_map    = config.PAD_CLASS_TO_IDX,
        dataset_name = "PAD-UFES-20",
    )
    plot_training_history(s1_history, stage_name="Stage1_PAD_Binary")

    # ──────────────────────────────────────────────────────────
    # STAGE 2 — Dermaco-In Multi-class Fine-tuning
    # ──────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  STAGE 2 — Dermaco-In Multi-class Fine-tuning")
    print("═"*65)

    # 2a. Build Stage 2 model architecture (still binary head, correct meta dim)
    #     We build with num_classes=2 to match Stage 1 weight shapes exactly,
    #     then replace the head. This avoids any shape mismatch during transfer.
    stage2_model = SkinDiseaseModel(
        num_classes     = config.PAD_NUM_CLASSES,     # 2 temporarily
        meta_input_dim  = config.META_INPUT_DIM_DERM, # may differ from PAD
        freeze_backbone = True,                        # transfer starts frozen
    ).to(device)

    # 2b. Transfer weights from Stage 1 checkpoint
    #     transfer_weights_cross_dataset() will:
    #       - transfer backbone + projection (always)
    #       - transfer fusion MLP (always — shape matches regardless of classes)
    #       - transfer metadata MLP only if META_INPUT_DIM_PAD == META_INPUT_DIM_DERM
    #       - SKIP the head (shape 2 in src vs 2 in dst — same here, but we
    #         replace it anyway for correctness and to apply Xavier init)
    transfer_report = transfer_weights_cross_dataset(
        stage2_model   = stage2_model,
        ckpt_path      = s1_ckpt,
        device         = device,
        transfer_meta_mlp = config.TRANSFER_META_MLP,
    )

    # 2c. Replace binary head with N-class Dermaco head
    stage2_model.replace_classifier_head(config.DERM_NUM_CLASSES)

    # 2d. Initialize Stage 2 unfreezing tracker
    stage2_model._unfrozen_groups = config.UNFREEZE_START_GROUPS
    stage2_model.unfreeze_last_n_groups(config.UNFREEZE_START_GROUPS)

    # 2e. Build Dermaco DataLoaders
    derm_train_loader, derm_val_loader, derm_class_wts = get_derm_loaders(
        derm_train_df, derm_val_df
    )

    s2_history, s2_ckpt = run_stage(
        model        = stage2_model,
        train_loader = derm_train_loader,
        val_loader   = derm_val_loader,
        stage        = "stage2",
        device       = device,
        class_weights= derm_class_wts,
        label_map    = config.DERM_CLASS_TO_IDX,
        dataset_name = "Dermaco-In",
    )
    plot_training_history(s2_history, stage_name="Stage2_Dermaco_Multiclass")

    # ──────────────────────────────────────────────────────────
    # STAGE 3 — Test Evaluation
    # ──────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  STAGE 3 — Dermaco-In Test Evaluation (held-out 10%)")
    print("═"*65)

    # 3a. Reload best Stage 2 checkpoint (strict)
    load_stage2_final_weights(stage2_model, s2_ckpt, device)

    # 3b. Temperature calibration on validation set (NOT test set)
    print("\n[Calibration] Fitting temperature scaling on Dermaco-In val set...")
    scaled_model = TemperatureScaler(stage2_model).to(device)
    scaled_model.calibrate(derm_val_loader, device)

    # 3c. Test DataLoader (deterministic val transform, no augmentation)
    derm_test_loader = get_derm_test_loader(derm_test_df)

    # 3d. Full evaluation with TTA on calibrated model
    metrics, all_preds, all_labels, all_probs = run_full_evaluation(
        model        = scaled_model,
        test_loader  = derm_test_loader,
        test_df      = derm_test_df,   # for TTA rebuild
        device       = device,
        use_tta      = True,
        n_tta_passes = 10,
    )

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  PIPELINE COMPLETE")
    print("═"*65)
    print(f"  Stage 1 checkpoint : {s1_ckpt}")
    print(f"  Stage 2 checkpoint : {s2_ckpt}")
    print(f"  Results            : {config.RESULTS_DIR}/")
    print(f"  Dermaco Test F1-Macro : {metrics['f1_macro']}")
    print(f"  Dermaco Test AUC-Macro: {metrics['auc_macro']}")
    print(f"  Dermaco Test Accuracy : {metrics['accuracy']}%")

    # Warn if transfer efficiency was low
    if transfer_report.get("mismatched"):
        print(f"\n  ⚠  {len(transfer_report['mismatched'])} weight tensor(s) "
              f"had shape mismatches and were re-initialized.")
        print(f"     Check config.TRANSFER_META_MLP and metadata dim settings.")


if __name__ == "__main__":
    main()
