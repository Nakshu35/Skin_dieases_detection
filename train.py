# =============================================================
# train.py  —  Main script. Run this to train the full pipeline.
# =============================================================
#
# HOW TO RUN:
#   python train.py
#
# What happens:
#   Step 1 → Pretraining  on HAM10000 + PAD + Derm (frozen backbone)
#   Step 2 → Fine-tuning  on PAD + Derm (unfrozen backbone)
#   Step 3 → Test evaluation + metrics
#   Step 4 → Grad-CAM visualizations

import os
import random
import torch
import numpy as np

import config
from dataset  import get_pretrain_loaders, get_finetune_loaders
from model    import SkinDiseaseModel
from trainer  import run_stage, load_checkpoint
from evaluate import evaluate_model, print_metrics, plot_confusion_matrix, \
                     plot_training_history
from gradcam  import run_gradcam_on_samples


# =============================================================
# Reproducibility — same random seed = same results every run
# =============================================================

def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================
# Main
# =============================================================

def main():
    set_seed()

    # ── Device ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Setup] Using device: {device}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR,    exist_ok=True)

    # ── Build Model ─────────────────────────────────────────
    # freeze_backbone=True: only the projection + metadata + classifier layers
    # are trained during pretraining. ResNet stays frozen to preserve ImageNet features.
    model = SkinDiseaseModel(freeze_backbone=True)
    model = model.to(device)
    model.count_parameters()

    # ============================================================
    # STAGE 1 — Pretraining
    # Goal: teach the model general skin lesion features
    # ============================================================
    print("\n[Stage 1] Loading pretrain data...")
    pretrain_loader = get_pretrain_loaders()

    pretrain_history, pretrain_ckpt = run_stage(
        model        = model,
        train_loader = pretrain_loader,
        val_loader   = None,           # no validation in pretraining (common practice)
        num_epochs   = config.PRETRAIN_EPOCHS,
        stage_name   = "pretrain",
        device       = device,
    )

    plot_training_history(pretrain_history, stage_name="Pretraining")

    # ============================================================
    # STAGE 2 — Fine-tuning
    # Goal: specialise the model on PAD + Dermaco-In
    # ============================================================
    print("\n[Stage 2] Unfreezing backbone and loading finetune data...")
    model.unfreeze_backbone()
    model.count_parameters()

    train_loader, val_loader, test_loader = get_finetune_loaders()

    finetune_history, finetune_ckpt = run_stage(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_epochs   = config.FINETUNE_EPOCHS,
        stage_name   = "finetune",
        device       = device,
    )

    plot_training_history(finetune_history, stage_name="Fine-tuning")

    # ============================================================
    # STAGE 3 — Test Evaluation
    # Load the best fine-tuned model checkpoint
    # ============================================================
    print("\n[Stage 3] Loading best checkpoint and evaluating on test set...")
    load_checkpoint(model, optimizer=None, filename=finetune_ckpt, device=device)

    metrics, all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device
    )
    print_metrics(metrics, split_name="Test Set")

    plot_confusion_matrix(
        all_preds, all_labels,
        save_path=os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    )

    # ============================================================
    # STAGE 4 — Grad-CAM Visualizations
    # ============================================================
    print("\n[Stage 4] Generating Grad-CAM explanations...")
    run_gradcam_on_samples(model, test_loader, device, num_samples=5)

    print("\n[Done] Full pipeline complete!")
    print(f"  Checkpoints → {config.CHECKPOINT_DIR}/")
    print(f"  Results     → {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
