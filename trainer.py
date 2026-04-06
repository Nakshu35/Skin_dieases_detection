# =============================================================
# trainer.py  —  Training and validation loops
# =============================================================
#
# This file has 3 main parts:
#   1. train_one_epoch()  — one pass through training data
#   2. validate()         — one pass through val data, return metrics
#   3. run_stage()        — runs full pretraining OR fine-tuning stage

import os
import time
import torch
import numpy as np
from tqdm import tqdm    # progress bar

import config
from losses import FocalLoss


# =============================================================
# 1. Train one epoch
# =============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_num):
    """
    Runs one full pass through the training data.

    Returns
    -------
    avg_loss : float  — average loss over all batches
    accuracy : float  — % of correct predictions
    """
    model.train()          # puts model in training mode (enables dropout, batchnorm updates)

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    # tqdm wraps the loader to show a live progress bar
    loop = tqdm(loader, desc=f"  Epoch {epoch_num} [Train]", leave=False)

    for images, metadata, labels in loop:
        # Move data to GPU (or CPU)
        images   = images.to(device)
        metadata = metadata.to(device)
        labels   = labels.to(device)

        # ── Forward pass ──────────────────────────────────────
        optimizer.zero_grad()          # clear gradients from last step
        logits = model(images, metadata)   # (batch, NUM_CLASSES)

        # ── Loss ──────────────────────────────────────────────
        loss = criterion(logits, labels)

        # ── Backward pass ─────────────────────────────────────
        loss.backward()                # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
        optimizer.step()               # update weights

        # ── Track metrics ─────────────────────────────────────
        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

        # Live display in progress bar
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples * 100.0
    return avg_loss, accuracy


# =============================================================
# 2. Validate
# =============================================================

def validate(model, loader, criterion, device, desc="Val"):
    """
    Runs one pass through val/test data.
    No gradient computation — faster and uses less memory.

    Returns
    -------
    avg_loss  : float
    accuracy  : float
    all_preds : numpy array of predicted class indices
    all_labels: numpy array of true class indices
    """
    model.eval()          # puts model in eval mode (disables dropout, fixes batchnorm)

    total_loss    = 0.0
    correct       = 0
    total_samples = 0
    all_preds     = []
    all_labels    = []

    loop = tqdm(loader, desc=f"  [{desc}]", leave=False)

    with torch.no_grad():   # don't build computation graph → saves memory
        for images, metadata, labels in loop:
            images   = images.to(device)
            metadata = metadata.to(device)
            labels   = labels.to(device)

            logits = model(images, metadata)
            loss   = criterion(logits, labels)

            total_loss    += loss.item() * images.size(0)
            preds          = logits.argmax(dim=1)
            correct       += (preds == labels).sum().item()
            total_samples += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples * 100.0
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# =============================================================
# 3. Save and Load checkpoints
# =============================================================

def save_checkpoint(model, optimizer, epoch, val_loss, filename):
    """Saves model weights + optimizer state to disk"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":   val_loss,
    }, filename)
    print(f"  [Checkpoint] Saved → {filename}")


def load_checkpoint(model, optimizer, filename, device):
    """Loads weights back into model (useful to resume training)"""
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  [Checkpoint] Loaded ← {filename}  (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"], checkpoint["val_loss"]


# =============================================================
# 4. Run a full training stage
# =============================================================

def run_stage(model, train_loader, val_loader,
              num_epochs, stage_name, device,
              alpha=None):
    """
    Runs a complete training stage (pretrain OR finetune).

    Parameters
    ----------
    model        : SkinDiseaseModel
    train_loader : DataLoader
    val_loader   : DataLoader (can be None for pretraining)
    num_epochs   : int
    stage_name   : "pretrain" or "finetune"
    device       : torch.device
    alpha        : optional per-class weights for FocalLoss
    """

    criterion = FocalLoss(gamma=config.FOCAL_GAMMA, alpha=alpha)

    # Optimizer — Adam works well for most deep learning tasks
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Learning rate scheduler — reduces LR when val loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR,
                                   f"best_{stage_name}.pth")

    print(f"\n{'='*55}")
    print(f"  Starting Stage: {stage_name.upper()}  ({num_epochs} epochs)")
    print(f"{'='*55}")

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        # ── Train ─────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # ── Validate ──────────────────────────────────────────
        if val_loader is not None:
            val_loss, val_acc, _, _ = validate(
                model, val_loader, criterion, device, desc="Val"
            )
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        else:
            val_loss, val_acc = train_loss, train_acc

        elapsed = time.time() - start

        # ── Log ───────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch [{epoch:>3}/{num_epochs}]  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}%  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%  "
              f"({elapsed:.1f}s)")

    print(f"\n  Best val loss: {best_val_loss:.4f}")
    return history, checkpoint_path
