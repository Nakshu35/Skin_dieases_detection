# =============================================================
# evaluate.py  —  Metrics, confusion matrix, severity scoring
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)

import config
from losses import FocalLoss


# =============================================================
# 1. Compute all metrics
# =============================================================

def compute_metrics(all_preds, all_labels, num_classes=config.NUM_CLASSES):
    """
    Computes all evaluation metrics from predictions and true labels.

    Parameters
    ----------
    all_preds  : numpy array (N,)  — predicted class indices
    all_labels : numpy array (N,)  — true class indices

    Returns
    -------
    dict of metric name → value
    """
    accuracy  = accuracy_score(all_labels, all_preds)

    # 'macro' = each class equally weighted → punishes bad performance on small classes
    f1        = f1_score(all_labels, all_preds, average="macro",
                         zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro",
                                zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="macro",
                             zero_division=0)

    # AUC needs one-hot labels and class probabilities
    # Here we use a simple approximation with predictions (not probabilities)
    # For proper AUC you'd need model.predict_proba — see evaluate_model() below
    metrics = {
        "accuracy":  round(accuracy * 100, 2),
        "f1_macro":  round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
    }

    return metrics


def print_metrics(metrics, split_name="Test"):
    print(f"\n{'─'*40}")
    print(f"  {split_name} Results")
    print(f"{'─'*40}")
    for key, val in metrics.items():
        print(f"  {key:<15} : {val}")
    print(f"{'─'*40}\n")


# =============================================================
# 2. Confusion Matrix plot
# =============================================================

def plot_confusion_matrix(all_preds, all_labels, save_path=None):
    """Plots and optionally saves confusion matrix heatmap"""
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True, fmt="d",
                xticklabels=config.DISEASE_CLASSES,
                yticklabels=config.DISEASE_CLASSES,
                cmap="Blues")
    plt.title("Confusion Matrix", fontsize=14)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  [Plot] Confusion matrix saved → {save_path}")
    plt.show()


# =============================================================
# 3. Training history plot
# =============================================================

def plot_training_history(history, stage_name, save_dir=config.RESULTS_DIR):
    """Plots loss and accuracy curves over epochs"""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="s")
    ax1.set_title(f"{stage_name} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Focal Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   marker="s")
    ax2.set_title(f"{stage_name} — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{stage_name}_history.png")
    plt.savefig(path, dpi=150)
    print(f"  [Plot] Training history saved → {path}")
    plt.show()


# =============================================================
# 4. Full evaluation on test set
# =============================================================

def evaluate_model(model, test_loader, device):
    """
    Runs the model on the test set, collects probabilities,
    computes all metrics including AUC.

    Returns
    -------
    metrics    : dict
    all_preds  : numpy array
    all_labels : numpy array
    all_probs  : numpy array (N, NUM_CLASSES) — softmax probabilities
    """
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    criterion  = FocalLoss()

    with torch.no_grad():
        for images, metadata, labels in test_loader:
            images   = images.to(device)
            metadata = metadata.to(device)
            labels   = labels.to(device)

            logits = model(images, metadata)
            probs  = F.softmax(logits, dim=1)   # convert logits → probabilities
            preds  = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Standard metrics
    metrics = compute_metrics(all_preds, all_labels)

    # AUC — needs probabilities and one-hot labels
    try:
        auc = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr",     # one-vs-rest
            average="macro"
        )
        metrics["auc"] = round(auc, 4)
    except ValueError as e:
        print(f"  [AUC] Skipped: {e}")

    # Per-class detailed report
    report = classification_report(
        all_labels, all_preds,
        target_names=config.DISEASE_CLASSES,
        zero_division=0
    )
    print("\n  Classification Report:\n")
    print(report)

    return metrics, all_preds, all_labels, all_probs


# =============================================================
# 5. Severity Scoring
# =============================================================

def compute_severity(probs, disease_names=config.DISEASE_CLASSES):
    """
    Computes severity score from predicted probabilities.

    Parameters
    ----------
    probs : numpy array (NUM_CLASSES,) or (N, NUM_CLASSES)
            — softmax probabilities for one or multiple samples

    Returns
    -------
    severity_score : float or numpy array
        Weighted sum of (probability × severity_weight) for each class.
        Range: 0.0 (benign) to 1.0 (maximum danger).
    predicted_class: str  — most likely disease name
    recommendation : str  — clinical recommendation string
    """

    # Build severity weight vector in same order as DISEASE_CLASSES
    weight_vector = np.array([
        config.SEVERITY_WEIGHTS[name] for name in disease_names
    ])

    single = probs.ndim == 1
    if single:
        probs = probs[np.newaxis, :]   # make it (1, NUM_CLASSES) for uniform handling

    severity_scores = (probs * weight_vector).sum(axis=1)   # (N,)
    pred_indices    = probs.argmax(axis=1)
    pred_names      = [disease_names[i] for i in pred_indices]

    if single:
        return float(severity_scores[0]), pred_names[0], _recommendation(severity_scores[0])
    else:
        return severity_scores, pred_names


def _recommendation(severity_score):
    """Returns a clinical recommendation based on severity score"""
    if severity_score < 0.2:
        return "Monitor regularly. No immediate action required."
    elif severity_score < 0.4:
        return "Schedule a dermatology appointment within 3 months."
    elif severity_score < 0.6:
        return "Dermatologist consultation recommended within 4 weeks."
    elif severity_score < 0.8:
        return "Urgent dermatology referral recommended."
    else:
        return "URGENT: Immediate specialist evaluation required."


def predict_single(model, image_tensor, metadata_tensor, device):
    """
    Makes a prediction for ONE sample and prints a full clinical summary.

    Parameters
    ----------
    image_tensor    : torch.Tensor (1, 3, 224, 224)  — preprocessed image
    metadata_tensor : torch.Tensor (1, 4)            — preprocessed metadata
    """
    model.eval()
    with torch.no_grad():
        image_tensor    = image_tensor.to(device)
        metadata_tensor = metadata_tensor.to(device)
        logits = model(image_tensor, metadata_tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    severity, pred_class, recommendation = compute_severity(probs)

    print("\n" + "═"*45)
    print("   SKIN DISEASE PREDICTION REPORT")
    print("═"*45)
    print(f"  Predicted Disease : {pred_class.replace('_', ' ').title()}")
    print(f"  Confidence        : {probs.max()*100:.1f}%")
    print(f"  Severity Score    : {severity:.3f} / 1.000")
    print(f"\n  All Probabilities:")
    for name, prob in zip(config.DISEASE_CLASSES, probs):
        bar = "█" * int(prob * 30)
        print(f"    {name:<30} {prob:.3f}  {bar}")
    print(f"\n  Recommendation: {recommendation}")
    print("═"*45 + "\n")

    return pred_class, float(probs.max()), severity, recommendation
