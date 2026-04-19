# =============================================================
# evaluate.py  —  Stage 3: Dermaco-In Test Evaluation
# =============================================================
#
# KEY DESIGN DECISIONS vs. PREVIOUS VERSION:
#
#   1. SINGLE DATASET FOCUS
#      Stage 3 evaluates ONLY on the Dermaco-In held-out test set.
#      There is no PAD test evaluation — Stage 1 (PAD binary) is a
#      pretraining step, not an evaluation target.
#
#   2. TTA REBUILDS DermacoDataset (not PADDataset)
#      TTA is applied with the Dermaco-In training augmentation pipeline
#      (standard strength, not strong). Each TTA pass independently
#      samples a new random augmentation for every image.
#
#   3. ALL OUTPUTS SAVED TO config.RESULTS_DIR
#      Outputs:
#        results/confusion_matrix.png
#        results/roc_curves.png
#        results/classification_report.txt
#        results/test_metrics.json
#        results/stage1_pad_history.png   (from train.py)
#        results/stage2_derm_history.png  (from train.py)
#
#   4. CALIBRATED MODEL IS EVALUATED
#      run_full_evaluation() accepts the TemperatureScaler-wrapped model
#      so confidence estimates are calibrated before metric computation.

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc as sk_auc,
)
from sklearn.preprocessing import label_binarize

import config
from dataset import DermacoDataset, get_train_transform, get_val_transform


# =============================================================
# 1. Standard inference
# =============================================================

@torch.no_grad()
def run_inference(model, loader, device) -> tuple:
    """
    Returns (all_preds, all_labels, all_probs) as numpy arrays.
    Works with both raw SkinDiseaseModel and TemperatureScaler wrapper.
    """
    model.eval()
    preds_list, labels_list, probs_list = [], [], []

    for imgs, meta, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        logits = model(imgs, meta)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        preds_list.extend(preds.cpu().numpy())
        labels_list.extend(lbls.numpy())
        probs_list.extend(probs.cpu().numpy())

    return (np.array(preds_list),
            np.array(labels_list),
            np.array(probs_list))


# =============================================================
# 2. Test Time Augmentation
# =============================================================

def run_tta(model, test_df: pd.DataFrame, device,
            n_passes: int = 10) -> tuple:
    """
    TTA on Dermaco-In test set.

    For each of `n_passes` augmentation passes:
        - Rebuild DermacoDataset with get_train_transform(dataset='derm')
        - Run inference
        - Accumulate softmax probabilities

    Final prediction = argmax of averaged probabilities.

    n_passes=10 is a good default; diminishing returns above 15.
    """
    model.eval()
    cumulative_probs = None
    ground_truth     = None

    for t in range(n_passes):
        ds = DermacoDataset(test_df, get_train_transform(dataset="derm"))
        loader = DataLoader(ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True)
        _, lbls, probs = run_inference(model, loader, device)

        if cumulative_probs is None:
            cumulative_probs = probs
            ground_truth     = lbls
        else:
            cumulative_probs += probs

    avg_probs = cumulative_probs / n_passes
    all_preds = avg_probs.argmax(axis=1)
    print(f"[TTA] {n_passes} passes averaged over {len(test_df)} samples.")
    return all_preds, ground_truth, avg_probs


# =============================================================
# 3. Metrics
# =============================================================

def compute_metrics(all_preds, all_labels, all_probs,
                    num_classes: int) -> dict:
    acc          = accuracy_score(all_labels, all_preds)
    f1_macro     = f1_score(all_labels, all_preds, average="macro",
                             zero_division=0)
    f1_weighted  = f1_score(all_labels, all_preds, average="weighted",
                             zero_division=0)
    precision    = precision_score(all_labels, all_preds, average="macro",
                                   zero_division=0)
    recall       = recall_score(all_labels, all_preds, average="macro",
                                 zero_division=0)
    try:
        auc_macro = roc_auc_score(all_labels, all_probs,
                                   multi_class="ovr", average="macro")
    except ValueError as e:
        print(f"  [AUC] Skipped: {e}")
        auc_macro = -1.0

    return {
        "accuracy":    round(float(acc)         * 100, 2),
        "f1_macro":    round(float(f1_macro),    4),
        "f1_weighted": round(float(f1_weighted), 4),
        "precision":   round(float(precision),   4),
        "recall":      round(float(recall),      4),
        "auc_macro":   round(float(auc_macro),   4),
    }


def print_metrics(metrics: dict, title: str = "Dermaco-In Test"):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        unit = "%" if k == "accuracy" else ""
        print(f"  {k:<18} : {v}{unit}")
    print(f"{'─'*50}\n")


# =============================================================
# 4. Confusion Matrix
# =============================================================

def plot_confusion_matrix(all_preds, all_labels, class_names: list,
                           save_path: str):
    cm     = confusion_matrix(all_labels, all_preds,
                              labels=list(range(len(class_names))))
    # Normalize to % of true class so imbalance doesn't distort visual
    cm_pct = (cm.astype(float)
               / cm.sum(axis=1, keepdims=True).clip(min=1) * 100)

    # Annotations: "count\n(xx.x%)"
    annots = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annots[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    fig_h = max(8, len(class_names) * 1.1)
    fig_w = max(9, len(class_names) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(cm_pct, annot=annots, fmt="", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.4, vmin=0, vmax=100)
    ax.set_title("Confusion Matrix — Dermaco-In Test Set\n"
                 "(cell: count / % of true class)", fontsize=12, pad=12)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Confusion matrix → {save_path}")


# =============================================================
# 5. Per-class ROC curves
# =============================================================

def plot_roc_curves(all_labels, all_probs, class_names: list,
                    save_path: str):
    """
    One ROC curve per class (OvR) plus macro-average.
    Falls back gracefully if a class has only one label in test set.
    """
    n_classes     = len(class_names)
    labels_bin    = label_binarize(all_labels, classes=list(range(n_classes)))
    interp_fpr    = np.linspace(0, 1, 300)
    mean_tpr      = np.zeros(300)
    n_valid        = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.tab20(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if i >= labels_bin.shape[1]:
            continue
        if len(np.unique(labels_bin[:, i])) < 2:
            print(f"  [ROC] Skipped {name!r}: only one label in test set.")
            continue
        try:
            fpr, tpr, _  = roc_curve(labels_bin[:, i], all_probs[:, i])
            roc_auc      = sk_auc(fpr, tpr)
            interp_tpr   = np.interp(interp_fpr, fpr, tpr)
            interp_tpr[0]= 0.0
            mean_tpr    += interp_tpr
            n_valid     += 1
            ax.plot(fpr, tpr, color=color, lw=1.5, alpha=0.85,
                    label=f"{name} (AUC={roc_auc:.3f})")
        except Exception as e:
            print(f"  [ROC] {name}: {e}")

    if n_valid > 0:
        mean_tpr  /= n_valid
        mean_tpr[-1] = 1.0
        macro_auc = sk_auc(interp_fpr, mean_tpr)
        ax.plot(interp_fpr, mean_tpr, "k--", lw=2.5,
                label=f"Macro avg (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-Class ROC Curves — Dermaco-In Test Set\n"
                 "(One-vs-Rest)", fontsize=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] ROC curves → {save_path}")


# =============================================================
# 6. Training history plots
# =============================================================

def plot_training_history(history: dict, stage_name: str,
                           save_dir: str = config.RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    has_auc = "val_auc" in history and any(v >= 0 for v in history["val_auc"])
    ncols   = 4 if has_auc else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    def _plot(ax, tr, va, title, ylabel):
        ax.plot(epochs, tr, label="Train", marker="o", markersize=2)
        ax.plot(epochs, va, label="Val",   marker="s", markersize=2)
        ax.set_title(f"{stage_name} — {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot(axes[0], history["train_loss"], history["val_loss"],   "Loss",     "Focal Loss")
    _plot(axes[1], history["train_acc"],  history["val_acc"],    "Accuracy", "Acc (%)")
    _plot(axes[2], history["train_f1"],   history["val_f1"],     "F1-Macro", "F1")
    if has_auc:
        axes[3].plot(epochs, history["val_auc"], label="Val AUC",
                     marker="s", markersize=2, color="orange")
        axes[3].set_title(f"{stage_name} — AUC")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("AUC")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir,
                         f"{stage_name.lower().replace(' ', '_')}_history.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [Plot] History → {fname}")


# =============================================================
# 7. Full Stage 3 evaluation entry point
# =============================================================

def run_full_evaluation(model, test_loader, test_df: pd.DataFrame,
                        device, use_tta: bool = True,
                        n_tta_passes: int = 10) -> tuple:
    """
    Complete Stage 3 evaluation on Dermaco-In held-out test set.

    Steps:
        1. TTA inference (or standard if use_tta=False)
        2. Compute + print all metrics
        3. Save classification report → results/classification_report.txt
        4. Save confusion matrix      → results/confusion_matrix.png
        5. Save ROC curves            → results/roc_curves.png
        6. Save metrics JSON          → results/test_metrics.json

    Parameters
    ----------
    model       : TemperatureScaler or SkinDiseaseModel (eval mode)
    test_loader : DataLoader built on test_df with val transform
    test_df     : pd.DataFrame of the held-out test rows (for TTA rebuild)
    device      : torch.device
    use_tta     : if True, run TTA inference
    n_tta_passes: number of TTA passes

    Returns
    -------
    metrics, all_preds, all_labels, all_probs
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    class_names = config.DERM_DISEASE_CLASSES
    num_classes = config.DERM_NUM_CLASSES

    print(f"\n{'='*65}")
    print(f"  STAGE 3 — Dermaco-In Test Evaluation")
    print(f"  Classes: {class_names}")
    print(f"  TTA: {'Yes (' + str(n_tta_passes) + ' passes)' if use_tta else 'No'}")
    print(f"{'='*65}")

    # ── Inference ─────────────────────────────────────────────
    if use_tta:
        all_preds, all_labels, all_probs = run_tta(
            model, test_df, device, n_passes=n_tta_passes
        )
    else:
        all_preds, all_labels, all_probs = run_inference(
            model, test_loader, device
        )

    # ── Metrics ───────────────────────────────────────────────
    metrics = compute_metrics(all_preds, all_labels, all_probs, num_classes)
    print_metrics(metrics)

    # ── Classification report ─────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names = class_names,
        labels       = list(range(num_classes)),
        zero_division = 0,
    )
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Dermaco-In Test Set — Classification Report\n")
        f.write("Transfer learning: PAD-UFES-20 (binary) → Dermaco-In (multi-class)\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n\nSummary Metrics\n" + "-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"  [Report] → {report_path}")
    print(f"\n{report}")

    # ── Confusion matrix ──────────────────────────────────────
    plot_confusion_matrix(
        all_preds, all_labels, class_names,
        save_path=os.path.join(config.RESULTS_DIR, "confusion_matrix.png"),
    )

    # ── ROC curves ────────────────────────────────────────────
    plot_roc_curves(
        all_labels, all_probs, class_names,
        save_path=os.path.join(config.RESULTS_DIR, "roc_curves.png"),
    )

    # ── JSON metrics ──────────────────────────────────────────
    metrics_path = os.path.join(config.RESULTS_DIR, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"dermaco_test": metrics,
                   "class_names": class_names,
                   "n_test_samples": len(all_labels),
                   "tta_passes": n_tta_passes if use_tta else 0}, f, indent=2)
    print(f"  [Metrics] → {metrics_path}")

    return metrics, all_preds, all_labels, all_probs
