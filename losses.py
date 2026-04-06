# =============================================================
# losses.py  —  Focal Loss for imbalanced datasets
# =============================================================
#
# WHY Focal Loss?
#   Skin disease datasets are heavily imbalanced.
#   Example: "nevus" might have 5000 samples, "melanoma" only 200.
#   Standard CrossEntropy treats all samples equally → model gets
#   lazy and just predicts the majority class.
#
#   Focal Loss adds a factor (1 - p_t)^gamma that:
#     • Down-weights easy examples (model already confident)
#     • Up-weights hard examples (model uncertain/wrong)
#   This forces the model to focus on rare, hard classes.
#
# Formula:
#   FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
#
#   gamma = 2.0  (standard value from original paper)
#   alpha = per-class weight (optional, for extra balancing)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Parameters
    ----------
    gamma : float  — focusing parameter. Higher = more focus on hard examples.
                     0 = standard CrossEntropy. Default 2.0.
    alpha : list or None  — per-class weights. Length must equal NUM_CLASSES.
                            Set None to treat all classes equally.
    reduction : str  — 'mean' (default) or 'sum'
    """

    def __init__(self, gamma=config.FOCAL_GAMMA,
                 alpha=config.FOCAL_ALPHA,
                 reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # If alpha is provided, register it as a buffer (moves to GPU automatically)
        if alpha is not None:
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        Parameters
        ----------
        logits  : Tensor (batch, num_classes)  — raw model output (before softmax)
        targets : Tensor (batch,)              — ground truth class indices
        """

        # Step 1: Standard cross-entropy loss (per sample, no reduction yet)
        # log_softmax is numerically more stable than log(softmax(x))
        log_probs = F.log_softmax(logits, dim=1)          # (batch, num_classes)
        ce_loss   = F.nll_loss(log_probs, targets,        # (batch,)
                               weight=self.alpha,
                               reduction="none")

        # Step 2: Get the probability of the TRUE class for each sample
        probs    = torch.exp(log_probs)                   # (batch, num_classes)
        # gather picks the probability at the true class index
        p_t = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # (batch,)

        # Step 3: Focal weight  =  (1 - p_t)^gamma
        #   If p_t is high (easy sample) → weight ≈ 0  → loss contribution tiny
        #   If p_t is low  (hard sample) → weight ≈ 1  → loss contribution full
        focal_weight = (1.0 - p_t) ** self.gamma          # (batch,)

        # Step 4: Apply focal weight to CE loss
        focal_loss = focal_weight * ce_loss                # (batch,)

        # Step 5: Reduce
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss   # no reduction — returns per-sample losses


def compute_class_weights(dataset, num_classes=config.NUM_CLASSES):
    """
    Computes inverse-frequency class weights from a dataset.
    Useful to pass as 'alpha' to FocalLoss for extra imbalance handling.

    Returns a list of floats, one per class.
    """
    counts = np.zeros(num_classes)

    for _, _, label in dataset:
        counts[label] += 1

    # Avoid division by zero
    counts = np.where(counts == 0, 1, counts)

    # Inverse frequency: rare classes get higher weight
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes   # normalise so they sum to num_classes

    print("[FocalLoss] Class weights:", {config.DISEASE_CLASSES[i]: round(weights[i], 4)
                                          for i in range(num_classes)})
    return weights.tolist()
