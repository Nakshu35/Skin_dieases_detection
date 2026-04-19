# =============================================================
# losses.py  —  Focal Loss with label smoothing
# =============================================================
#
# Identical in structure to the previous version but:
#   1. gamma is now a constructor argument (not pulled from config)
#      so Stage 1 (PAD_FOCAL_GAMMA=2.5) and Stage 2 (DERM_FOCAL_GAMMA=2.0)
#      can use different focus intensities without a global change.
#   2. label_smoothing applies only in Stage 2 (passed as 0.0 for Stage 1).
#   3. num_classes is required when label_smoothing > 0.

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional label smoothing.

    Parameters
    ----------
    gamma           : focusing parameter. Higher → more weight on hard examples.
                      Stage 1 (PAD): use 2.5 for stronger imbalance focus.
                      Stage 2 (Dermaco): use 2.0 (label smoothing compensates).
    alpha           : per-class weight tensor (CPU) | None
    label_smoothing : 0.0 = hard labels. 0.1 = standard smoothing.
    num_classes     : required when label_smoothing > 0.
    reduction       : 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha=None,
                 label_smoothing: float = 0.0,
                 num_classes: int = 2, reduction: str = "mean"):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.num_classes     = num_classes
        self.reduction       = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        log_probs = F.log_softmax(logits, dim=1)   # (B, C) numerically stable

        if self.label_smoothing > 0.0:
            C          = self.num_classes
            smooth_val = self.label_smoothing / max(C - 1, 1)
            one_hot    = torch.full((logits.size(0), C), smooth_val,
                                    device=logits.device, dtype=logits.dtype)
            one_hot.scatter_(1, targets.unsqueeze(1),
                             1.0 - self.label_smoothing)
            ce_loss = -(one_hot * log_probs).sum(dim=1)   # (B,)
        else:
            ce_loss = F.nll_loss(log_probs, targets,
                                 weight=self.alpha, reduction="none")

        # Focal modulation: down-weight easy examples
        probs        = torch.exp(log_probs)
        p_t          = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss   = focal_weight * ce_loss

        # Apply alpha in smooth path (not handled by nll_loss above)
        if self.label_smoothing > 0.0 and self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
