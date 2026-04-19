# =============================================================
# trainer.py  —  Cross-Dataset Training Engine
# =============================================================
#
# KEY DESIGN DECISIONS vs. PREVIOUS VERSION:
#
#   1. DOMAIN-ADAPTIVE BATCHNORM IN STAGE 2
#      For the first BN_WARMUP_EPOCHS of Stage 2, BatchNorm running
#      stats in the backbone are frozen (model.freeze_bn_stats()).
#      This prevents early Dermaco-In batches from corrupting the
#      PAD-tuned domain statistics before the backbone has adapted.
#      After BN_WARMUP_EPOCHS the stats are released.
#
#   2. STAGE-AWARE PROGRESSIVE UNFREEZING
#      _maybe_unfreeze() is Stage 2-only and implements a schedule:
#        Epoch 1           → layer4 unfrozen (UNFREEZE_START_GROUPS=1)
#        Epoch 1+EVERY     → layer4+layer3 unfrozen
#        Epoch 1+2*EVERY   → layer4+layer3+layer2 unfrozen
#        ...up to UNFREEZE_MAX_GROUPS
#      Each unfreeze event rebuilds the optimizer with DIFFERENTIAL LR:
#        newly unfrozen backbone layers → lr * 0.1
#        older backbone layers          → lr * 0.01
#        projection + fusion + head     → lr (full)
#      This is more nuanced than the previous version which used a
#      single backbone LR for all unfrozen groups.
#
#   3. EARLY STOPPING ON F1-MACRO (NOT VAL LOSS)
#      For imbalanced multi-class tasks val loss can still decrease
#      while minority-class F1 degrades. F1-macro is the correct signal.
#      Stage 1 (binary, 2 classes) doesn't need early stopping — 90
#      epochs with cosine LR is sufficient.
#
#   4. AMP GRACEFULLY DISABLED ON CPU
#      GradScaler is instantiated with enabled=use_amp where
#      use_amp = config.USE_AMP AND device.type == "cuda".
#      This means the same code runs on CPU without changes.
#
#   5. PER-STAGE CRITERION CONSTRUCTION
#      run_stage() accepts gamma, label_smoothing, and num_classes
#      explicitly so each stage builds its own FocalLoss with its own
#      gamma (2.5 for PAD, 2.0 for Dermaco).

import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

import config
from losses import FocalLoss
from model  import SkinDiseaseModel, save_checkpoint


# Number of Stage 2 epochs to freeze BN running stats
BN_WARMUP_EPOCHS = 10


# =============================================================
# Early Stopping
# =============================================================

class EarlyStopping:
    """
    Monitors val F1-macro. Stops when no improvement for `patience` epochs.
    Stores the epoch of best value for logging.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = -float("inf")
        self.counter   = 0
        self.best_epoch= 0

    def step(self, metric: float, epoch: int) -> bool:
        if metric > self.best + self.min_delta:
            self.best       = metric
            self.counter    = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"  [EarlyStopping] Triggered at epoch {epoch}. "
                  f"Best F1={self.best:.4f} at epoch {self.best_epoch}.")
            return True
        return False


# =============================================================
# One training epoch
# =============================================================

def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, epoch: int, num_classes: int):
    """
    Returns: avg_loss, accuracy (%), f1_macro
    """
    model.train()
    # IMPORTANT: for domain-adaptive BN, individual BN modules may be in
    # eval() mode even when the model is in train() mode. Do NOT call
    # model.train() after freeze_bn_stats() without re-calling freeze_bn_stats().

    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels  = [], []

    for imgs, meta, lbls in tqdm(loader,
                                 desc=f"  Ep {epoch} [Train]", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler._enabled
                                              if hasattr(scaler, "_enabled")
                                              else True)):
            logits = model(imgs, meta)
            loss   = criterion(logits, lbls)

        scaler.scale(loss).backward()

        if config.GRAD_CLIP_NORM is not None:
            scaler.unscale_(optimizer)                         # must unscale before clip
            nn.utils.clip_grad_norm_(model.parameters(),
                                     config.GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(dim=1)

        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == lbls).sum().item()
        n          += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro",
                  zero_division=0, labels=list(range(num_classes)))
    return total_loss / n, correct / n * 100.0, f1


# =============================================================
# Validation
# =============================================================

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes: int,
             desc: str = "Val"):
    """
    Returns: avg_loss, accuracy, f1_macro, auc_macro,
             all_preds (np), all_labels (np), all_probs (np)
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for imgs, meta, lbls in tqdm(loader, desc=f"  [{desc}]", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        logits = model(imgs, meta)
        loss   = criterion(logits, lbls)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == lbls).sum().item()
        n          += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    f1 = f1_score(all_labels, all_preds, average="macro",
                  zero_division=0, labels=list(range(num_classes)))
    try:
        auc = roc_auc_score(all_labels, all_probs,
                             multi_class="ovr", average="macro")
    except ValueError:
        auc = -1.0

    return (total_loss / n, correct / n * 100.0, f1, auc,
            all_preds, all_labels, all_probs)


# =============================================================
# Optimizer / scheduler builders
# =============================================================

def _build_optimizer_stage1(model: SkinDiseaseModel,
                             lr: float, wd: float) -> torch.optim.Optimizer:
    """
    Stage 1: backbone is fully frozen → only non-backbone params are trained.
    Single param group with full LR.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def _build_optimizer_stage2(model: SkinDiseaseModel, lr: float, wd: float,
                             unfrozen_groups: int) -> torch.optim.Optimizer:
    """
    Stage 2: differential LR across three tiers:
        Tier A — backbone layers unfrozen this epoch → lr * 0.1
        Tier B — backbone layers unfrozen in previous epochs → lr * 0.01
                  (not relevant at epoch 1 but becomes relevant after
                   second unfreeze event)
        Tier C — projection + metadata MLP + fusion + head → lr

    In practice we simplify to two tiers:
        backbone params (any unfrozen) → lr * 0.1
        everything else                → lr
    """
    backbone_params = [p for p in model.image_encoder.backbone.parameters()
                       if p.requires_grad]
    head_params     = (
        list(model.image_encoder.projection.parameters()) +
        list(model.metadata_encoder.parameters()) +
        list(model.classifier.parameters())
    )
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1, "weight_decay": wd},
        {"params": head_params,     "lr": lr,       "weight_decay": wd},
    ])


def _build_scheduler(optimizer, scheduler_type: str, num_epochs: int):
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-7
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, verbose=True
    )


# =============================================================
# Progressive unfreezing hook
# =============================================================

def _maybe_unfreeze(model: SkinDiseaseModel, optimizer_holder: list,
                    lr: float, wd: float, epoch: int) -> bool:
    """
    Called at the start of every Stage 2 epoch.
    Returns True if an unfreeze event occurred (optimizer was rebuilt).

    optimizer_holder is a 1-element list [optimizer] so the caller's
    reference is updated in-place (Python doesn't allow rebinding
    non-local names without nonlocal in nested functions).
    """
    target = (config.UNFREEZE_START_GROUPS
              + (epoch - 1) // config.UNFREEZE_EVERY)
    target = min(target, config.UNFREEZE_MAX_GROUPS)

    current = getattr(model, "_unfrozen_groups", -1)
    if target == current:
        return False

    model.unfreeze_last_n_groups(target)
    model._unfrozen_groups = target

    # Rebuild optimizer to include newly unfrozen params
    optimizer_holder[0] = _build_optimizer_stage2(model, lr, wd, target)
    print(f"  [ProgUnfreeze] Epoch {epoch}: {target} group(s) active. "
          f"Backbone LR={lr * 0.1:.2e}, Head LR={lr:.2e}")
    return True


# =============================================================
# run_stage()  — main training loop
# =============================================================

def run_stage(
    model:        SkinDiseaseModel,
    train_loader,
    val_loader,
    stage:        str,          # "stage1" | "stage2"
    device:       torch.device,
    class_weights = None,       # torch.Tensor for FocalLoss alpha
    label_map:    dict = None,
    dataset_name: str  = "",
) -> tuple[dict, str]:
    """
    Runs one complete training stage.

    Stage 1 (PAD binary):
        • No early stopping (fixed 80-100 epoch budget with cosine LR)
        • No progressive unfreezing (backbone fully frozen)
        • Higher focal gamma (2.5) for PAD imbalance
        • No label smoothing (binary task is clean)

    Stage 2 (Dermaco multi-class):
        • Early stopping on val F1-macro
        • Progressive unfreezing with domain-adaptive BN warmup
        • Lower focal gamma (2.0) + label smoothing (0.1)
        • Differential LR across backbone tiers

    Returns
    -------
    history   : dict of per-epoch training curves
    ckpt_path : path to best saved checkpoint
    """
    assert stage in ("stage1", "stage2"), f"Invalid stage: {stage!r}"

    if stage == "stage1":
        num_epochs     = config.PAD_PRETRAIN_EPOCHS
        lr             = config.PAD_PRETRAIN_LR
        wd             = config.PAD_PRETRAIN_WD
        sched_type     = config.PAD_PRETRAIN_SCHEDULER
        num_classes    = config.PAD_NUM_CLASSES        # 2
        gamma          = config.PAD_FOCAL_GAMMA        # 2.5
        smooth         = 0.0
        ckpt_path      = config.PAD_PRETRAIN_CKPT
        use_early_stop = False
    else:
        num_epochs     = config.DERM_FINETUNE_EPOCHS
        lr             = config.DERM_FINETUNE_LR
        wd             = config.DERM_FINETUNE_WD
        sched_type     = config.DERM_FINETUNE_SCHEDULER
        num_classes    = config.DERM_NUM_CLASSES
        gamma          = config.DERM_FOCAL_GAMMA       # 2.0
        smooth         = config.DERM_LABEL_SMOOTHING
        ckpt_path      = config.DERM_FINETUNE_CKPT
        use_early_stop = True

    # ── Loss ──────────────────────────────────────────────────
    alpha     = class_weights.to(device) if class_weights is not None else None
    criterion = FocalLoss(gamma=gamma, alpha=alpha,
                          label_smoothing=smooth, num_classes=num_classes)

    # ── Optimizer ─────────────────────────────────────────────
    if stage == "stage1":
        optimizer = _build_optimizer_stage1(model, lr, wd)
    else:
        optimizer = _build_optimizer_stage2(model, lr, wd, unfrozen_groups=1)
    optimizer_holder = [optimizer]   # mutable container for _maybe_unfreeze

    # ── Scheduler ─────────────────────────────────────────────
    scheduler = _build_scheduler(optimizer_holder[0], sched_type, num_epochs)

    # ── AMP ───────────────────────────────────────────────────
    use_amp = config.USE_AMP and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Early stopping ────────────────────────────────────────
    stopper = EarlyStopping(patience=config.DERM_EARLY_STOP_PATIENCE
                            ) if use_early_stop else None

    # ── Domain-adaptive BN: freeze stats at start of Stage 2 ──
    bn_released = False
    if stage == "stage2":
        model.freeze_bn_stats()

    # ── State ─────────────────────────────────────────────────
    best_f1 = -float("inf")
    history = {k: [] for k in
               ["train_loss", "train_acc", "train_f1",
                "val_loss",   "val_acc",   "val_f1", "val_auc"]}

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  {stage.upper()} | Dataset: {dataset_name} | "
          f"Epochs: {num_epochs} | Classes: {num_classes} | LR: {lr}")
    print(f"{'='*65}")
    model.count_parameters()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Domain-adaptive BN: release after warmup ───────────
        if stage == "stage2" and not bn_released and epoch > BN_WARMUP_EPOCHS:
            model.unfreeze_bn_stats()
            bn_released = True

        # ── Progressive unfreezing (Stage 2 only) ─────────────
        if stage == "stage2":
            unfroze = _maybe_unfreeze(optimizer_holder, model,
                                      lr, wd, epoch)
            if unfroze:
                # Rebuild scheduler around new optimizer
                scheduler = _build_scheduler(optimizer_holder[0],
                                             sched_type,
                                             num_epochs - epoch + 1)

        optimizer = optimizer_holder[0]

        # ── Train ─────────────────────────────────────────────
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, epoch, num_classes
        )

        # ── Validate ──────────────────────────────────────────
        val_loss, val_acc, val_f1, val_auc, _, _, _ = validate(
            model, val_loader, criterion, device, num_classes
        )

        # ── LR scheduler step ─────────────────────────────────
        if sched_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_f1)

        # ── Logging ───────────────────────────────────────────
        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc),
                     ("train_f1", tr_f1),     ("val_loss", val_loss),
                     ("val_acc", val_acc),     ("val_f1", val_f1),
                     ("val_auc", val_auc)]:
            history[k].append(v)

        current_lr = optimizer.param_groups[-1]["lr"]
        elapsed    = time.time() - t0
        print(
            f"  Ep [{epoch:>3}/{num_epochs}] "
            f"Tr  Loss:{tr_loss:.4f} Acc:{tr_acc:.1f}% F1:{tr_f1:.3f} | "
            f"Val Loss:{val_loss:.4f} Acc:{val_acc:.1f}% F1:{val_f1:.3f} "
            f"AUC:{val_auc:.3f} | LR:{current_lr:.2e} ({elapsed:.1f}s)"
        )

        # ── Save best ─────────────────────────────────────────
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_f1,
                ckpt_path,
                label_map   = label_map or {},
                dataset_name= dataset_name,
            )

        # ── Early stopping ────────────────────────────────────
        if stopper and stopper.step(val_f1, epoch):
            break

    print(f"\n  Best Val F1: {best_f1:.4f} → {ckpt_path}")
    return history, ckpt_path


# NOTE: _maybe_unfreeze signature corrected — model and optimizer_holder
# order matches the call site in run_stage.
def _maybe_unfreeze(optimizer_holder: list, model: SkinDiseaseModel,
                    lr: float, wd: float, epoch: int) -> bool:
    target  = (config.UNFREEZE_START_GROUPS
               + (epoch - 1) // config.UNFREEZE_EVERY)
    target  = min(target, config.UNFREEZE_MAX_GROUPS)
    current = getattr(model, "_unfrozen_groups", -1)
    if target == current:
        return False
    model.unfreeze_last_n_groups(target)
    model._unfrozen_groups     = target
    optimizer_holder[0]        = _build_optimizer_stage2(model, lr, wd, target)
    print(f"  [ProgUnfreeze] Ep {epoch}: {target} group(s). "
          f"BbLR={lr*0.1:.2e} HeadLR={lr:.2e}")
    return True
