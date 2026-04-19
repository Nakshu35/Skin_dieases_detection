# =============================================================
# model.py  —  Cross-Dataset Transfer Learning Model
# =============================================================
#
# KEY DESIGN DECISIONS vs. PREVIOUS VERSION:
#
#   1. CROSS-DATASET WEIGHT TRANSFER SAFETY
#      The central problem in cross-dataset transfer is that Stage 1 and
#      Stage 2 may have different metadata input dimensions OR different
#      number of output classes. transfer_weights_cross_dataset() handles
#      both mismatches explicitly:
#        • Head mismatch: always expected → re-initialized.
#        • Metadata MLP mismatch: detected at runtime → optionally skipped
#          based on config.TRANSFER_META_MLP.
#        • Backbone: always transferred (this is the primary transfer target).
#
#   2. META_INPUT_DIM IS PASSED AT CONSTRUCTION TIME
#      The previous version hardcoded META_INPUT_DIM from config.META_INPUT_DIM.
#      Now SkinDiseaseModel takes meta_input_dim as an explicit argument so
#      Stage 1 (PAD) and Stage 2 (Dermaco-In) can have different metadata
#      schemas without changing config.
#
#   3. DOMAIN-ADAPTIVE BATCHNORM
#      freeze_bn_stats() allows freezing BatchNorm running stats (mean/var)
#      when fine-tuning on Dermaco-In. This prevents the Dermaco-In domain
#      statistics from corrupting the PAD-tuned running stats in the early
#      fine-tuning epochs when few Dermaco-In batches have been seen.
#      After UNFREEZE_EVERY * 2 epochs, BN stats are released.
#
#   4. WEIGHT TRANSFER REPORT
#      transfer_weights_cross_dataset() prints a detailed report:
#        - How many layers were transferred
#        - Which layers were skipped (mismatch or missing)
#        - What % of non-head weights were successfully transferred
#      This makes debugging weight-loading issues easy.
#
#   5. ENSEMBLE CHECKPOINT FORMAT EXTENDED
#      Checkpoint now stores dataset name alongside label_map so an
#      ensemble of PAD-pretrained Dermaco-In models can be verified to
#      share the same class mapping before averaging logits.

import torch
import torch.nn as nn
from torchvision import models

import config


# =============================================================
# 1. Image Encoder  (ResNet50)
# =============================================================

class ImageEncoder(nn.Module):
    """
    ResNet50 backbone with learnable projection head.

    Progressive unfreezing API:
        freeze_all()              → use at Stage 1 start
        unfreeze_last_n_groups(n) → use at Stage 2 with progressive schedule
        freeze_bn_stats()         → freeze BN running stats only (domain adaptation)
        unfreeze_bn_stats()       → release BN running stats after domain warmup
    """

    def __init__(self, output_dim: int = 512, freeze_backbone: bool = True):
        super().__init__()
        resnet        = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features   = resnet.fc.in_features   # 2048
        resnet.fc     = nn.Identity()
        self.backbone = resnet

        # 2048 → output_dim projection with regularization
        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
        )
        if freeze_backbone:
            self.freeze_all()

    def get_layer_groups(self) -> list:
        """
        5 ResNet50 groups ordered from input to output:
            0: stem (conv1 + bn1 + relu + maxpool)
            1: layer1 (3 bottleneck blocks)
            2: layer2 (4 bottleneck blocks)
            3: layer3 (6 bottleneck blocks)
            4: layer4 (3 bottleneck blocks)
        Progressive unfreezing starts from group 4 (closest to output).
        """
        b = self.backbone
        return [
            nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool),
            b.layer1, b.layer2, b.layer3, b.layer4,
        ]

    def freeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_groups(self, n: int):
        """
        Freeze all backbone params, then selectively unfreeze the
        last n layer groups. Projection head always stays trainable.
        """
        self.freeze_all()
        groups = self.get_layer_groups()
        n = min(max(n, 0), len(groups))
        for group in groups[-n:]:
            for p in group.parameters():
                p.requires_grad = True
        n_trainable = sum(p.numel() for p in self.backbone.parameters()
                          if p.requires_grad)
        print(f"[ImageEncoder] {n} group(s) unfrozen | "
              f"Trainable backbone params: {n_trainable:,}")

    def freeze_bn_stats(self):
        """
        Keep BN layers in eval() mode during forward pass so their
        running_mean / running_var are NOT updated by Dermaco-In batches.
        Call this at the start of Stage 2 to preserve PAD domain stats.
        Only affects BN in the backbone (not in projection head).
        """
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        print("[ImageEncoder] BatchNorm stats frozen (domain adaptation mode).")

    def unfreeze_bn_stats(self):
        """Release BN stats after the domain warmup period."""
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                for p in m.parameters():
                    p.requires_grad = True
        print("[ImageEncoder] BatchNorm stats released.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(x))   # (B, output_dim)


# =============================================================
# 2. Metadata Encoder  (MLP)
# =============================================================

class MetadataEncoder(nn.Module):
    """
    Clinical metadata → embedding vector.
    input_dim is explicit so PAD (4 features) and Dermaco-In (4 or 3)
    can use different dims without a global config change.
    """

    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# =============================================================
# 3. Fusion Classifier
# =============================================================

class FusionClassifier(nn.Module):
    """
    Concat(img_feat, meta_feat) → class logits.
    Head is kept as a separate attribute for surgical replacement.
    """

    def __init__(self, img_dim: int = 512, meta_dim: int = 128,
                 num_classes: int = 2):
        super().__init__()
        fused = img_dim + meta_dim   # 640
        self.fusion = nn.Sequential(
            nn.Linear(fused, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.3),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, img_feat: torch.Tensor,
                meta_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([img_feat, meta_feat], dim=1)
        return self.head(self.fusion(combined))


# =============================================================
# 4. Full Model
# =============================================================

class SkinDiseaseModel(nn.Module):
    """
    Multimodal skin disease classifier.

    Parameters
    ----------
    num_classes     : number of output classes (2 for Stage 1, N for Stage 2)
    meta_input_dim  : number of metadata features (PAD: 4, Dermaco: 4 or 3)
    freeze_backbone : freeze ResNet at construction time
    img_output_dim  : projection head output (512)
    meta_output_dim : metadata encoder output (128)
    """

    def __init__(self,
                 num_classes:     int  = 2,
                 meta_input_dim:  int  = 4,
                 freeze_backbone: bool = True,
                 img_output_dim:  int  = 512,
                 meta_output_dim: int  = 128):
        super().__init__()
        self.img_dim  = img_output_dim
        self.meta_dim = meta_output_dim

        self.image_encoder    = ImageEncoder(output_dim=img_output_dim,
                                             freeze_backbone=freeze_backbone)
        self.metadata_encoder = MetadataEncoder(input_dim=meta_input_dim,
                                                output_dim=meta_output_dim)
        self.classifier       = FusionClassifier(img_dim=img_output_dim,
                                                 meta_dim=meta_output_dim,
                                                 num_classes=num_classes)

    def forward(self, images: torch.Tensor,
                metadata: torch.Tensor) -> torch.Tensor:
        return self.classifier(
            self.image_encoder(images),
            self.metadata_encoder(metadata),
        )

    def replace_classifier_head(self, new_num_classes: int):
        """Replaces ONLY the final linear head, preserving fusion MLP weights."""
        in_f = self.classifier.head.in_features
        self.classifier.head = nn.Linear(in_f, new_num_classes)
        # Xavier init for the new head — better than default random init
        nn.init.xavier_uniform_(self.classifier.head.weight)
        nn.init.zeros_(self.classifier.head.bias)
        print(f"[Model] Head replaced: {in_f} → {new_num_classes} (Xavier init)")

    def freeze_backbone(self):
        self.image_encoder.freeze_all()

    def unfreeze_last_n_groups(self, n: int):
        self.image_encoder.unfreeze_last_n_groups(n)

    def freeze_bn_stats(self):
        self.image_encoder.freeze_bn_stats()

    def unfreeze_bn_stats(self):
        self.image_encoder.unfreeze_bn_stats()

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total: {total:,} | Trainable: {trainable:,}")


# =============================================================
# 5. Cross-Dataset Weight Transfer
# =============================================================

def transfer_weights_cross_dataset(
    stage2_model: SkinDiseaseModel,
    ckpt_path: str,
    device,
    transfer_meta_mlp: bool = True,
) -> dict:
    """
    Transfers weights from a Stage 1 PAD checkpoint into a Stage 2
    Dermaco-In model with surgical precision.

    Transfer policy:
    ┌──────────────────────────────────┬──────────────────────────────┐
    │ Component                        │ Action                       │
    ├──────────────────────────────────┼──────────────────────────────┤
    │ ResNet50 backbone                │ Always transferred           │
    │ Projection head (2048→512)       │ Always transferred           │
    │ Metadata encoder MLP             │ Transferred if same input    │
    │                                  │ dim; skipped if mismatch     │
    │ Fusion MLP (640→256→128)         │ Always transferred           │
    │ Classifier head (128→N_classes)  │ ALWAYS skipped (size differs)│
    └──────────────────────────────────┴──────────────────────────────┘

    WHY NOT strict=True?
        The head has shape (128, 2) in Stage 1 and (128, N) in Stage 2.
        strict=True would raise an error. strict=False silently skips
        all mismatched keys, but we add explicit logging so you can
        verify which keys were actually transferred.

    WHY TRANSFER FUSION MLP?
        The fusion MLP has learned to combine visual and clinical features
        into a meaningful 128-dim representation during Stage 1. Even though
        the class space changes, this combination skill transfers well.

    Parameters
    ----------
    stage2_model     : SkinDiseaseModel built for Dermaco-In (N classes)
    ckpt_path        : path to Stage 1 PAD checkpoint
    device           : torch.device
    transfer_meta_mlp: whether to attempt metadata MLP weight transfer

    Returns
    -------
    report : dict with keys transferred, skipped, mismatch — for logging
    """
    ckpt       = torch.load(ckpt_path, map_location=device)
    src_state  = ckpt["model_state"]
    dst_state  = stage2_model.state_dict()

    transferred = []
    skipped     = []
    mismatched  = []

    for key, src_tensor in src_state.items():
        # ── 1. Always skip the classifier head ─────────────────
        if key.startswith("classifier.head"):
            skipped.append(key)
            continue

        # ── 2. Optionally skip metadata encoder ────────────────
        if key.startswith("metadata_encoder") and not transfer_meta_mlp:
            skipped.append(key)
            continue

        # ── 3. Check that key exists in target model ───────────
        if key not in dst_state:
            skipped.append(key)
            continue

        # ── 4. Check shape compatibility ───────────────────────
        if dst_state[key].shape != src_tensor.shape:
            mismatched.append(f"{key}: src={src_tensor.shape} "
                              f"dst={dst_state[key].shape}")
            skipped.append(key)
            continue

        # ── 5. Transfer ────────────────────────────────────────
        dst_state[key] = src_tensor.clone()
        transferred.append(key)

    stage2_model.load_state_dict(dst_state, strict=True)

    # Print transfer report
    total_keys = len(src_state)
    print(f"\n[Transfer Report]  {ckpt_path}")
    print(f"  Transferred : {len(transferred):>4} / {total_keys} keys")
    print(f"  Skipped     : {len(skipped):>4}  (head + policy exclusions)")
    if mismatched:
        print(f"  ⚠ Mismatched shapes ({len(mismatched)} keys — skipped):")
        for m in mismatched:
            print(f"      {m}")
    pct = len(transferred) / max(total_keys, 1) * 100
    print(f"  Transfer efficiency: {pct:.1f}%\n")

    return {"transferred": transferred, "skipped": skipped,
            "mismatched": mismatched}


def load_stage2_final_weights(model: SkinDiseaseModel, path: str, device):
    """Strict load for Stage 3 evaluation (model must match exactly)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"[Checkpoint] Stage 2 weights loaded ← {path} "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"val_F1={ckpt.get('val_metric', '?')})")
    return ckpt


def save_checkpoint(model: SkinDiseaseModel, optimizer, scheduler,
                    epoch: int, val_metric: float,
                    path: str, label_map: dict,
                    dataset_name: str = ""):
    """
    Ensemble-ready checkpoint. Stores architecture metadata, label mapping,
    and dataset name so multiple checkpoints can be verified before ensembling.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "val_metric":   val_metric,
        "model_state":  model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict() if scheduler else None,
        "arch": {
            "img_output_dim":  model.img_dim,
            "meta_output_dim": model.meta_dim,
            "num_classes":     model.classifier.head.out_features,
            "meta_input_dim":  model.metadata_encoder.input_dim,
        },
        "label_map":    label_map,
        "dataset":      dataset_name,
    }, path)
    print(f"[Checkpoint] Saved → {path}")


import os   # needed by save_checkpoint — placed after class defs for clarity


# =============================================================
# 6. Temperature Scaling  (post-hoc calibration)
# =============================================================

class TemperatureScaler(nn.Module):
    """
    Wraps a trained model and learns a single temperature T.
    Optimized using LBFGS on the validation set (Dermaco-In val).
    Model weights are NOT updated — only T.

    Reference: Guo et al., ICML 2017.
    """

    def __init__(self, model: SkinDiseaseModel):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, images, metadata):
        return self.model(images, metadata) / self.temperature

    def calibrate(self, val_loader, device,
                  lr: float = 0.01, max_iter: int = 50):
        self.to(device)
        self.model.eval()
        nll       = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature],
                                       lr=lr, max_iter=max_iter)
        logits_all, labels_all = [], []

        with torch.no_grad():
            for imgs, meta, lbls in val_loader:
                logits_all.append(
                    self.model(imgs.to(device), meta.to(device)).cpu()
                )
                labels_all.append(lbls)

        logits_all = torch.cat(logits_all)
        labels_all = torch.cat(labels_all)

        def _closure():
            optimizer.zero_grad()
            loss = nll(logits_all / self.temperature, labels_all)
            loss.backward()
            return loss

        optimizer.step(_closure)
        print(f"[Calibration] Temperature = {self.temperature.item():.4f}")
