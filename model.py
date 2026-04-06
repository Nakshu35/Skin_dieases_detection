# =============================================================
# model.py  —  Full model architecture
# =============================================================
#
# Architecture at a glance:
#
#   Image  ──► ResNet50 (pretrained) ──► 512-dim vector
#                                              │
#   Metadata ──► MLP (4 → 64 → 128) ──► 128-dim vector
#                                              │
#                 Concat → [512 + 128 = 640] ──┘
#                                │
#                           Fusion MLP
#                                │
#                     Softmax (NUM_CLASSES)

import torch
import torch.nn as nn
from torchvision import models

import config


# =============================================================
# 1. Image Encoder  (ResNet50 backbone)
# =============================================================

class ImageEncoder(nn.Module):
    """
    Uses a pretrained ResNet50 to extract features from skin images.
    We remove the final classification layer and replace it with a
    smaller linear layer that outputs a 512-dim feature vector.
    """

    def __init__(self, output_dim=512, freeze_backbone=False):
        super().__init__()

        # Load ResNet50 with pretrained ImageNet weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the last fully-connected layer (we add our own)
        # resnet.fc was Linear(2048, 1000)  →  we replace it
        in_features = resnet.fc.in_features   # 2048

        resnet.fc = nn.Identity()             # pass 2048 features through unchanged
        self.backbone = resnet

        # Our custom projection layer: 2048 → output_dim
        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        # Optionally freeze backbone (useful during early pretraining)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x shape: (batch, 3, 224, 224)
        features = self.backbone(x)           # (batch, 2048)
        out = self.projection(features)       # (batch, 512)
        return out

    def unfreeze(self):
        """Call this when switching to fine-tuning stage"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# =============================================================
# 2. Metadata Encoder  (small MLP)
# =============================================================

class MetadataEncoder(nn.Module):
    """
    Small MLP that processes patient metadata:
        age, sex, location, skin_type  →  128-dim vector
    """

    def __init__(self, input_dim=config.META_INPUT_DIM, output_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x):
        # x shape: (batch, 4)
        return self.mlp(x)                    # (batch, 128)


# =============================================================
# 3. Fusion + Classifier
# =============================================================

class FusionClassifier(nn.Module):
    """
    Concatenates image features + metadata features, then classifies.
    """

    def __init__(self, img_dim=512, meta_dim=128, num_classes=config.NUM_CLASSES):
        super().__init__()

        fused_dim = img_dim + meta_dim        # 512 + 128 = 640

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(256, num_classes),      # final logits (no softmax here —
                                              # CrossEntropy/FocalLoss does that)
        )

    def forward(self, img_feat, meta_feat):
        # Concatenate along feature dimension
        combined = torch.cat([img_feat, meta_feat], dim=1)  # (batch, 640)
        logits = self.fusion(combined)                       # (batch, num_classes)
        return logits


# =============================================================
# 4. Full Model  (wraps all three parts)
# =============================================================

class SkinDiseaseModel(nn.Module):
    """
    The complete multimodal model.

    Usage:
        model = SkinDiseaseModel()
        logits = model(images, metadata)
    """

    def __init__(self,
                 img_output_dim=512,
                 meta_output_dim=128,
                 num_classes=config.NUM_CLASSES,
                 freeze_backbone=True):
        super().__init__()

        self.image_encoder    = ImageEncoder(output_dim=img_output_dim,
                                             freeze_backbone=freeze_backbone)
        self.metadata_encoder = MetadataEncoder(output_dim=meta_output_dim)
        self.classifier       = FusionClassifier(img_dim=img_output_dim,
                                                 meta_dim=meta_output_dim,
                                                 num_classes=num_classes)

    def forward(self, images, metadata):
        img_feat  = self.image_encoder(images)      # (batch, 512)
        meta_feat = self.metadata_encoder(metadata) # (batch, 128)
        logits    = self.classifier(img_feat, meta_feat)  # (batch, NUM_CLASSES)
        return logits

    def unfreeze_backbone(self):
        """Unfreeze ResNet backbone — call before fine-tuning stage"""
        self.image_encoder.unfreeze()
        print("[Model] ResNet backbone unfrozen for fine-tuning.")

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total params: {total:,}  |  Trainable: {trainable:,}")
