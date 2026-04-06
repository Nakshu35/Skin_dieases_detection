# =============================================================
# gradcam.py  —  Grad-CAM visual explanations
# =============================================================
#
# Grad-CAM answers: "Which part of the image did the model look at?"
# It highlights the important regions using a heatmap overlay.
#
# How it works (simply):
#   1. Run forward pass to get prediction
#   2. Backpropagate gradients to the LAST conv layer of ResNet
#   3. Compute average gradient per feature map channel
#   4. Weight each feature map by its gradient average
#   5. Sum weighted maps → saliency map → resize → overlay on image

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config


class GradCAM:
    """
    Grad-CAM for the ImageEncoder (ResNet50 backbone).

    Usage:
        gcam = GradCAM(model)
        heatmap = gcam.generate(image_tensor, metadata_tensor)
        gcam.visualize(original_image, heatmap, pred_class)
    """

    def __init__(self, model):
        self.model = model
        self.gradients   = None   # will store gradients here
        self.activations = None   # will store feature maps here

        # Hook into the last conv layer of ResNet50
        # layer4[-1].conv3 is the last conv in the last residual block
        target_layer = model.image_encoder.backbone.layer4[-1].conv3

        # Forward hook: captures the output (feature maps) of target layer
        self._forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )

        # Backward hook: captures gradients flowing back through target layer
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        """Called automatically during forward pass"""
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Called automatically during backward pass"""
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, metadata_tensor, target_class=None):
        """
        Generates Grad-CAM heatmap.

        Parameters
        ----------
        image_tensor    : (1, 3, 224, 224)  — preprocessed image
        metadata_tensor : (1, 4)            — metadata
        target_class    : int or None       — class to explain.
                          If None, uses the predicted class.

        Returns
        -------
        heatmap : numpy array (224, 224) — values in [0, 1]
        pred_class_idx : int
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        image_tensor    = image_tensor.to(device)
        metadata_tensor = metadata_tensor.to(device)

        # Forward pass (need gradients for this, so no torch.no_grad())
        image_tensor.requires_grad_(False)
        logits = self.model(image_tensor, metadata_tensor)  # (1, NUM_CLASSES)
        probs  = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = probs.argmax(dim=1).item()

        # Zero gradients, then backpropagate only for target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # ── Compute Grad-CAM ──────────────────────────────────
        # gradients shape: (1, C, H, W)
        # activations shape: (1, C, H, W)

        # Global Average Pooling over spatial dims → weights per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of feature maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU: keep only positive contributions
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(cam,
                            size=(config.IMG_SIZE, config.IMG_SIZE),
                            mode="bilinear",
                            align_corners=False)   # (1, 1, 224, 224)

        cam = cam.squeeze().cpu().numpy()           # (224, 224)

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, target_class

    def visualize(self, original_image, heatmap, pred_class_idx,
                  save_path=None, alpha=0.4):
        """
        Overlays heatmap on original image and displays it.

        Parameters
        ----------
        original_image  : PIL.Image  — original (unnormalized) image
        heatmap         : numpy (224, 224)
        pred_class_idx  : int
        save_path       : str or None
        alpha           : float  — heatmap transparency (0=invisible, 1=opaque)
        """
        # Resize original image to match
        orig = original_image.resize((config.IMG_SIZE, config.IMG_SIZE))
        orig_array = np.array(orig) / 255.0

        # Apply colormap to heatmap (jet: blue=low, red=high)
        colormap   = cm.get_cmap("jet")
        colored_hm = colormap(heatmap)[:, :, :3]   # drop alpha channel → (H, W, 3)

        # Overlay: blend original image + heatmap
        overlay = (1 - alpha) * orig_array + alpha * colored_hm
        overlay = np.clip(overlay, 0, 1)

        pred_name = config.DISEASE_CLASSES[pred_class_idx]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        axes[0].imshow(orig_array)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay — Predicted: {pred_name.replace('_', ' ').title()}")
        axes[2].axis("off")

        plt.suptitle("Grad-CAM Explanation", fontsize=14)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"  [GradCAM] Saved → {save_path}")
        plt.show()

    def remove_hooks(self):
        """Always call this when done to free memory"""
        self._forward_hook.remove()
        self._backward_hook.remove()


def run_gradcam_on_samples(model, test_loader, device, num_samples=5,
                           save_dir=config.RESULTS_DIR):
    """
    Convenience function: runs Grad-CAM on the first N test samples.
    """
    gcam = GradCAM(model)
    inv_transform = transforms.Normalize(
        mean=[-m/s for m, s in zip(config.MEAN, config.STD)],
        std=[1/s for s in config.STD]
    )
    to_pil = transforms.ToPILImage()

    count = 0
    for images, metadata, labels in test_loader:
        for i in range(images.size(0)):
            if count >= num_samples:
                break

            img_tensor  = images[i:i+1]
            meta_tensor = metadata[i:i+1]
            true_label  = labels[i].item()

            heatmap, pred_idx = gcam.generate(img_tensor, meta_tensor)

            # De-normalize image for display
            orig_img = to_pil(inv_transform(images[i]).clamp(0, 1))

            save_path = os.path.join(save_dir, "gradcam",
                                     f"sample_{count}_true_{config.DISEASE_CLASSES[true_label]}.png")
            gcam.visualize(orig_img, heatmap, pred_idx, save_path=save_path)
            count += 1

        if count >= num_samples:
            break

    gcam.remove_hooks()
    print(f"  [GradCAM] Generated {count} visualizations in {save_dir}/gradcam/")
