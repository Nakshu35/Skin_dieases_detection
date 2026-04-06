# =============================================================
# predict.py  —  Run inference on a single image + metadata
# =============================================================
#
# HOW TO USE:
#   python predict.py --image path/to/image.jpg --age 45 --sex male
#                     --location back --skin_type 2

import argparse
import torch
from PIL import Image

import config
from model    import SkinDiseaseModel
from dataset  import (encode_sex, encode_location, normalize_age,
                      encode_skin_type, LOCATION_VOCAB, get_val_transform)
from evaluate import predict_single
from trainer  import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Skin Disease Prediction")
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to skin lesion image")
    parser.add_argument("--age",        type=float, default=None)
    parser.add_argument("--sex",        type=str,   default=None,
                        help="male / female")
    parser.add_argument("--location",   type=str,   default=None,
                        help="Body location e.g. back, face, arm")
    parser.add_argument("--skin_type",  type=int,   default=None,
                        help="Fitzpatrick skin type 1-6")
    parser.add_argument("--checkpoint", type=str,
                        default=f"{config.CHECKPOINT_DIR}/best_finetune.pth",
                        help="Path to saved model weights")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────
    model = SkinDiseaseModel(freeze_backbone=False).to(device)
    load_checkpoint(model, optimizer=None,
                    filename=args.checkpoint, device=device)
    model.eval()

    # ── Preprocess image ────────────────────────────────────
    transform    = get_val_transform()
    raw_image    = Image.open(args.image).convert("RGB")
    image_tensor = transform(raw_image).unsqueeze(0)    # add batch dim → (1, 3, 224, 224)

    # ── Encode metadata ─────────────────────────────────────
    age      = normalize_age(args.age)
    sex      = encode_sex(args.sex)
    location = encode_location(args.location, LOCATION_VOCAB)
    skin     = encode_skin_type(args.skin_type)

    metadata_tensor = torch.tensor([[age, sex, location, skin]],
                                    dtype=torch.float32)

    # ── Predict ─────────────────────────────────────────────
    predict_single(model, image_tensor, metadata_tensor, device)

    # ── Grad-CAM for this image ──────────────────────────────
    from gradcam import GradCAM
    import torchvision.transforms as T

    gcam    = GradCAM(model)
    heatmap, pred_idx = gcam.generate(image_tensor, metadata_tensor)
    gcam.visualize(raw_image, heatmap, pred_idx,
                   save_path=f"{config.RESULTS_DIR}/predict_gradcam.png")
    gcam.remove_hooks()


if __name__ == "__main__":
    main()
