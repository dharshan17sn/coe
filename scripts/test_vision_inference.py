"""
Vision & Multimodal Inference Demonstration Script.
Passes a satellite image crop (and optional physical coordinates) through the trained
deep learning vision models to predict cyclone intensity and IMD classification.
"""

import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import CustomCycloneCNN, ModifiedAlexNet, MultimodalEfficientNet

IMD_CLASSES = [
    "Low Pressure Area",
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm",
]


def load_vision_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_from_image(image_path, model_path="models/custom_cnn.pt", model_type="cnn"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load image
    raw_img = Image.open(image_path).convert("RGB")
    transform = load_vision_transform()
    img_tensor = transform(raw_img).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    # Load model
    if model_type == "cnn":
        model = CustomCycloneCNN(in_channels=3, num_classes=8)
    elif model_type == "alexnet":
        model = ModifiedAlexNet(in_channels=3, num_classes=8, pretrained=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    weights = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(weights)
    model.eval()

    with torch.no_grad():
        p_wind_scaled, p_cls_logits = model(img_tensor)

        # Approximate wind scale from training distribution
        wind_mean, wind_std = 43.5, 22.0
        pred_wind = float(p_wind_scaled.item() * wind_std + wind_mean)
        probs = torch.softmax(p_cls_logits, dim=1).numpy()[0]
        pred_cls_idx = int(np.argmax(probs))
        confidence = float(probs[pred_cls_idx])

    return {
        "image_file": os.path.basename(image_path),
        "predicted_wind_kt": round(max(pred_wind, 17.0), 1),
        "predicted_wind_kmh": round(max(pred_wind, 17.0) * 1.852, 1),
        "predicted_class": IMD_CLASSES[pred_cls_idx] if pred_cls_idx < len(IMD_CLASSES) else "Unknown",
        "confidence_pct": round(confidence * 100, 1),
        "probabilities": {IMD_CLASSES[i]: round(float(probs[i]) * 100, 1) for i in range(min(len(IMD_CLASSES), len(probs)))},
    }


def demo_vision_inference():
    print("=" * 70)
    print("VISION MODEL INFERENCE DEMO ON PROCESSED SATELLITE IMAGERY")
    print("=" * 70)

    sample_imgs = [
        "datasets/processed/images/processed_224x224_BayOfBengal_INSAT3D_ASIA_IR1_20261109_041146.png",
        "datasets/processed/images/processed_224x224_ArabianSea_INSAT3D_ASIA_IR1_20261109_041146.png",
        "datasets/processed/images/processed_224x224_BayOfBengal_INSAT3D_RAPID_SCAN_IR1_20261109_041149.png",
    ]

    for p in sample_imgs:
        if os.path.exists(p):
            res = predict_from_image(p, model_path="models/custom_cnn.pt", model_type="cnn")
            print(f"\nImage Input: {res['image_file']}")
            print(f" -> Predicted Wind Speed: {res['predicted_wind_kt']} knots ({res['predicted_wind_kmh']} km/h)")
            print(f" -> IMD Classification:   {res['predicted_class']} (Confidence: {res['confidence_pct']}%)")
            print(" -> Class Probabilities:")
            for cls_name, prob in list(res['probabilities'].items())[:4]:
                print(f"     * {cls_name:<28}: {prob:>5.1f}%")

    print("\n" + "=" * 70)
    print("Inference completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demo_vision_inference()
