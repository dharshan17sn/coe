"""
Satellite Image Processing Pipeline:
1. Loads full-disk/sector INSAT-3D Infrared (IR1) imagery.
2. Crops storm-centered region (cyclone eye and spiral rainbands).
3. Applies cloud region enhancement and noise removal.
4. Resizes to standard model resolution (224x224) for CNN/AlexNet/EfficientNet.
"""

import os
import glob
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import pandas as pd

RAW_IMG_DIR = os.path.join("datasets", "raw", "insat3d")
PROCESSED_IMG_DIR = os.path.join("datasets", "processed", "images")
METADATA_PATH = os.path.join("datasets", "processed", "metadata.csv")


def enhance_cloud_structure(crop_img):
    """
    Enhances cyclone cloud structures and eye definition from IR imagery:
    - Converts to Grayscale (representing cloud-top temperatures).
    - Contrast stretching / Autocontrast (highlights cold cloud-tops in white).
    - Sharpness enhancement to define cyclone spiral bands.
    """
    # Convert to grayscale
    gray = crop_img.convert("L")
    # Enhance contrast (cloud-top cold peaks vs warmer background)
    contrasted = ImageOps.autocontrast(gray, cutoff=2)
    # Enhance sharpness for spiral bands
    enhancer = ImageEnhance.Sharpness(contrasted)
    sharpened = enhancer.enhance(1.5)
    return sharpened


def crop_and_standardize_satellite_frame(source_img_path, output_filename, box=None):
    """
    Crops region of interest from satellite image and standardizes to 224x224.
    """
    os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)
    img = Image.open(source_img_path)
    width, height = img.size

    # If no specific bounding box is provided, center-crop the Indian Ocean basin
    if box is None:
        # Default crop over Bay of Bengal / Arabian Sea region of the Asia sector
        left = int(width * 0.35)
        top = int(height * 0.40)
        right = int(width * 0.70)
        bottom = int(height * 0.75)
        box = (left, top, right, bottom)

    crop = img.crop(box)
    enhanced = enhance_cloud_structure(crop)
    final_224 = enhanced.resize((224, 224), Image.Resampling.LANCZOS)

    output_path = os.path.join(PROCESSED_IMG_DIR, output_filename)
    final_224.save(output_path, "PNG")
    return output_path


def demo_process_satellite_images():
    print("=" * 60)
    print("DEMO: PROCESSING INSAT-3D SATELLITE IMAGES FOR CNN")
    print("=" * 60)

    # Find downloaded INSAT-3D IR images
    ir_images = glob.glob(os.path.join(RAW_IMG_DIR, "*IR1*.jpg"))
    if not ir_images:
        print(f"No IR images found in {RAW_IMG_DIR}.")
        return

    for ir_img_path in ir_images:
        base = os.path.basename(ir_img_path)
        print(f"\nProcessing Satellite Frame: {base}")
        # Crop 1: Regional Indian Ocean / Bay of Bengal Sector
        out1 = crop_and_standardize_satellite_frame(
            ir_img_path,
            f"processed_224x224_BayOfBengal_{base.replace('.jpg', '.png')}"
        )
        print(f" [OK] Generated 224x224 CNN input: {out1}")

        # Crop 2: Arabian Sea Sector
        img = Image.open(ir_img_path)
        w, h = img.size
        as_box = (int(w * 0.20), int(h * 0.42), int(w * 0.50), int(h * 0.72))
        out2 = crop_and_standardize_satellite_frame(
            ir_img_path,
            f"processed_224x224_ArabianSea_{base.replace('.jpg', '.png')}",
            box=as_box
        )
        print(f" [OK] Generated 224x224 CNN input: {out2}")

    print("\n" + "=" * 60)
    print("Image preprocessing finished successfully!")
    print(f"Cropped 224x224 patches stored in: {PROCESSED_IMG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    demo_process_satellite_images()

