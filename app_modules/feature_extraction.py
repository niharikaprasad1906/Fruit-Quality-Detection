"""
Feature Extraction Module
=========================
Calculates the exact 7 HSV colour and GLCM-style texture features
matching the training pipeline in preprocessing/extract_features.py.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
import pandas as pd
from .model_loader import FEATURE_COLUMNS

# Explanations of each feature for dashboard display
FEATURE_METADATA = {
    "mean_hue": {
        "name": "Mean Hue",
        "category": "Colour (HSV)",
        "unit": "Angle (0-179)",
        "description": "Represents the dominant perceived pigment or colour tone of the fruit skin.",
        "role": "Differentiates fruit types (e.g., red strawberries vs green apples/guavas) and ripening/browning."
    },
    "mean_saturation": {
        "name": "Mean Saturation",
        "category": "Colour (HSV)",
        "unit": "Intensity (0-255)",
        "description": "Measures colour purity and richness of the fruit skin.",
        "role": "Fresh fruits generally maintain vibrant saturation, while rotting fruit loses vividness."
    },
    "mean_brightness": {
        "name": "Mean Brightness (Value)",
        "category": "Colour (HSV)",
        "unit": "Luminance (0-255)",
        "description": "Quantifies the lightness and reflectance of the segmented fruit surface.",
        "role": "Dark spots, fungal rot, and necrosis significantly reduce local and overall surface brightness."
    },
    "contrast": {
        "name": "Contrast",
        "category": "Texture (GLCM)",
        "unit": "Intensity Variance",
        "description": "Measures local gray-level variations between adjacent foreground pixel pairs.",
        "role": "Higher contrast indicates blotchy, bruised, or moldy rotten textures compared to smooth skin."
    },
    "homogeneity": {
        "name": "Homogeneity",
        "category": "Texture (GLCM)",
        "unit": "Similarity (0-1)",
        "description": "Evaluates the closeness of the GLCM distribution to its diagonal (surface uniformity).",
        "role": "Fresh fruits with consistent skin exhibit higher homogeneity than degraded rotten surfaces."
    },
    "energy": {
        "name": "Energy (Angular Second Moment)",
        "category": "Texture (GLCM)",
        "unit": "Uniformity (0-1)",
        "description": "Square root of the sum of squared GLCM elements; reflects structural regularity.",
        "role": "Represents orderly texture patterns across the fruit skin."
    },
    "correlation": {
        "name": "Correlation",
        "category": "Texture (GLCM)",
        "unit": "Linear Dependence (-1 to 1)",
        "description": "Measures joint probability occurrence of specified pixel pairs across gray scales.",
        "role": "Evaluates linear spatial dependencies and directional pattern consistency across the fruit."
    }
}


def extract_features_from_segmented(
    rgb_image: np.ndarray,
    mask: np.ndarray
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Extracts the exact 7 colour and texture features from the segmented RGB image and mask.

    Parameters:
        rgb_image: (224, 224, 3) uint8 numpy array
        mask: (224, 224) boolean numpy array of foreground pixels

    Returns:
        tuple (feature_dict, feature_df)
    """
    # --------------------------------------------------
    # 1. HSV COLOUR FEATURES
    # --------------------------------------------------
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    fruit_pixels = hsv[mask]

    if len(fruit_pixels) == 0:
        raise ValueError("No foreground fruit pixels available for colour feature calculation.")

    mean_hue = float(np.mean(fruit_pixels[:, 0]))
    mean_saturation = float(np.mean(fruit_pixels[:, 1]))
    mean_brightness = float(np.mean(fruit_pixels[:, 2]))

    # --------------------------------------------------
    # 2. GLCM-STYLE TEXTURE FEATURES
    # --------------------------------------------------
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_quantized = (gray // 32).astype(np.uint8)

    left_pixels = gray_quantized[:, :-1]
    right_pixels = gray_quantized[:, 1:]

    valid_pairs = mask[:, :-1] & mask[:, 1:]

    left_values = left_pixels[valid_pairs]
    right_values = right_pixels[valid_pairs]

    if len(left_values) < 10:
        raise ValueError("Not enough foreground texture pairs detected for GLCM calculation.")

    # 8-bin GLCM matrix
    glcm = np.histogram2d(
        left_values,
        right_values,
        bins=8,
        range=((0, 8), (0, 8))
    )[0]

    glcm_sum = glcm.sum()
    if glcm_sum == 0:
        raise ValueError("GLCM sum is zero; unable to normalize texture matrix.")

    glcm = glcm / glcm_sum

    i, j = np.indices((8, 8))

    contrast = float(np.sum(glcm * ((i - j) ** 2)))
    homogeneity = float(np.sum(glcm / (1 + ((i - j) ** 2))))
    energy = float(np.sqrt(np.sum(glcm ** 2)))

    mean_i = np.sum(i * glcm)
    mean_j = np.sum(j * glcm)

    std_i = np.sqrt(np.sum(((i - mean_i) ** 2) * glcm))
    std_j = np.sqrt(np.sum(((j - mean_j) ** 2) * glcm))

    if std_i == 0 or std_j == 0:
        correlation = 1.0
    else:
        correlation = float(np.sum(((i - mean_i) * (j - mean_j) * glcm)) / (std_i * std_j))

    feature_dict = {
        "mean_hue": mean_hue,
        "mean_saturation": mean_saturation,
        "mean_brightness": mean_brightness,
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
    }

    # Verify all extracted features are finite numbers
    for k, v in feature_dict.items():
        if not np.isfinite(v):
            raise ValueError(f"Feature calculation resulted in non-finite value for '{k}': {v}")

    # Create DataFrame with exact column ordering
    feature_df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]

    return feature_dict, feature_df
