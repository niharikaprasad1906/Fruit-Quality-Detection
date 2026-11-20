"""
Model Loader Module
===================
Handles loading and caching the trained Random Forest classifier,
feature definitions, and class mapping.
"""

from pathlib import Path
from typing import Optional, Tuple
import joblib

# Try importing streamlit for caching; fallback to standard function if running standalone
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# The 16 target classes in the exact alphabetical order used by LabelEncoder
CLASS_NAMES = [
    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Grape_Fresh",
    "Grape_Rotten",
    "Guava_Fresh",
    "Guava_Rotten",
    "Jujube_Fresh",
    "Jujube_Rotten",
    "Orange_Fresh",
    "Orange_Rotten",
    "Pomegranate_Fresh",
    "Pomegranate_Rotten",
    "Strawberry_Fresh",
    "Strawberry_Rotten",
]

# The 8 fruit categories
FRUIT_TYPES = [
    "Apple",
    "Banana",
    "Grape",
    "Guava",
    "Jujube",
    "Orange",
    "Pomegranate",
    "Strawberry",
]

# The 2 quality categories
QUALITY_TYPES = ["Fresh", "Rotten"]

# The 7 input features expected by the model in exact order
FEATURE_COLUMNS = [
    "mean_hue",
    "mean_saturation",
    "mean_brightness",
    "contrast",
    "homogeneity",
    "energy",
    "correlation",
]

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_fruit_quality_model.pkl"


def _load_model_raw(model_path: Path):
    """Internal function to load model from disk."""
    if not model_path.exists():
        # Fallback to local models directory
        alt_path = Path("models/best_fruit_quality_model.pkl")
        if alt_path.exists():
            model_path = alt_path
        else:
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Please verify that 'best_fruit_quality_model.pkl' exists in the 'models' directory."
            )
    return joblib.load(str(model_path))


if HAS_STREAMLIT:
    @st.cache_resource(show_spinner=False)
    def load_trained_model(model_path: Optional[Path] = None):
        """Loads and caches the Random Forest model using Streamlit's cache."""
        path = model_path or DEFAULT_MODEL_PATH
        return _load_model_raw(path)
else:
    def load_trained_model(model_path: Optional[Path] = None):
        """Loads the Random Forest model without Streamlit caching."""
        path = model_path or DEFAULT_MODEL_PATH
        return _load_model_raw(path)


def parse_class_label(class_identifier) -> Tuple[str, str, str]:
    """
    Parses a class identifier (integer index or class string) into:
    (full_class_name, fruit, quality)

    Example:
        parse_class_label(0) -> ('Apple_Fresh', 'Apple', 'Fresh')
        parse_class_label('Strawberry_Rotten') -> ('Strawberry_Rotten', 'Strawberry', 'Rotten')
    """
    if isinstance(class_identifier, (int, float)):
        idx = int(class_identifier)
        if 0 <= idx < len(CLASS_NAMES):
            class_name = CLASS_NAMES[idx]
        else:
            raise IndexError(f"Class index {idx} out of range (0-{len(CLASS_NAMES)-1})")
    elif isinstance(class_identifier, str):
        class_name = class_identifier
    else:
        raise TypeError(f"Unsupported class identifier type: {type(class_identifier)}")

    if "_" in class_name:
        parts = class_name.split("_", 1)
        fruit, quality = parts[0], parts[1]
    else:
        fruit, quality = class_name, "Unknown"

    return class_name, fruit, quality
