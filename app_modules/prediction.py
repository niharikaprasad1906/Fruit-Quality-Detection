"""
Prediction Module
=================
Executes inference using the trained Random Forest classifier, maps outputs
to fruit type and quality class, and computes probability rankings.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .model_loader import (
    FEATURE_COLUMNS,
    CLASS_NAMES,
    parse_class_label,
)


def predict_fruit_quality(
    model,
    feature_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Runs classification inference on the extracted 7 features.

    Parameters:
        model: Trained scikit-learn RandomForestClassifier
        feature_df: DataFrame with exactly the 7 feature columns

    Returns:
        dict containing:
            - 'fruit': str (e.g. 'Apple')
            - 'quality': str ('Fresh' or 'Rotten')
            - 'class_label': str (e.g. 'Apple_Fresh')
            - 'class_index': int
            - 'probability_score': float (0.0 to 1.0)
            - 'probability_percent': float (0.0 to 100.0)
            - 'status_color': str hex color
            - 'status_bg': str hex background
            - 'status_icon': str emoji
            - 'all_probabilities': list of dicts for all 16 classes sorted descending
            - 'top_classes': list of top-3 predictions
    """
    # 1. Validation
    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError("feature_df must be a pandas DataFrame")

    missing_cols = [c for c in FEATURE_COLUMNS if c not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    # Ensure exact column ordering
    input_data = feature_df[FEATURE_COLUMNS]

    # Verify no NaN or infinite values
    if input_data.isna().any().any():
        raise ValueError("Input features contain NaN values.")
    if not np.isfinite(input_data.values).all():
        raise ValueError("Input features contain infinite values.")

    # 2. Model Prediction
    raw_pred = model.predict(input_data)[0]
    raw_proba = model.predict_proba(input_data)[0]

    model_classes = list(model.classes_)
    if raw_pred in model_classes:
        predicted_position = model_classes.index(raw_pred)
    else:
        raise ValueError(
            f"Model returned unrecognized class {raw_pred!r}; "
            f"expected one of {model_classes}"
        )

    if isinstance(raw_pred, (int, np.integer)):
        class_label, fruit, quality = parse_class_label(int(raw_pred))
    else:
        class_label, fruit, quality = parse_class_label(str(raw_pred))

    prob_score = float(raw_proba[predicted_position])

    # 3. Probability Breakdown for all 16 classes
    all_probs: List[Dict[str, Any]] = []
    for position, prob in enumerate(raw_proba):
        model_class = model_classes[position]
        if isinstance(model_class, (int, np.integer)):
            model_class_index = int(model_class)
            c_label, c_fruit, c_qual = parse_class_label(model_class_index)
        else:
            c_label, c_fruit, c_qual = parse_class_label(str(model_class))
            model_class_index = model_classes.index(model_class)
        all_probs.append({
            "class_index": model_class_index,
            "class_label": c_label,
            "fruit": c_fruit,
            "quality": c_qual,
            "probability": float(prob),
            "probability_percent": round(float(prob) * 100.0, 2)
        })

    # Sort descending by probability
    all_probs.sort(key=lambda x: x["probability"], reverse=True)

    # 4. Status Styling
    is_fresh = (quality == "Fresh")
    status_color = "#2E7D32" if is_fresh else "#C62828"
    status_bg = "#E8F5E9" if is_fresh else "#FFEBEE"
    status_icon = "🌿" if is_fresh else "⚠️"
    status_badge = "FRESH" if is_fresh else "ROTTEN"

    return {
        "fruit": fruit,
        "quality": quality,
        "class_label": class_label,
        "class_index": int(raw_pred) if isinstance(raw_pred, (int, np.integer)) else predicted_position,
        "probability_score": prob_score,
        "probability_percent": round(prob_score * 100.0, 2),
        "status_color": status_color,
        "status_bg": status_bg,
        "status_icon": status_icon,
        "status_badge": status_badge,
        "all_probabilities": all_probs,
        "top_classes": all_probs[:3],
    }
