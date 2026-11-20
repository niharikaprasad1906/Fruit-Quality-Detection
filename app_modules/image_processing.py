"""
Image Processing & Segmentation Module
======================================
Performs background removal using U2-Net via rembg, resizing to 224x224,
and alpha-channel foreground mask extraction matching the training pipeline.
"""

from typing import Dict, Any, Union
import io
import numpy as np
from PIL import Image
from rembg import remove, new_session

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Expected image dimensions from training pipeline
IMAGE_SIZE = (224, 224)


def _init_session():
    """Initializes U2-Net session."""
    return new_session("u2net")


if HAS_STREAMLIT:
    @st.cache_resource(show_spinner=False)
    def get_u2net_session():
        """Returns cached U2-Net segmentation session."""
        return _init_session()
else:
    _GLOBAL_SESSION = None

    def get_u2net_session():
        """Returns singleton U2-Net segmentation session."""
        global _GLOBAL_SESSION
        if _GLOBAL_SESSION is None:
            _GLOBAL_SESSION = _init_session()
        return _GLOBAL_SESSION


def load_image_to_pil(image_input: Union[str, bytes, io.BytesIO, Image.Image]) -> Image.Image:
    """
    Safely converts various image input formats into an RGB PIL Image.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, (str, bytes, io.BytesIO)):
        try:
            img = Image.open(image_input)
            img.load()  # Verify image can be decoded
            return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open or decode image: {e}")
    elif hasattr(image_input, "read"):
        try:
            # File-like object (e.g., Streamlit UploadedFile)
            image_input.seek(0)
            img = Image.open(image_input)
            img.load()
            return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to read image from file stream: {e}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def segment_fruit_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image],
    session=None
) -> Dict[str, Any]:
    """
    Performs U2-Net segmentation, resizes to 224x224, and computes foreground mask.

    Returns:
        dict containing:
            - 'original_pil': Original input image as RGB PIL Image
            - 'segmented_rgba': 224x224 RGBA PIL Image with transparent background
            - 'preview_white_bg': 224x224 RGB PIL Image on white background for UI display
            - 'mask_pil': 224x224 Grayscale PIL Image of the binary mask
            - 'rgb_image': (224, 224, 3) uint8 numpy array
            - 'alpha': (224, 224) uint8 numpy array
            - 'mask': (224, 224) boolean numpy array of foreground pixels
            - 'foreground_pixels': int count of valid fruit pixels
            - 'foreground_ratio': float percentage (0.0 to 100.0)
    """
    original_pil = load_image_to_pil(image_input)

    # Use provided or cached session
    active_session = session if session is not None else get_u2net_session()

    # Apply U2-Net segmentation
    segmented_rgba = remove(original_pil, session=active_session).convert("RGBA")

    # Resize to exact training resolution (224, 224) using LANCZOS filter
    segmented_resized = segmented_rgba.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    segmented_array = np.array(segmented_resized)

    rgb_image = segmented_array[:, :, :3]
    alpha = segmented_array[:, :, 3]

    # Foreground thresholding logic matching extract_features.py exactly:
    mask = alpha >= 128
    if np.count_nonzero(mask) < 100:
        mask = alpha >= 32
    if np.count_nonzero(mask) < 100:
        mask = alpha > 0
    if np.count_nonzero(mask) < 100:
        raise ValueError(
            "Segmentation failed: Not enough foreground fruit pixels were detected. "
            "Please ensure the image contains a clear, unobstructed fruit."
        )

    fg_count = int(np.count_nonzero(mask))
    total_pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    fg_ratio = (fg_count / total_pixels) * 100.0

    # Composite on white background for clean light-mode UI rendering
    white_bg = Image.new("RGB", IMAGE_SIZE, (255, 255, 255))
    white_bg.paste(segmented_resized, mask=segmented_resized.split()[3])

    # Binary mask visualization image
    mask_visual = Image.fromarray((mask.astype(np.uint8) * 255))

    return {
        "original_pil": original_pil,
        "segmented_rgba": segmented_resized,
        "preview_white_bg": white_bg,
        "mask_pil": mask_visual,
        "rgb_image": rgb_image,
        "alpha": alpha,
        "mask": mask,
        "foreground_pixels": fg_count,
        "foreground_ratio": fg_ratio,
    }
