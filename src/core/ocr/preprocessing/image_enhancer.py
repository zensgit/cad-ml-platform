"""Basic CPU image preprocessing to improve OCR recall.

Operations:
- Load from bytes (PIL)
- Convert to grayscale (L)
- Resize with max dimension <= max_res (keep aspect ratio)
- Light denoise (MedianFilter) and sharpen (UnsharpMask)

Returns processed bytes (PNG) and numpy array if numpy is available.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple

try:
    from PIL import Image, ImageFilter  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageFilter = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


def enhance_image_for_ocr(image_bytes: bytes, max_res: int = 2048) -> Tuple[bytes, Optional["np.ndarray"]]:
    if Image is None:
        # No PIL — return original bytes, no ndarray
        return image_bytes, None

    try:
        with BytesIO(image_bytes) as bio:
            img = Image.open(bio)
            img = img.convert("L")  # grayscale
    except Exception:
        # Not an image or PIL failed — return original bytes
        return image_bytes, None

    w, h = img.size
    scale = 1.0
    max_side = max(w, h)
    if max_side > max_res:
        scale = max_res / float(max_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Light denoise and sharpen (safe defaults)
    if ImageFilter is not None:
        try:
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
        except Exception:
            pass

    out = BytesIO()
    img.save(out, format="PNG")
    processed_bytes = out.getvalue()
    arr = None
    if np is not None:
        try:
            arr = np.array(img)
        except Exception:
            arr = None
    return processed_bytes, arr
