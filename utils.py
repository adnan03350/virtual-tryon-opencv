# utils.py

from rembg import remove
import cv2
import numpy as np

def remove_background(img):
    """Remove background from a BGR image array."""
    _, encoded = cv2.imencode('.png', img)
    result = remove(encoded.tobytes())  # rembg works on byte-encoded images
    decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED)

    # If alpha exists, blend over white background
    if decoded.shape[2] == 4:
        alpha = decoded[:, :, 3] / 255.0
        bg = np.ones_like(decoded[:, :, :3], dtype=np.uint8) * 255
        fg = decoded[:, :, :3]
        blended = (alpha[..., None] * fg + (1 - alpha[..., None]) * bg).astype(np.uint8)
        return blended
    else:
        return decoded
