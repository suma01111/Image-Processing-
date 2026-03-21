import base64
import io
from typing import Literal, Tuple

import numpy as np
import cv2
from fastapi import UploadFile
from PIL import Image

ImageMode = Literal["grayscale", "rgb"]


def _strip_data_url_prefix(b64: str) -> str:
    if "," in b64 and b64.strip().startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


def decode_base64_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(_strip_data_url_prefix(image_b64))
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image from base64")
    return img


def encode_image_to_base64(img: np.ndarray, *, ext: str = ".png") -> str:
    if img.ndim == 2:
        # grayscale
        ok, buf = cv2.imencode(ext, img)
    else:
        # OpenCV uses BGR ordering
        ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("Could not encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def load_uploadfile_image(file: UploadFile) -> np.ndarray:
    raw = file.file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode uploaded image")
    return img


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # Convert BGR->RGB for frontend usage consistency
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_center_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side].copy()


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def decode_to_mode(image_b64: str, mode: ImageMode) -> np.ndarray:
    img = decode_base64_image(image_b64)
    if mode == "grayscale":
        return ensure_grayscale(img)
    if mode == "rgb":
        # backend stores RGB for display; later processing expects uint8.
        return ensure_rgb(img)
    raise ValueError(f"Unknown image mode: {mode}")


def encode_mode_image(img: np.ndarray, mode: ImageMode) -> str:
    if mode == "grayscale":
        if img.ndim == 3:
            img = ensure_grayscale(img)
        return encode_image_to_base64(img)
    if mode == "rgb":
        # encode_image_to_base64 expects OpenCV BGR for correct colors, so convert RGB->BGR.
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError("Expected RGB image")
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return encode_image_to_base64(bgr)
    raise ValueError(f"Unknown image mode: {mode}")

