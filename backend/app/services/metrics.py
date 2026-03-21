import numpy as np


def psnr(ref: np.ndarray, target: np.ndarray, *, max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.
    Works for grayscale (H,W) and color (H,W,3) images.
    """
    ref_f = ref.astype(np.float64)
    tgt_f = target.astype(np.float64)
    if ref_f.shape != tgt_f.shape:
        raise ValueError(f"Shape mismatch: {ref_f.shape} vs {tgt_f.shape}")

    mse = np.mean((ref_f - tgt_f) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((max_val**2) / mse)


def histogram_8bit(img: np.ndarray) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("histogram_8bit expects a grayscale 2D array")
    hist = np.bincount(img.flatten(), minlength=256).astype(np.float64)
    return hist

