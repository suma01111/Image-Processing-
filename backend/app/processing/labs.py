from __future__ import annotations

import base64
import io
from typing import Any, Dict, Literal, Optional

import numpy as np
import cv2
from fastapi import UploadFile
from pydantic import BaseModel, Field

from app.services.image_io import (
    ImageMode,
    crop_center_square,
    decode_to_mode,
    encode_image_to_base64,
    ensure_grayscale,
    ensure_rgb,
    encode_mode_image,
    load_uploadfile_image,
    to_uint8,
)
from app.services.metrics import histogram_8bit, psnr


class PreprocessResult(BaseModel):
    original_b64: str
    processed_gray_b64: str
    processed_rgb_b64: str
    original_width: int
    original_height: int
    crop_side: int


def _side_from_image(img: np.ndarray) -> int:
    return int(img.shape[1])


async def do_preprocess(image: UploadFile) -> PreprocessResult:
    raw_img = load_uploadfile_image(image)
    original_height, original_width = raw_img.shape[:2]

    original_rgb = ensure_rgb(raw_img)
    gray = ensure_grayscale(raw_img)
    rgb_square = crop_center_square(original_rgb)
    gray_square = crop_center_square(gray)

    # Encode as base64 PNG
    original_b64 = encode_mode_image(original_rgb, mode="rgb")
    processed_gray_b64 = encode_mode_image(gray_square, mode="grayscale")
    processed_rgb_b64 = encode_mode_image(rgb_square, mode="rgb")

    return PreprocessResult(
        original_b64=original_b64,
        processed_gray_b64=processed_gray_b64,
        processed_rgb_b64=processed_rgb_b64,
        original_width=int(original_width),
        original_height=int(original_height),
        crop_side=int(gray_square.shape[1]),
    )


def _decode_input(image_b64: str, image_mode: ImageMode) -> np.ndarray:
    return decode_to_mode(image_b64=image_b64, mode=image_mode)


def _encode_output(img: np.ndarray, image_mode: ImageMode) -> str:
    return encode_mode_image(img, mode=image_mode)


def do_lab01(image_b64: str, image_mode: ImageMode, params: Dict[str, Any]) -> Dict[str, Any]:
    if image_mode != "grayscale":
        gray = _decode_input(image_b64, "grayscale")
    else:
        gray = _decode_input(image_b64, image_mode)

    side = _side_from_image(gray)
    res = int(params.get("spatialResolution", 400))
    bits = int(params.get("intensityBits", 4))
    res = max(32, min(res, 2048))
    bits = max(1, min(bits, 8))

    # Downsample to spatial resolution then quantize intensities.
    small = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA)
    levels = int(2**bits)
    q = np.round(small.astype(np.float64) / 255.0 * (levels - 1)).astype(np.uint8)

    # Reconstruct to original size and intensity range, not [0,levels-1].
    recon_q = cv2.resize(q, (side, side), interpolation=cv2.INTER_NEAREST)
    recon = np.round(recon_q.astype(np.float64) / float(levels - 1) * 255.0).astype(np.uint8)

    encoded_bits = int(res * res * bits)
    encoded_bytes = encoded_bits / 8.0

    return {
        "image_mode": "grayscale",
        "image_b64": _encode_output(recon, "grayscale"),
        "encoded": {
            "spatialResolution": res,
            "intensityBits": bits,
            "levels": levels,
            "encodedBits": encoded_bits,
            "encodedBytes": encoded_bytes,
        },
    }


def do_lab02(image_b64: str, image_mode: ImageMode, params: Dict[str, Any]) -> Dict[str, Any]:
    gray = _decode_input(image_b64, "grayscale")  # lab 02 is intensity transform
    side = _side_from_image(gray)

    sx = float(params.get("scaleX", 1.0))
    sy = float(params.get("scaleY", 1.0))
    angle_deg = float(params.get("rotationDeg", 0.0))
    tx = float(params.get("translateX", 0.0))
    ty = float(params.get("translateY", 0.0))
    shx = float(params.get("shearX", 0.0))
    shy = float(params.get("shearY", 0.0))

    cx = (side - 1) / 2.0
    cy = (side - 1) / 2.0

    angle = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64)
    S = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)
    Sh = np.array([[1.0, shx], [shy, 1.0]], dtype=np.float64)

    linear = R @ Sh @ S

    T_to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    T_back = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    T_trans = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)

    H_fwd = T_trans @ T_back @ np.block([[linear, np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]])
    # Simpler: apply center translation around linear transform.
    H_fwd = T_trans @ T_back @ np.block(
        [
            [linear, np.zeros((2, 1))],
            [np.zeros((1, 2)), np.ones((1, 1))],
        ]
    ) @ T_to_origin

    H_inv = np.linalg.inv(H_fwd)
    M_inv = H_inv[:2, :].astype(np.float64)

    warped = cv2.warpAffine(
        gray,
        M_inv,
        (side, side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return {"image_mode": "grayscale", "image_b64": _encode_output(warped, "grayscale")}


def do_lab03(image_b64: str, image_mode: ImageMode, params: Dict[str, Any]) -> Dict[str, Any]:
    gray = _decode_input(image_b64, "grayscale")
    hist_before = histogram_8bit(gray).tolist()

    neg = bool(params.get("negative", False))
    do_log = bool(params.get("log", False))
    gamma = float(params.get("gamma", 1.0))
    do_contrast = bool(params.get("contrastStretch", True))

    x = gray.astype(np.float64) / 255.0
    if neg:
        x = 1.0 - x
    if do_log:
        # Normalized log transform (outputs ~[0,1])
        x = np.log1p(x * 255.0) / np.log1p(255.0)
    if gamma != 1.0:
        x = np.power(np.clip(x, 0.0, 1.0), gamma)
    if do_contrast:
        x_min = float(x.min())
        x_max = float(x.max())
        if x_max - x_min > 1e-8:
            x = (x - x_min) / (x_max - x_min)
    out = np.clip(x * 255.0, 0, 255).astype(np.uint8)

    hist_after = histogram_8bit(out).tolist()

    return {
        "image_mode": "grayscale",
        "image_b64": _encode_output(out, "grayscale"),
        "histogram": {"before": hist_before, "after": hist_after},
    }


def _center_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.indices((h, w))
    parity = (yy + xx) % 2
    return 1.0 - 2.0 * parity  # even->+1, odd->-1


def do_lab04(image_b64: str, image_mode: ImageMode, params: Dict[str, Any]) -> Dict[str, Any]:
    gray = _decode_input(image_b64, "grayscale")
    side = _side_from_image(gray)

    D0 = float(params.get("cutoffRadius", 50.0))
    D0 = max(1.0, min(D0, float(side) / 2.0))

    h, w = gray.shape[:2]
    x = gray.astype(np.float64)

    mask = _center_mask(h, w)
    x_centered = x * mask

    F = np.fft.fft2(x_centered)
    mag = np.abs(F)
    mag_log = np.log1p(mag)

    # Low-pass: circular mask in frequency domain (centered).
    uu, vv = np.indices((h, w))
    du = uu - (h / 2.0)
    dv = vv - (w / 2.0)
    D = np.sqrt(du * du + dv * dv)
    H = (D <= D0).astype(np.float64)

    F_filt = F * H
    mag_filt = np.abs(F_filt)
    mag_log_filt = np.log1p(mag_filt)

    # Inverse FFT then un-center.
    inv = np.fft.ifft2(F_filt)
    inv_un = np.real(inv) * mask

    # Reduce near-black rendering for low cutoff settings by full-contrast normalization.
    out = cv2.normalize(inv_un, None, 0, 255, cv2.NORM_MINMAX)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Downsample spectrum for display (Plotly heatmap).
    display_size = int(params.get("spectrumDisplaySize", 256))
    display_size = max(64, min(display_size, side))
    mag_disp = cv2.resize(mag_log, (display_size, display_size), interpolation=cv2.INTER_AREA)
    mag_disp_filt = cv2.resize(mag_log_filt, (display_size, display_size), interpolation=cv2.INTER_AREA)

    mag_disp = mag_disp.tolist()
    mag_disp_filt = mag_disp_filt.tolist()

    return {
        "image_mode": "grayscale",
        "image_b64": _encode_output(out, "grayscale"),
        "spectrum": {
            "magnitudeBefore": mag_disp,
            "magnitudeAfter": mag_disp_filt,
            "cutoffRadius": D0,
            "displaySize": display_size,
        },
    }


def _add_noise(img: np.ndarray, noise_type: str, noise_amount: float, salt_amount: float) -> np.ndarray:
    x = img.astype(np.float64)
    if noise_type == "gaussian":
        sigma = noise_amount
        noise = np.random.normal(0.0, sigma, size=x.shape)
        out = x + noise
    elif noise_type == "salt_pepper":
        # salt_amount in [0,1] (total impulse fraction)
        amt = float(salt_amount)
        amt = max(0.0, min(amt, 1.0))
        out = x.copy()
        rnd = np.random.rand(*x.shape)
        salt = rnd < (amt / 2.0)
        pepper = rnd > 1.0 - (amt / 2.0)
        out[salt] = 255.0
        out[pepper] = 0.0
    elif noise_type == "uniform":
        a = float(noise_amount)
        noise = np.random.uniform(-a, a, size=x.shape)
        out = x + noise
    else:
        out = x
    return np.clip(out, 0, 255).astype(np.uint8)


def do_lab05(image_b64: str, image_mode: ImageMode, params: Dict[str, Any]) -> Dict[str, Any]:
    gray = _decode_input(image_b64, "grayscale")
    original = gray
    side = _side_from_image(gray)

    noise_type = str(params.get("noiseType", "gaussian"))
    noise_amount = float(params.get("noiseAmount", 10.0))
    salt_amount = float(params.get("saltPepperAmount", 0.05))

    filter_type = str(params.get("filterType", "median"))

    noisy = _add_noise(original, noise_type, noise_amount, salt_amount)

    if filter_type == "median":
        k = int(params.get("medianKernel", 3))
        k = k if k % 2 == 1 else k + 1
        k = max(1, min(k, 31))
        den = cv2.medianBlur(noisy, k)
    elif filter_type == "gaussian":
        k = int(params.get("gaussianKernel", 5))
        k = k if k % 2 == 1 else k + 1
        k = max(1, min(k, 31))
        sigma = float(params.get("gaussianSigma", 1.0))
        den = cv2.GaussianBlur(noisy, (k, k), sigmaX=sigma, sigmaY=sigma)
    elif filter_type == "average":
        k = int(params.get("avgKernel", 3))
        k = k if k % 2 == 1 else k + 1
        k = max(1, min(k, 31))
        den = cv2.blur(noisy, (k, k))
    elif filter_type == "freq_lowpass":
        cutoff = float(params.get("cutoffRadius", 50.0))
        cutoff = max(1.0, min(cutoff, side / 2.0))
        # Reuse lab04 logic but output only image.
        h, w = noisy.shape[:2]
        x = noisy.astype(np.float64)
        mask = _center_mask(h, w)
        x_centered = x * mask
        F = np.fft.fft2(x_centered)
        uu, vv = np.indices((h, w))
        du = uu - (h / 2.0)
        dv = vv - (w / 2.0)
        D = np.sqrt(du * du + dv * dv)
        H = (D <= cutoff).astype(np.float64)
        F_filt = F * H
        inv = np.fft.ifft2(F_filt)
        inv_un = np.real(inv) * mask
        den = np.clip(inv_un, 0, 255).astype(np.uint8)
    else:
        den = noisy

    psnr_noisy = psnr(original, noisy)
    psnr_denoised = psnr(original, den)

    return {
        "image_mode": "grayscale",
        "image_b64": _encode_output(den, "grayscale"),
        "metrics": {"psnrNoisy": psnr_noisy, "psnrDenoised": psnr_denoised},
    }


def _palette_from_anchors(k: int) -> np.ndarray:
    # Simple 4-anchor gradient: blue -> cyan -> yellow -> red.
    anchors = np.array([[0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0]], dtype=np.float64)
    t = np.linspace(0, 1, 256)
    # Piecewise linear interpolation for each channel.
    x_anchors = np.linspace(0, 1, anchors.shape[0])
    r = np.interp(t, x_anchors, anchors[:, 0])
    g = np.interp(t, x_anchors, anchors[:, 1])
    b = np.interp(t, x_anchors, anchors[:, 2])
    cmap = np.stack([r, g, b], axis=1).astype(np.uint8)
    if k <= 1:
        return cmap[:1]
    # Sample k evenly from 256.
    idx = np.linspace(0, 255, k).round().astype(int)
    return cmap[idx]


def _continuous_palette_map(gray: np.ndarray) -> np.ndarray:
    # Map intensity to the 256-entry gradient palette.
    gray_u = gray.astype(np.uint8)
    cmap = _palette_from_anchors(256)
    out = cmap[gray_u]  # (H,W,3)
    return out.astype(np.uint8)


def do_lab06(image_b64: str, image_mode: ImageMode, params: Dict[str, Any]) -> Dict[str, Any]:
    # Output is RGB.
    if image_mode == "rgb":
        rgb = _decode_input(image_b64, "rgb")  # RGB uint8 expected
    else:
        gray = _decode_input(image_b64, "grayscale")
        rgb = ensure_rgb(gray)

    gray_for_mapping = ensure_grayscale(rgb)
    side = _side_from_image(gray_for_mapping)

    mode = str(params.get("colorMode", "intensity_slicing"))
    out_rgb: np.ndarray

    if mode == "intensity_slicing":
        slices = int(params.get("slices", 4))
        slices = max(2, min(slices, 12))
        palette = _palette_from_anchors(slices)  # (slices,3)
        idx = np.floor(gray_for_mapping.astype(np.float64) / 256.0 * slices).astype(int)
        idx = np.clip(idx, 0, slices - 1)
        out_rgb = palette[idx]
    elif mode == "continuous_mapping":
        out_rgb = _continuous_palette_map(gray_for_mapping)
    elif mode == "rgb_sinusoidal":
        freq = float(params.get("sinFrequency", 3.0))
        phase_r = float(params.get("phaseR", 0.0))
        phase_g = float(params.get("phaseG", 2.0))
        phase_b = float(params.get("phaseB", 4.0))
        x = gray_for_mapping.astype(np.float64) / 255.0
        # Use 2*pi*f for frequency in cycles over [0,1].
        w = 2.0 * np.pi * freq * x
        r = 127.5 * (1.0 + np.sin(w + phase_r))
        g = 127.5 * (1.0 + np.sin(w + phase_g))
        b = 127.5 * (1.0 + np.sin(w + phase_b))
        out_rgb = np.stack([r, g, b], axis=-1)
        out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
    elif mode == "false_color":
        # Synthesize 3 bands from the same grayscale image (educational/composite).
        x = gray_for_mapping.astype(np.float64)
        x_norm = x / 255.0
        band_r = x_norm
        band_g = np.log1p(255.0 * x_norm) / np.log1p(255.0)
        gamma = float(params.get("falseColorGamma", 0.8))
        band_b = np.power(np.clip(x_norm, 0, 1), gamma)
        rgb = np.stack([band_r, band_g, band_b], axis=-1) * 255.0
        out_rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        out_rgb = _continuous_palette_map(gray_for_mapping)

    # encode_mode_image expects RGB.
    return {
        "image_mode": "rgb",
        "image_b64": _encode_output(out_rgb, "rgb"),
        "colorMode": mode,
        "squareSide": side,
    }

