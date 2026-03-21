# Interactive Digital Image Processing Lab (Demo)

Inspired by **Gonzalez & Woods, Digital Image Processing (4th Ed.)**.

## Tech Stack

- Frontend: **React + Tailwind CSS** (with Plotly for visualizations)
- Backend: **FastAPI (Python)** + **OpenCV / NumPy**

## Project Layout

- `frontend/` - UI (module tabs, split-screen before/after, histograms/spectra, presets, download)
- `backend/` - API + image processing pipeline
  - `app/routes/` - endpoints
  - `app/services/` - image IO + metrics
  - `app/processing/` - lab algorithms

## Run Instructions

### 1) Start Backend (FastAPI)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

### 2) Start Frontend (Vite)

```bash
cd frontend
npm install
npm run dev -- --port 5175
```

Open the shown `http://localhost:5175/` URL in your browser.

The frontend reads `VITE_BACKEND_URL` from `frontend/.env` (currently set to `http://localhost:8002`).

## Modules Implemented (Current)

- **Calibration & Quantization (01)**: Sampling + intensity quantization (encoded size + reconstruction)
- **Geometric Registration (02)**: Affine warps via sliders (scale x/y, rotation, translation, shear)
- **Photometric Correction (03)**: Negative / log / gamma / contrast stretch + histogram before/after
- **Spectral Analysis (04)**: 2D FFT magnitude spectrum (centered using `(-1)^(x+y)`) + interactive low-pass
- **Image Restoration (05)**: Noise models + denoising filters + PSNR
- **Colorimetric Mapping (06)**: Intensity-to-RGB mapping (palette, continuous, RGB sinusoidal, false-color composite)

## Notes

- WebAssembly acceleration is not included in this first version (server-side processing is used for clarity and correctness).

