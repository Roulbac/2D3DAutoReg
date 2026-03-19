# 2D3DAutoReg — DRR Workbench

Web-based 2D/3D image registration tool. Load a CT volume (NIfTI), render Digitally Reconstructed Radiographs (DRRs) at arbitrary 6-DOF camera poses, and run automatic pose registration against a target X-ray.

## Quick Start

```bash
make dev    # backend on :8000, frontend on :3000
```

## Local Development

```bash
# Backend (requires Python 3.12+, uv)
cd backend
uv sync
NIFTI_PATH=../sample_data/HN_P001.nii.gz uv run uvicorn app.main:app --reload --port 8000

# Frontend (requires Node 20+)
cd frontend
npm install && npm run dev
```

Or use `make dev` to run both concurrently.

## Features

- **DRR Rendering** — PyTorch cone-beam renderer with Beer-Lambert attenuation, configurable HU threshold
- **Camera Presets** — AP, Lateral, PA views with full 6-DOF pose control
- **3D Scene View** — Interactive Three.js visualization of CT volume and camera geometry
- **Registration** — Scipy Powell optimizer with 4 similarity metrics (NCC, gradient correlation, MRSD, MI), SSE-streamed live progress
- **Volume Upload** — Swap CT volumes at runtime via the UI

## Stack

| Layer | Tech |
|-------|------|
| Frontend | React, Three.js (@react-three/fiber), Vite |
| Backend | FastAPI, PyTorch, SimpleITK, scipy |
| Packaging | uv (backend), npm (frontend), Modal (cloud deploy) |
