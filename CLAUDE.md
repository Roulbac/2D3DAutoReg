# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker (recommended for first run)
```bash
docker compose up --build   # Start both frontend and backend
docker compose down         # Stop all services
```

### Backend (FastAPI + uv)
```bash
cd backend
uv sync                     # Install/update dependencies into .venv
KMP_DUPLICATE_LIB_OK=TRUE NIFTI_PATH=../sample_data/HN_P001.nii.gz uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- `KMP_DUPLICATE_LIB_OK=TRUE` is needed on macOS when conda and pip torch coexist (duplicate OpenMP libs)
- `NIFTI_PATH` defaults to `sample_data/HN_P001.nii.gz` (relative to repo root)
- Dependencies managed with `uv` via `pyproject.toml` + `uv.lock` (minimum version requirements, not pinned)

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev       # Dev server on port 3000 (HMR enabled)
npm run build     # Production build
```

### Makefile shortcuts
```bash
make install      # Set up both backend and frontend
make dev          # Run backend + frontend concurrently
```

## Architecture

2D/3D image registration workbench: upload a CT volume (NIfTI), render Digitally Reconstructed Radiographs (DRRs) at arbitrary 6-DOF poses, and run automatic pose registration against a target X-ray.

### Backend (`backend/app/`)

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI app with all endpoints, lifespan volume loading, SSE streaming for registration |
| `drr_engine.py` | PyTorch cone-beam DRR renderer: volume loading (SimpleITK), ray generation, AABB intersection, trilinear `grid_sample`, Beer-Lambert accumulation |
| `registration.py` | Background scipy Powell optimizer with SSE progress streaming, early convergence detection (patience-based stale metric check) |
| `metrics.py` | Image similarity metrics: NCC, gradient correlation, mean reciprocal squared difference, mutual information |

**Global state:** singleton `DRREngine` instance and optional `RegistrationRunner` thread, initialized in FastAPI lifespan.

**Device selection:** CUDA → MPS → CPU (auto-detected at startup). PyTorch ≥2.10 required for native MPS `grid_sample` support.

**Key API endpoints:**
- `POST /api/drr/generate` — render DRR at pose, returns base64 PNG
- `POST /api/registration/start` — SSE stream of optimization progress
- `POST /api/volume/upload` — replace CT volume at runtime
- `GET/PUT /api/intrinsics` — camera K matrix
- `GET /api/scene` — volume extent + camera geometry for 3D widget

### Frontend (`frontend/src/`)

Single-page React app in `App.jsx`. No external state manager — all state via hooks.

**Key components (all in App.jsx):**
- `FrameIllustration` — Three.js 3D scene showing CT volume wireframe, camera frustum, and posed camera axes
- Pose panel — 6-DOF controls (Tx/Ty/Tz mm, Rx/Ry/Rz degrees) with presets (AP, LAT, PA)
- Results panel — DRR display with target overlay and opacity slider
- Registration panel — target upload, metric selector, start/cancel with live SSE progress

**Coordinate systems:**
- Backend uses LPS (Left-Posterior-Superior): X=right, Y=anterior, Z=superior
- Three.js: X=right, Y=up, Z=toward viewer
- Mapping: `scene_x = world_x, scene_y = world_z, scene_z = -world_y`

**Rotation convention:** R = Rz · Ry · Rx (intrinsic), applied in camera-local frame then conjugated to world frame. Math is duplicated identically in both frontend JS (`applyPose`) and backend Python (`_apply_pose`).

### Data

- `sample_data/HN_P001.nii.gz` — Head/neck CT volume (26 MB NIfTI)
- `sample_data/target_tx10_ry5.png` — Test target X-ray rendered at tx=10mm, ry=5°

## Key Technical Details

- DRR rendering processes rays in tiles (4096 on CPU/MPS) to manage memory; CUDA processes all rays at once
- Registration streams intermediate results via SSE every N function evaluations; frontend auto-updates pose controls from the stream
- Volume can be swapped at runtime via upload endpoint; the engine reinitializes camera geometry from the new NIfTI header
- Beer-Lambert law: `intensity = exp(-∫μ(x)ds)` where μ is derived from HU values above a configurable threshold
