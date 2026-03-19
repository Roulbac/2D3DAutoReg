# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker (recommended)
```bash
docker compose up --build   # Start both frontend and backend
docker compose down         # Stop all services
```

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev       # Dev server on port 3000
npm run build     # Production build
npm run preview   # Serve production build
```

## Architecture

This project is a 2D/3D image registration tool being modernized from a legacy PySide2 desktop app into a web application. The work is currently in **Phase 1**: frontend/backend separation with a stub API.

### Current Web Architecture

**Frontend** (`frontend/src/App.jsx`) — React + Three.js
- Single-page app with a 6-DOF pose control panel (Tx, Ty, Tz in mm; Rx, Ry, Rz in degrees)
- Real-time 3D frame widget (`FrameIllustration`) showing CT volume, patient model, and AP camera frustum using `@react-three/fiber`
- Rotation convention: R = Rz · Ry · Rx
- "Generate DRR" button posts to the backend and displays 4 returned DRR views
- No external state manager; all state lives in `App.jsx` via React hooks

**Backend** (`backend/app/main.py`) — FastAPI
- `POST /api/drr/generate` — accepts pose parameters, currently returns 4 placeholder 512×512 PNGs (PIL-generated) as base64 data URIs
- `GET /health` — health check
- Phase 1 stub: real DRR computation (raytracing) is not yet wired in

**API contract:**
```json
// Request
{ "tx": 0, "ty": 0, "tz": 0, "rx": 0, "ry": 0, "rz": 0 }

// Response
{ "drrs": ["data:image/png;base64,...", ...] }  // 4 views
```

### Legacy Desktop Code (root-level `.py` files)

The real DRR computation logic lives in root-level Python files — not yet integrated into the backend:
- `raybox.py` — CPU/GPU ray tracing engine (uses pycuda optionally); `kernels.cu` holds the CUDA kernels
- `camera.py` / `camera_set.py` — camera projection model with intrinsics (K) and extrinsics
- `drr_registration.py` — scipy optimization loop comparing synthetic DRRs to X-ray images
- `compute_error.py` — image similarity metrics, masking, DLT reconstruction
- `metrics.py` — NCC and other registration metrics
- `main_window.py` — legacy PySide2 GUI (reference for porting logic)
- `test_funs.py` — procedural integration tests (not pytest); expects test data in `Test_Data/Sawbones_L2L3/`

### Modernization Roadmap
Phase 1 (current): UI + stub API + Docker
Phase 2 (next): Wire real DRR computation from legacy code into the FastAPI backend

## Git Worktree Gotcha

When Claude Code works inside a **git worktree** (`.claude/worktrees/<name>`), the `.git` entry is a *file* (not a folder) containing a pointer like:

```
gitdir: /path/to/repo.git/worktrees/<name>
```

Git can resolve the metadata from this pointer, but it cannot infer the working-tree path from the Bash tool's CWD (which is outside the worktree). Plain `git` commands will fail with:

```
fatal: this operation must be run in a work tree
```

**Fix:** always prefix git commands with both environment variables:

```bash
GIT_DIR=/Users/reda/Projects/2D3DAutoReg.git/worktrees/nostalgic-poitras \
GIT_WORK_TREE=/Users/reda/Projects/2D3DAutoReg.git/main/.claude/worktrees/nostalgic-poitras \
git <command>
```

Or export them once at the top of a multi-command script:

```bash
export GIT_DIR=...  GIT_WORK_TREE=...
git status
git add ...
git commit ...
```

This is required for the entire duration of a worktree session because the Bash tool's working directory never changes to the worktree path automatically.
