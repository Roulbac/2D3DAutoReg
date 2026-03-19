# Modernization Plan (Phase 1 Focus)

## Goal
Split the legacy desktop DRR app into:
- React frontend for pose controls and DRR generation trigger.
- FastAPI backend shell with a stable API contract.

Phase 1 intentionally excludes registration/optimization and real DRR compute migration.

## Scope for Phase 1
1. Introduce repository layout for service separation.
2. Implement frontend controls for `Tx, Ty, Tz, Rx, Ry, Rz` and `Generate DRR` action.
3. Implement backend API stub that accepts pose parameters and returns fixed placeholder DRR images.
4. Add Docker Compose for one-command local startup.

## Deliverables
1. `/backend` FastAPI service
- `POST /api/drr/generate` accepts pose payload and returns deterministic placeholder DRR images.
- `GET /health` for liveness checks.

2. `/frontend` React app
- Six sliders for pose parameters.
- Generate button invoking backend.
- DRR preview tiles rendering response images.

3. Root orchestration
- `docker-compose.yml` starts frontend and backend.
- Frontend exposed on `http://localhost:3000`.
- Backend exposed on `http://localhost:8000`.

## Acceptance Criteria
1. `docker compose up --build` starts both containers successfully.
2. Opening frontend shows pose sliders and `Generate DRR` button.
3. Clicking button calls backend and renders DRR placeholders.
4. No dependency on legacy PySide2 GUI path for this flow.

## Out of Scope (Phase 1)
1. SciPy optimization / registration workflows.
2. Real CT loading and ray tracing in backend.
3. Authentication, persistence, job queueing, and async compute.

## Next Phase Preview
1. Backend: replace stub response with real DRR generation service layer.
2. Frontend: add multi-view controls, parameter presets, and loading states per-view.
3. Shared API contracts via OpenAPI-generated client/types.
