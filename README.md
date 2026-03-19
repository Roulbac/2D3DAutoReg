# 2D3DAutoReg - Phase 1 Web Split

This repository now includes a Phase 1 modernization setup:
- `frontend/`: React UI with pose sliders + "Generate DRR" button.
- `backend/`: FastAPI stub that returns fixed placeholder DRR images.
- `docker-compose.yml`: Runs both services.

Legacy desktop code remains in the root Python modules.

## Run with Docker Compose

```bash
docker compose up --build
```

Then open:
- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend health: [http://localhost:8000/health](http://localhost:8000/health)

## API Contract (Phase 1)

`POST /api/drr/generate`

Request:

```json
{
  "pose": {
    "tx": 0,
    "ty": 0,
    "tz": 0,
    "rx": 0,
    "ry": 0,
    "rz": 0
  }
}
```

Response: deterministic placeholder DRR images for 4 views.
