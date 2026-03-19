import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.drr_engine import DRREngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Global engine reference – initialised during lifespan startup
# ---------------------------------------------------------------------------
engine: DRREngine | None = None


class PoseParams(BaseModel):
    tx: float = 0
    ty: float = 0
    tz: float = 0
    rx: float = 0
    ry: float = 0
    rz: float = 0


class DrrRequest(BaseModel):
    pose: PoseParams


# ---------------------------------------------------------------------------
# Lifespan: load CT volume before accepting requests
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    nifti_path = os.environ.get("NIFTI_PATH", "Test_Data/HN_P001.nii.gz")
    logger.info("Initialising DRR engine with volume: %s", nifti_path)
    engine = DRREngine(nifti_path)
    logger.info("DRR engine ready.")
    yield
    engine = None
    logger.info("DRR engine shut down.")


app = FastAPI(title="DRR Backend", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine_loaded": engine is not None}


@app.post("/api/drr/generate")
def generate_drr(payload: DrrRequest) -> dict:
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")

    pose = payload.pose
    drr_data_url = engine.render_base64(
        tx=pose.tx, ty=pose.ty, tz=pose.tz,
        rx=pose.rx, ry=pose.ry, rz=pose.rz,
    )

    return {
        "pose": pose.model_dump(),
        "drrs": [
            {"view": "Main View", "image": drr_data_url},
        ],
    }
