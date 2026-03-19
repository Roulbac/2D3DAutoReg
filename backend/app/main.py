import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
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
    preset: str = "AP"
    threshold: float | None = None


class CameraRequest(BaseModel):
    pose: PoseParams
    preset: str = "AP"


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


app = FastAPI(title="DRR Backend", version="0.3.0", lifespan=lifespan)

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


@app.get("/api/scene")
def get_scene(preset: str = Query("AP")) -> dict:
    """Return static scene geometry (volume extent, default camera) for the 3D widget."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")
    return engine.get_scene_info(preset=preset)


@app.post("/api/drr/generate")
def generate_drr(payload: DrrRequest) -> dict:
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")

    pose = payload.pose
    preset = payload.preset
    drr_data_url = engine.render_base64(
        tx=pose.tx, ty=pose.ty, tz=pose.tz,
        rx=pose.rx, ry=pose.ry, rz=pose.rz,
        preset=preset,
        threshold=payload.threshold,
    )

    camera = engine.get_posed_camera(
        tx=pose.tx, ty=pose.ty, tz=pose.tz,
        rx=pose.rx, ry=pose.ry, rz=pose.rz,
        preset=preset,
    )

    return {
        "pose": pose.model_dump(),
        "preset": preset,
        "camera": camera,
        "drrs": [
            {"view": "Main View", "image": drr_data_url},
        ],
    }


@app.post("/api/camera/pose")
def get_camera_pose(payload: CameraRequest) -> dict:
    """Lightweight endpoint: return posed camera (source + basis) without rendering."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")

    pose = payload.pose
    camera = engine.get_posed_camera(
        tx=pose.tx, ty=pose.ty, tz=pose.tz,
        rx=pose.rx, ry=pose.ry, rz=pose.rz,
        preset=payload.preset,
    )
    return {"camera": camera}


@app.post("/api/camera/transform")
def get_camera_transform(payload: CameraRequest) -> dict:
    """Return the full 4x4 world-to-camera extrinsic matrix."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")

    pose = payload.pose
    extrinsic = engine.get_extrinsic_4x4(
        tx=pose.tx, ty=pose.ty, tz=pose.tz,
        rx=pose.rx, ry=pose.ry, rz=pose.rz,
        preset=payload.preset,
    )
    return {"extrinsic_4x4": extrinsic}
