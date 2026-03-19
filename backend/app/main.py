import json
import logging
import os
import queue
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.drr_engine import DRREngine
from app.metrics import METRIC_REGISTRY
from app.registration import RegistrationRunner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Global engine reference – initialised during lifespan startup
# ---------------------------------------------------------------------------
engine: DRREngine | None = None
_active_runner: RegistrationRunner | None = None


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


class IntrinsicsRequest(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


class RegistrationRequest(BaseModel):
    pose: PoseParams
    preset: str = "AP"
    threshold: float | None = None
    metric: str = "ncc"
    report_every_n: int = 5


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


@app.get("/api/intrinsics")
def get_intrinsics() -> dict:
    """Return current camera intrinsic matrix parameters."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")
    return engine.get_intrinsics()


@app.put("/api/intrinsics")
def set_intrinsics(payload: IntrinsicsRequest) -> dict:
    """Update camera intrinsic matrix."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")
    engine.set_intrinsics(fx=payload.fx, fy=payload.fy, cx=payload.cx, cy=payload.cy)
    return engine.get_intrinsics()


@app.post("/api/intrinsics/reset")
def reset_intrinsics() -> dict:
    """Reset camera intrinsics to auto-computed defaults."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")
    engine.reset_intrinsics()
    return engine.get_intrinsics()


# ---------------------------------------------------------------------------
# Volume upload
# ---------------------------------------------------------------------------
@app.post("/api/volume/upload")
async def upload_volume(file: UploadFile) -> dict:
    """Upload a .nii.gz CT volume to replace the current one."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")
    if _active_runner and _active_runner.is_running:
        raise HTTPException(status_code=409, detail="Cannot change volume while registration is running")

    suffix = ".nii.gz" if file.filename and file.filename.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        engine.load_volume(tmp_path)
    except Exception as exc:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Failed to load volume: {exc}")

    return {
        "filename": file.filename,
        **engine.get_scene_info(),
    }


@app.post("/api/volume/clear")
def clear_volume() -> dict:
    """Unload the current CT volume."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")
    if _active_runner and _active_runner.is_running:
        raise HTTPException(status_code=409, detail="Cannot clear volume while registration is running")
    engine.clear_volume()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Registration endpoints
# ---------------------------------------------------------------------------
@app.post("/api/registration/target")
async def upload_target(file: UploadFile) -> dict:
    """Upload a target X-ray image for registration."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")
    if _active_runner and _active_runner.is_running:
        raise HTTPException(status_code=409, detail="Cannot change target while registration is running")
    image_bytes = await file.read()
    info = engine.set_target(image_bytes)
    target_b64 = engine.get_target_base64()
    return {**info, "target_image": target_b64}


@app.get("/api/registration/target")
def get_target() -> dict:
    """Return current target image as base64."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")
    b64 = engine.get_target_base64()
    if b64 is None:
        raise HTTPException(status_code=404, detail="No target image uploaded")
    return {"target_image": b64}


@app.delete("/api/registration/target")
def delete_target() -> dict:
    """Clear the current target image."""
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")
    if _active_runner and _active_runner.is_running:
        raise HTTPException(status_code=409, detail="Cannot change target while registration is running")
    engine.clear_target()
    return {"status": "cleared"}


@app.get("/api/registration/metrics")
def list_metrics() -> dict:
    """Return available similarity metrics."""
    return {"metrics": list(METRIC_REGISTRY.keys())}


@app.post("/api/registration/start")
def start_registration(payload: RegistrationRequest):
    """Start registration optimisation and stream progress as SSE."""
    global _active_runner
    if engine is None:
        raise HTTPException(status_code=503, detail="DRR engine not initialised")
    if engine._target is None:
        raise HTTPException(status_code=400, detail="No target image uploaded")
    if _active_runner and _active_runner.is_running:
        raise HTTPException(status_code=409, detail="Registration already running")

    runner = RegistrationRunner(
        engine=engine,
        metric_name=payload.metric,
        preset=payload.preset,
        threshold=payload.threshold,
        initial_pose=payload.pose.model_dump(),
        report_every_n=payload.report_every_n,
    )
    _active_runner = runner
    runner.start()

    def event_generator():
        while True:
            try:
                event = runner.event_queue.get(timeout=30)
            except queue.Empty:
                yield "event: keepalive\ndata: {}\n\n"
                continue
            if event is None:
                break
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/registration/cancel")
def cancel_registration() -> dict:
    """Cancel the running registration."""
    if _active_runner and _active_runner.is_running:
        _active_runner.cancel()
        return {"status": "cancelling"}
    return {"status": "not_running"}
