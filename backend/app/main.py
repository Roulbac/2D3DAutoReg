import logging
import os
from contextlib import asynccontextmanager

import anyio

from fastapi import FastAPI, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.drr_engine import DRREngine
from app.metrics import METRIC_REGISTRY
from app.registration import RegistrationRunner
from app.session_manager import SessionManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Session manager – replaces the old global engine singleton
# ---------------------------------------------------------------------------
session_manager = SessionManager()
_default_nifti_path: str | None = None


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_engine(session_id: str) -> DRREngine:
    """Look up session and return its engine, raising appropriate HTTP errors."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.engine is None:
        raise HTTPException(status_code=503, detail="No volume loaded in this session")
    return session.engine


def _get_session_or_404(session_id: str):
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
async def _stale_session_reaper():
    """Periodically clean up sessions that lost their WebSocket without closing."""
    while True:
        await anyio.sleep(60)
        session_manager.cleanup_stale(max_idle_seconds=3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _default_nifti_path
    _default_nifti_path = os.environ.get("NIFTI_PATH", "sample_data/HN_P001.nii.gz")
    logger.info("Default NIFTI path: %s", _default_nifti_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_stale_session_reaper)
        yield
        tg.cancel_scope.cancel()
    session_manager.destroy_all()
    logger.info("All sessions destroyed on shutdown.")


app = FastAPI(title="DRR Backend", version="0.4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# WebSocket – session lifecycle + registration streaming
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = session_manager.create_session()
    session_id = session.id

    # Auto-load default volume if configured
    if _default_nifti_path and os.path.isfile(_default_nifti_path):
        try:
            session.engine = DRREngine(_default_nifti_path)
            logger.info("Session %s: auto-loaded volume from %s", session_id, _default_nifti_path)
        except Exception as exc:
            logger.warning("Session %s: failed to auto-load volume: %s", session_id, exc)

    await ws.send_json({"type": "session_start", "session_id": session_id})

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "registration_start":
                await _handle_registration_start(ws, session, data)
            elif msg_type == "registration_cancel":
                if session.runner and session.runner.is_running:
                    session.runner.cancel()
            else:
                await ws.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("Session %s: WebSocket disconnected", session_id)
    except Exception as exc:
        logger.exception("Session %s: WebSocket error: %s", session_id, exc)
    finally:
        session_manager.destroy_session(session_id)


async def _handle_registration_start(ws: WebSocket, session, data: dict):
    """Start registration and stream progress over the WebSocket.

    Runs two concurrent tasks: one streams events from the runner queue to
    the WebSocket, the other listens for incoming messages (e.g. cancel).
    """
    if session.engine is None:
        await ws.send_json({"type": "error", "message": "No volume loaded"})
        return
    if session.engine._target is None:
        await ws.send_json({"type": "error", "message": "No target image uploaded"})
        return
    if session.runner and session.runner.is_running:
        await ws.send_json({"type": "error", "message": "Registration already running"})
        return

    pose_data = data.get("pose", {})
    pose = PoseParams(**pose_data)
    metric = data.get("metric", "ncc")
    preset = data.get("preset", "AP")
    threshold = data.get("threshold")
    report_every_n = data.get("report_every_n", 5)

    try:
        runner = RegistrationRunner(
            engine=session.engine,
            metric_name=metric,
            preset=preset,
            threshold=threshold,
            initial_pose=pose.model_dump(),
            report_every_n=report_every_n,
        )
    except ValueError as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        return

    session.runner = runner
    runner.start()

    # Stream events from the runner's queue to the WebSocket,
    # while still listening for incoming messages (cancel).
    # When streaming finishes, we cancel the listener via the task group.
    async def stream_events(tg: anyio.abc.TaskGroup):
        try:
            while True:
                try:
                    with anyio.fail_after(30):
                        event = await anyio.to_thread.run_sync(runner.event_queue.get)
                except TimeoutError:
                    await ws.send_json({"type": "keepalive"})
                    continue

                if event is None:
                    break
                await ws.send_json({"type": event["event"], "data": event["data"]})
        finally:
            session.runner = None
            tg.cancel_scope.cancel()

    async def listen_messages():
        while True:
            try:
                msg = await ws.receive_json()
            except Exception:
                break
            if msg.get("type") == "registration_cancel":
                runner.cancel()

    async with anyio.create_task_group() as tg:
        tg.start_soon(stream_events, tg)
        tg.start_soon(listen_messages)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "active_sessions": session_manager.active_count,
    }


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@app.get("/api/scene")
def get_scene(session_id: str = Query(...), preset: str = Query("AP")) -> dict:
    """Return static scene geometry for the 3D widget."""
    engine = _get_engine(session_id)
    return engine.get_scene_info(preset=preset)


# ---------------------------------------------------------------------------
# DRR generation
# ---------------------------------------------------------------------------
@app.post("/api/drr/generate")
def generate_drr(payload: DrrRequest, session_id: str = Query(...)) -> dict:
    engine = _get_engine(session_id)

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


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
@app.post("/api/camera/pose")
def get_camera_pose(payload: CameraRequest, session_id: str = Query(...)) -> dict:
    """Return posed camera (source + basis) without rendering."""
    engine = _get_engine(session_id)

    pose = payload.pose
    camera = engine.get_posed_camera(
        tx=pose.tx, ty=pose.ty, tz=pose.tz,
        rx=pose.rx, ry=pose.ry, rz=pose.rz,
        preset=payload.preset,
    )
    return {"camera": camera}


@app.post("/api/camera/transform")
def get_camera_transform(payload: CameraRequest, session_id: str = Query(...)) -> dict:
    """Return the full 4x4 world-to-camera extrinsic matrix."""
    engine = _get_engine(session_id)

    pose = payload.pose
    extrinsic = engine.get_extrinsic_4x4(
        tx=pose.tx, ty=pose.ty, tz=pose.tz,
        rx=pose.rx, ry=pose.ry, rz=pose.rz,
        preset=payload.preset,
    )
    return {"extrinsic_4x4": extrinsic}


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------
@app.get("/api/intrinsics")
def get_intrinsics(session_id: str = Query(...)) -> dict:
    """Return current camera intrinsic matrix parameters."""
    engine = _get_engine(session_id)
    return engine.get_intrinsics()


@app.put("/api/intrinsics")
def set_intrinsics(payload: IntrinsicsRequest, session_id: str = Query(...)) -> dict:
    """Update camera intrinsic matrix."""
    engine = _get_engine(session_id)
    engine.set_intrinsics(fx=payload.fx, fy=payload.fy, cx=payload.cx, cy=payload.cy)
    return engine.get_intrinsics()


@app.post("/api/intrinsics/reset")
def reset_intrinsics(session_id: str = Query(...)) -> dict:
    """Reset camera intrinsics to auto-computed defaults."""
    engine = _get_engine(session_id)
    engine.reset_intrinsics()
    return engine.get_intrinsics()


# ---------------------------------------------------------------------------
# Volume upload
# ---------------------------------------------------------------------------
@app.post("/api/volume/upload")
async def upload_volume(file: UploadFile, session_id: str = Query(...)) -> dict:
    """Upload a .nii.gz CT volume (kept in memory, no disk write)."""
    session = _get_session_or_404(session_id)
    if session.runner and session.runner.is_running:
        raise HTTPException(status_code=409, detail="Cannot change volume while registration is running")

    file_bytes = await file.read()
    filename = file.filename or "upload.nii.gz"

    try:
        if session.engine is None:
            session.engine = DRREngine.from_bytes(file_bytes, filename)
        else:
            session.engine.load_volume_from_bytes(file_bytes, filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load volume: {exc}")

    return {
        "filename": filename,
        **session.engine.get_scene_info(),
    }


@app.post("/api/volume/clear")
def clear_volume(session_id: str = Query(...)) -> dict:
    """Unload the current CT volume."""
    session = _get_session_or_404(session_id)
    if session.runner and session.runner.is_running:
        raise HTTPException(status_code=409, detail="Cannot clear volume while registration is running")
    if session.engine is not None:
        session.engine.clear_volume()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Registration target
# ---------------------------------------------------------------------------
@app.post("/api/registration/target")
async def upload_target(file: UploadFile, session_id: str = Query(...)) -> dict:
    """Upload a target X-ray image for registration."""
    session = _get_session_or_404(session_id)
    engine = session.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="No volume loaded in this session")
    if session.runner and session.runner.is_running:
        raise HTTPException(status_code=409, detail="Cannot change target while registration is running")
    image_bytes = await file.read()
    info = engine.set_target(image_bytes)
    target_b64 = engine.get_target_base64()
    return {**info, "target_image": target_b64}


@app.get("/api/registration/target")
def get_target(session_id: str = Query(...)) -> dict:
    """Return current target image as base64."""
    engine = _get_engine(session_id)
    b64 = engine.get_target_base64()
    if b64 is None:
        raise HTTPException(status_code=404, detail="No target image uploaded")
    return {"target_image": b64}


@app.delete("/api/registration/target")
def delete_target(session_id: str = Query(...)) -> dict:
    """Clear the current target image."""
    session = _get_session_or_404(session_id)
    engine = session.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="No volume loaded in this session")
    if session.runner and session.runner.is_running:
        raise HTTPException(status_code=409, detail="Cannot change target while registration is running")
    engine.clear_target()
    return {"status": "cleared"}


@app.get("/api/registration/metrics")
def list_metrics() -> dict:
    """Return available similarity metrics."""
    return {"metrics": list(METRIC_REGISTRY.keys())}
