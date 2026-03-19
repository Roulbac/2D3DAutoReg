import base64
from io import BytesIO

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw


class PoseParams(BaseModel):
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float


class DrrRequest(BaseModel):
    pose: PoseParams


def build_stub_drr(size: int = 512) -> str:
    """Generate a deterministic placeholder DRR image and return it as base64 PNG."""
    img = Image.new("L", (size, size), color=20)
    draw = ImageDraw.Draw(img)

    # Synthetic structure so frontend wiring can be validated visually.
    draw.ellipse((80, 80, size - 80, size - 80), outline=210, width=6)
    draw.rectangle((150, 150, size - 150, size - 150), outline=150, width=4)
    draw.line((0, 0, size, size), fill=110, width=3)
    draw.line((0, size, size, 0), fill=110, width=3)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


STUB_DRR_BASE64 = build_stub_drr()

app = FastAPI(title="DRR Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/drr/generate")
def generate_drr(payload: DrrRequest) -> dict:
    # Phase 1 behavior: ignore pose and return fixed images.
    drr_data_url = f"data:image/png;base64,{STUB_DRR_BASE64}"
    return {
        "pose": payload.pose.model_dump(),
        "drrs": [
            {"view": "Main View", "image": drr_data_url},
        ],
    }
