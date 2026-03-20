# /// script
# requires-python = ">=3.12"
# dependencies = ["modal"]
# ///
"""Deploy the 2D/3D AutoReg workbench to Modal cloud.

Usage:
    uv run deploy_modal.py        # deploy to Modal (prints public URL)

Prerequisites:
    modal token new               # one-time Modal authentication
"""

import modal

app = modal.App("2d3d-autoreg")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # System deps: OpenMP for PyTorch, curl for Node.js setup script
    .apt_install("libgomp1", "curl")
    # Install Node.js 20.x LTS for frontend build
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
    )
    # Python backend dependencies via uv (locked, single source of truth)
    .uv_sync("backend")
    # Copy frontend source, build with relative URLs, then clean up node_modules
    .add_local_dir("frontend", remote_path="/app/frontend")
    .run_commands(
        "cd /app/frontend && VITE_API_BASE_URL='' npm install && npm run build",
        "rm -rf /app/frontend/node_modules",
    )
    # Copy backend source
    .add_local_dir("backend/app", remote_path="/app/backend/app")
    # Copy sample CT volume
    .add_local_dir("sample_data", remote_path="/app/sample_data")
    # Default volume path
    .env({"NIFTI_PATH": "/app/sample_data/HN_P001.nii.gz"})
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/app/backend")

    from app.main import app as fastapi_app
    from starlette.staticfiles import StaticFiles

    # Serve built frontend at root — after all API/WS routes so they take priority
    fastapi_app.mount(
        "/",
        StaticFiles(directory="/app/frontend/dist", html=True),
        name="frontend",
    )

    return fastapi_app
