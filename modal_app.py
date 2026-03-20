"""Deploy the 2D/3D AutoReg workbench to Modal cloud.

Usage:
    uvx modal serve modal_app.py        # deploy to Modal (prints public URL)

Prerequisites:
    uvx modal token new               # one-time Modal authentication
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
    # Copy frontend manifests first so dependency install stays cached when source changes.
    .add_local_file(
        "frontend/package.json", remote_path="/build/frontend/package.json", copy=True
    )
    .add_local_file(
        "frontend/package-lock.json",
        remote_path="/build/frontend/package-lock.json",
        copy=True,
    )
    .run_commands("mkdir -p /build/frontend && cd /build/frontend && npm ci")
    .add_local_dir(
        "frontend",
        remote_path="/build/frontend",
        copy=True,
        ignore=["node_modules", "dist", "package.json", "package-lock.json"],
    )
    .run_commands(
        "cd /build/frontend && npm run build && mkdir -p /app/frontend && cp -r /build/frontend/dist /app/frontend/dist && rm -rf /build/frontend",
    )
    # Copy backend runtime source last so code changes do not invalidate dependency layers.
    .add_local_dir("backend/app", remote_path="/app/backend/app", copy=True)
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
)
@modal.concurrent(max_inputs=10)
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
