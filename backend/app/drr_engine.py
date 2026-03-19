"""PyTorch-based cone-beam DRR engine.

Replaces legacy Numba/CUDA ray tracing with a fully vectorised PyTorch
pipeline that auto-selects device (CUDA → MPS → CPU).
"""

import base64
import logging
import math
from io import BytesIO

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Physics constants (from legacy raybox.py)
MU_WATER = 0.037
MU_AIR = 0.00046
DEFAULT_HU_THRESHOLD = 300
DEFAULT_SID = 1000.0  # Source-to-Image Distance in mm
DEFAULT_IMAGE_SIZE = 512
NUM_SAMPLES = 256


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DRREngine:
    """Loads a CT volume and renders DRRs for arbitrary 6-DOF poses."""

    def __init__(
        self,
        nifti_path: str,
        image_size: int = DEFAULT_IMAGE_SIZE,
        sid: float = DEFAULT_SID,
        threshold: float = DEFAULT_HU_THRESHOLD,
        num_samples: int = NUM_SAMPLES,
    ):
        self.image_size = image_size
        self.sid = sid
        self.threshold = threshold
        self.num_samples = num_samples
        self.device = _select_device()
        logger.info("DRR engine using device: %s", self.device)

        self._target: np.ndarray | None = None

        self._load_volume(nifti_path)
        self._setup_camera()

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        filename: str = "upload.nii.gz",
        image_size: int = DEFAULT_IMAGE_SIZE,
        sid: float = DEFAULT_SID,
        threshold: float = DEFAULT_HU_THRESHOLD,
        num_samples: int = NUM_SAMPLES,
    ) -> "DRREngine":
        """Create a DRREngine from in-memory NIfTI bytes (no disk I/O)."""
        instance = cls.__new__(cls)
        instance.image_size = image_size
        instance.sid = sid
        instance.threshold = threshold
        instance.num_samples = num_samples
        instance.device = _select_device()
        instance._target = None
        logger.info("DRR engine using device: %s", instance.device)
        instance.load_volume_from_bytes(data, filename)
        return instance

    # ------------------------------------------------------------------
    # 1. Volume loading
    # ------------------------------------------------------------------
    def _load_volume(self, path: str) -> None:
        """Load NIfTI CT volume via SimpleITK (mirrors legacy read_rho)."""
        logger.info("Loading volume from %s …", path)
        sitk_img = sitk.ReadImage(path)
        spacing = np.array(sitk_img.GetSpacing(), dtype=np.float32)  # (sx, sy, sz)
        # SimpleITK returns (Z, Y, X) numpy array; transpose to (X, Y, Z)
        vol_np = sitk.GetArrayFromImage(sitk_img).transpose((2, 1, 0)).astype(np.float32)

        self.vol_shape = np.array(vol_np.shape, dtype=np.float32)  # (Nx, Ny, Nz)
        self.spacing = spacing  # (sx, sy, sz)
        self.vol_extent = self.vol_shape * self.spacing  # physical size in mm

        # Store as (1,1,Nx,Ny,Nz) for grid_sample – D=Nx, H=Ny, W=Nz
        self.volume = (
            torch.from_numpy(vol_np)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        logger.info(
            "Volume loaded: shape=%s, spacing=%s, extent=%s mm",
            vol_np.shape, spacing, self.vol_extent,
        )

    def load_volume(self, path: str) -> None:
        """Reload the engine with a new NIfTI volume."""
        self._load_volume(path)
        self._setup_camera()
        self._target = None

    def load_volume_from_bytes(self, data: bytes, filename: str = "upload.nii.gz") -> None:
        """Load a NIfTI volume from in-memory bytes (no disk write)."""
        import gzip

        logger.info("Loading volume from %d bytes (%s) …", len(data), filename)
        # Decompress gzip if needed
        if filename.endswith(".gz"):
            data = gzip.decompress(data)
        fh = nib.FileHolder(fileobj=BytesIO(data))
        img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
        spacing = np.array(img.header.get_zooms()[:3], dtype=np.float32)
        vol_np = np.asarray(img.dataobj, dtype=np.float32)
        self.vol_shape = np.array(vol_np.shape[:3], dtype=np.float32)
        self.spacing = spacing
        self.vol_extent = self.vol_shape * self.spacing
        self.volume = (
            torch.from_numpy(vol_np)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        self._setup_camera()
        self._target = None
        logger.info(
            "Volume loaded from bytes: shape=%s, spacing=%s, extent=%s mm",
            vol_np.shape, spacing, self.vol_extent,
        )

    def clear_volume(self) -> None:
        """Unload the current volume and reset state."""
        self.volume = None
        self.vol_shape = None
        self.spacing = None
        self.vol_extent = None
        self._target = None
        logger.info("Volume cleared")

    # ------------------------------------------------------------------
    # 2. Auto AP camera (no calibration file needed)
    # ------------------------------------------------------------------
    def _setup_camera(self) -> None:
        """Build intrinsic K and camera presets from volume extent."""
        # Volume centroid in world coordinates (origin at corner)
        self.centroid = torch.tensor(
            self.vol_extent / 2.0, dtype=torch.float32, device=self.device
        )

        # Focal length in pixels – maps SID to image size
        # Pixel size ≈ max_extent / image_size, focal_px = SID / pixel_size
        max_extent = float(self.vol_extent.max())
        pixel_size = max_extent / self.image_size
        focal_px = self.sid / pixel_size

        cx = self.image_size / 2.0
        cy = self.image_size / 2.0

        # Intrinsic matrix K (3×3)
        self.K = torch.tensor(
            [[focal_px, 0, cx],
             [0, focal_px, cy],
             [0, 0, 1]],
            dtype=torch.float32, device=self.device,
        )
        self.K_inv = torch.inverse(self.K)

        # --- Camera presets ---------------------------------------------------
        # Each preset: (default_source, default_R_w2c)

        # AP: Source on Anterior side (-Y in LPS), beam toward Posterior (+Y)
        source_ap = self.centroid.clone()
        source_ap[1] = self.centroid[1] - self.sid
        R_ap = torch.tensor(
            [[-1.0, 0.0, 0.0],
             [0.0, 0.0, -1.0],
             [0.0, -1.0, 0.0]],
            dtype=torch.float32, device=self.device,
        )

        # LAT (Lateral): Source on Left side (+X in LPS), beam toward Right (-X)
        # Row 2 (cam-Z) = opposite of beam [-1,0,0] → [+1,0,0]
        # Row 1 (cam-Y) = -Superior → [0,0,-1]
        # Row 0 (cam-X) = cam-Y × cam-Z = [0,0,-1]×[1,0,0] = [0,-1,0]
        source_lat = self.centroid.clone()
        source_lat[0] = self.centroid[0] + self.sid
        R_lat = torch.tensor(
            [[0.0, -1.0, 0.0],
             [0.0, 0.0, -1.0],
             [1.0, 0.0, 0.0]],
            dtype=torch.float32, device=self.device,
        )

        # PA: Source on Posterior side (+Y in LPS), beam toward Anterior (-Y)
        # Row 2 (cam-Z) = opposite of beam [0,-1,0] → [0,+1,0]
        # Row 1 (cam-Y) = -Superior → [0,0,-1]
        # Row 0 (cam-X) = cam-Y × cam-Z = [0,0,-1]×[0,1,0] = [1,0,0]
        source_pa = self.centroid.clone()
        source_pa[1] = self.centroid[1] + self.sid
        R_pa = torch.tensor(
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, -1.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32, device=self.device,
        )

        self.presets = {
            "AP": (source_ap, R_ap),
            "LAT": (source_lat, R_lat),
            "PA": (source_pa, R_pa),
        }

        # Backwards-compat aliases
        self.default_source = source_ap
        self.default_R = R_ap

    # ------------------------------------------------------------------
    # 3. Pose composition
    # ------------------------------------------------------------------
    @staticmethod
    def _euler_to_rotation(rx: float, ry: float, rz: float, device: torch.device) -> torch.Tensor:
        """Build rotation matrix R = Rz · Ry · Rx from angles in degrees."""
        rx = math.radians(rx)
        ry = math.radians(ry)
        rz = math.radians(rz)

        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]],
                          dtype=torch.float32, device=device)
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]],
                          dtype=torch.float32, device=device)
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]],
                          dtype=torch.float32, device=device)

        return Rz @ Ry @ Rx

    def _apply_pose(self, tx: float, ty: float, tz: float,
                    rx: float, ry: float, rz: float,
                    preset: str = "AP"):
        """Return (source_world, R_world2cam) after applying camera-relative 6-DOF pose.

        Perturbations are in the camera's own coordinate frame:
        - tx/ty/tz translate along camera X/Y/Z axes
        - rx/ry/rz rotate around camera X/Y/Z axes (orbiting the volume centroid)
        """
        default_source, default_R = self.presets[preset]

        # Camera-relative rotation → conjugate to world frame
        R_local = self._euler_to_rotation(rx, ry, rz, self.device)
        R_world = default_R.T @ R_local @ default_R

        # Orbit source around centroid using world-frame rotation
        source_rel = default_source - self.centroid
        source_rotated = R_world @ source_rel + self.centroid

        # Camera-relative translation → convert to world via posed R_c2w
        R_w2c = default_R @ R_world.T
        R_c2w = R_w2c.T
        t_cam = torch.tensor([tx, ty, tz], dtype=torch.float32, device=self.device)
        t_world = R_c2w @ t_cam

        source_world = source_rotated + t_world

        return source_world, R_w2c

    # ------------------------------------------------------------------
    # 4. Vectorised ray generation
    # ------------------------------------------------------------------
    def _generate_rays(self, source_world: torch.Tensor, R_w2c: torch.Tensor):
        """Unproject pixel grid to world-space ray directions.

        Returns
        -------
        origins : (H*W, 3)   all equal to source_world
        dirs    : (H*W, 3)   unit direction vectors in world space
        """
        H = W = self.image_size
        # Pixel coordinates grid
        v, u = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=self.device),
            torch.arange(W, dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        ones = torch.ones_like(u)
        pixels = torch.stack([u, v, ones], dim=-1).reshape(-1, 3)  # (H*W, 3)

        # Camera-space directions via K_inv
        dirs_cam = (self.K_inv @ pixels.T).T  # (H*W, 3)

        # Camera convention: scene in -Z → negate Z component
        dirs_cam[:, 2] = -dirs_cam[:, 2]

        # Rotate to world space: R_w2c maps world→cam, so cam→world = R_w2c^T
        R_c2w = R_w2c.T
        dirs_world = (R_c2w @ dirs_cam.T).T  # (H*W, 3)

        # Normalise
        dirs_world = F.normalize(dirs_world, dim=-1)

        origins = source_world.unsqueeze(0).expand(H * W, -1)
        return origins, dirs_world

    # ------------------------------------------------------------------
    # 5. AABB slab intersection
    # ------------------------------------------------------------------
    def _aabb_intersect(self, origins: torch.Tensor, dirs: torch.Tensor):
        """Vectorised AABB ray-box intersection.

        Returns t_near, t_far, valid mask.
        """
        # Box bounds: (0,0,0) to vol_extent
        box_min = torch.zeros(3, dtype=torch.float32, device=self.device)
        box_max = torch.tensor(self.vol_extent, dtype=torch.float32, device=self.device)

        inv_dir = 1.0 / (dirs + 1e-10)

        t1 = (box_min - origins) * inv_dir  # (N, 3)
        t2 = (box_max - origins) * inv_dir  # (N, 3)

        t_min = torch.minimum(t1, t2)  # (N, 3)
        t_max = torch.maximum(t1, t2)  # (N, 3)

        t_near = t_min.max(dim=-1).values  # (N,)
        t_far = t_max.min(dim=-1).values   # (N,)

        # Clamp t_near to >= 0 (don't go behind the source)
        t_near = t_near.clamp(min=0.0)

        valid = t_near < t_far

        return t_near, t_far, valid

    # ------------------------------------------------------------------
    # 6 & 7. Volume sampling + Beer-Lambert accumulation
    # ------------------------------------------------------------------
    def _sample_and_accumulate(
        self,
        origins: torch.Tensor,
        dirs: torch.Tensor,
        t_near: torch.Tensor,
        t_far: torch.Tensor,
        valid: torch.Tensor,
        threshold: float | None = None,
    ) -> torch.Tensor:
        """Sample the volume along rays and accumulate attenuation.

        Uses tiled processing to manage memory on CPU.
        """
        N = origins.shape[0]
        intensities = torch.ones(N, dtype=torch.float32, device=self.device)

        if not valid.any():
            return intensities

        # Work only with valid rays
        valid_idx = valid.nonzero(as_tuple=True)[0]
        o = origins[valid_idx]        # (Nv, 3)
        d = dirs[valid_idx]           # (Nv, 3)
        tn = t_near[valid_idx]        # (Nv,)
        tf = t_far[valid_idx]         # (Nv,)

        # Tile size for memory management (MPS also needs tiling)
        tile_size = 4096 if self.device.type in ("cpu", "mps") else len(valid_idx)
        accumulated = torch.zeros(len(valid_idx), dtype=torch.float32, device=self.device)

        for start in range(0, len(valid_idx), tile_size):
            end = min(start + tile_size, len(valid_idx))
            o_t = o[start:end]            # (T, 3)
            d_t = d[start:end]            # (T, 3)
            tn_t = tn[start:end]          # (T,)
            tf_t = tf[start:end]          # (T,)

            T = o_t.shape[0]
            S = self.num_samples

            # Parametric sample positions along each ray
            t_vals = torch.linspace(0, 1, S, device=self.device)  # (S,)
            t_samples = tn_t.unsqueeze(1) + t_vals.unsqueeze(0) * (tf_t - tn_t).unsqueeze(1)  # (T, S)

            # Step size per sample
            dt = (tf_t - tn_t) / S  # (T,)

            # 3D sample positions in world space
            pts = o_t.unsqueeze(1) + d_t.unsqueeze(1) * t_samples.unsqueeze(2)  # (T, S, 3)

            # Normalise to [-1, 1] for grid_sample
            # Volume spans [0, vol_extent] in world coords
            extent = torch.tensor(self.vol_extent, dtype=torch.float32, device=self.device)
            norm_pts = 2.0 * pts / extent - 1.0  # (T, S, 3) in [-1, 1]

            # grid_sample expects (N,C,D,H,W) volume and (N,D_out,H_out,W_out,3) grid
            # where grid coords are (norm_x_W, norm_y_H, norm_z_D)
            # Our volume is stored as (1,1,Nx,Ny,Nz) → D=Nx, H=Ny, W=Nz
            # So grid coords must be (norm_z, norm_y, norm_x) — reversed from world XYZ
            grid = torch.stack([norm_pts[..., 2], norm_pts[..., 1], norm_pts[..., 0]], dim=-1)
            # Reshape to (1, T, S, 1, 3) for 5D grid_sample
            grid = grid.unsqueeze(0).unsqueeze(3)  # (1, T, S, 1, 3)

            sampled = F.grid_sample(
                self.volume, grid,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )  # (1, 1, T, S, 1)
            hu_values = sampled.squeeze(0).squeeze(0).squeeze(-1)  # (T, S)

            # Apply HU threshold and convert to linear attenuation
            mu = (hu_values * (MU_WATER - MU_AIR) / 1000.0 + MU_WATER)
            hu_threshold = threshold if threshold is not None else self.threshold
            mu = mu * (hu_values >= hu_threshold).float()

            # Physical step length in mm
            ray_len = (tf_t - tn_t)  # (T,)
            step_mm = ray_len / S  # (T,)

            # Beer-Lambert: accumulated optical depth
            optical_depth = (mu * step_mm.unsqueeze(1)).sum(dim=1)  # (T,)
            accumulated[start:end] = optical_depth

        intensities[valid_idx] = torch.exp(-accumulated)
        return intensities

    # ------------------------------------------------------------------
    # 8. Output encoding
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_png_base64(image: np.ndarray) -> str:
        """Encode a [0,1] float32 array as base64 PNG data URI."""
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode="L")
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_available_presets(self) -> list[str]:
        """Return list of available camera preset names."""
        return list(self.presets.keys())

    def get_intrinsics(self) -> dict:
        """Return current camera intrinsics."""
        K = self.K.cpu().tolist()
        return {
            "fx": K[0][0],
            "fy": K[1][1],
            "cx": K[0][2],
            "cy": K[1][2],
            "image_size": self.image_size,
            "K": K,
        }

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """Update the intrinsic matrix K and recompute K_inv."""
        self.K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=torch.float32, device=self.device,
        )
        self.K_inv = torch.inverse(self.K)
        logger.info("Intrinsics updated: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", fx, fy, cx, cy)

    def reset_intrinsics(self) -> None:
        """Reset intrinsics to auto-computed defaults from volume extent and SID."""
        max_extent = float(self.vol_extent.max())
        pixel_size = max_extent / self.image_size
        focal_px = self.sid / pixel_size
        cx = cy = self.image_size / 2.0
        self.set_intrinsics(focal_px, focal_px, cx, cy)
        logger.info("Intrinsics reset to defaults")

    # ------------------------------------------------------------------
    # Target image for registration
    # ------------------------------------------------------------------
    def set_target(self, image_bytes: bytes) -> dict:
        """Decode image bytes, convert to grayscale float32 [0,1], resize to image_size."""
        img = Image.open(BytesIO(image_bytes)).convert("L")
        original_size = list(img.size)  # (W, H)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        self._target = np.array(img, dtype=np.float32) / 255.0
        logger.info("Target image set: original=%s, resized=%dx%d", original_size, self.image_size, self.image_size)
        return {"width": self.image_size, "height": self.image_size, "original_size": original_size}

    def get_target_base64(self) -> str | None:
        """Return current target image as base64 PNG data URI, or None."""
        if self._target is None:
            return None
        img_uint8 = (self._target * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode="L")
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def clear_target(self) -> None:
        """Remove the current target image."""
        self._target = None
        logger.info("Target image cleared")

    def get_scene_info(self, preset: str = "AP") -> dict:
        """Return static scene geometry for the 3D frame sketch."""
        default_source, default_R = self.presets[preset]
        centroid = self.centroid.cpu().tolist()
        vol_extent = self.vol_extent.tolist()
        source = default_source.cpu().tolist()
        R_c2w = default_R.T
        return {
            "volume": {
                "centroid": centroid,
                "extent": vol_extent,
            },
            "camera": {
                "source": source,
                "sid": self.sid,
                "basis": {
                    "x": R_c2w[:, 0].cpu().tolist(),
                    "y": R_c2w[:, 1].cpu().tolist(),
                    "z": R_c2w[:, 2].cpu().tolist(),
                },
            },
            "available_presets": self.get_available_presets(),
        }

    def get_posed_camera(
        self,
        tx: float = 0, ty: float = 0, tz: float = 0,
        rx: float = 0, ry: float = 0, rz: float = 0,
        preset: str = "AP",
    ) -> dict:
        """Return camera position and basis after applying a 6-DOF pose."""
        with torch.no_grad():
            source, R_w2c = self._apply_pose(tx, ty, tz, rx, ry, rz, preset)
        R_c2w = R_w2c.T
        return {
            "source": source.cpu().tolist(),
            "basis": {
                "x": R_c2w[:, 0].cpu().tolist(),
                "y": R_c2w[:, 1].cpu().tolist(),
                "z": R_c2w[:, 2].cpu().tolist(),
            },
        }

    def get_extrinsic_4x4(
        self,
        tx: float = 0, ty: float = 0, tz: float = 0,
        rx: float = 0, ry: float = 0, rz: float = 0,
        preset: str = "AP",
    ) -> list[list[float]]:
        """Return the full 4x4 world-to-camera extrinsic matrix."""
        with torch.no_grad():
            source, R_w2c = self._apply_pose(tx, ty, tz, rx, ry, rz, preset)
        t = -R_w2c @ source
        M = torch.eye(4, dtype=torch.float32, device=self.device)
        M[:3, :3] = R_w2c
        M[:3, 3] = t
        return M.cpu().tolist()

    def render(
        self,
        tx: float = 0, ty: float = 0, tz: float = 0,
        rx: float = 0, ry: float = 0, rz: float = 0,
        preset: str = "AP",
        threshold: float | None = None,
    ) -> np.ndarray:
        """Render a single DRR at the given 6-DOF pose.

        Returns a (image_size, image_size) float32 numpy array in [0, 1].
        """
        with torch.no_grad():
            source, R_w2c = self._apply_pose(tx, ty, tz, rx, ry, rz, preset)
            origins, dirs = self._generate_rays(source, R_w2c)
            t_near, t_far, valid = self._aabb_intersect(origins, dirs)
            intensities = self._sample_and_accumulate(origins, dirs, t_near, t_far, valid, threshold=threshold)

        img = intensities.cpu().numpy().reshape(self.image_size, self.image_size)

        # Invert so that dense structures appear bright (like X-ray)
        img = 1.0 - img

        # Normalise to full [0, 1] range
        vmin, vmax = img.min(), img.max()
        if vmax - vmin > 1e-8:
            img = (img - vmin) / (vmax - vmin)

        return img

    def render_base64(
        self,
        tx: float = 0, ty: float = 0, tz: float = 0,
        rx: float = 0, ry: float = 0, rz: float = 0,
        preset: str = "AP",
        threshold: float | None = None,
    ) -> str:
        """Render a DRR and return as a base64 PNG data URI."""
        img = self.render(tx, ty, tz, rx, ry, rz, preset, threshold=threshold)
        return self._encode_png_base64(img)
