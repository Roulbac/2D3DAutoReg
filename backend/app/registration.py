"""Registration runners: scipy Powell and PyTorch gradient-based optimisation.

Runs in a background thread and pushes progress/complete/error events to a
queue that the FastAPI WebSocket endpoint reads from.
"""

import logging
import queue
import threading

import numpy as np
import torch
from scipy.optimize import minimize

from app.drr_engine import DRREngine
from app.metrics import METRIC_REGISTRY, TORCH_METRIC_REGISTRY

logger = logging.getLogger(__name__)


class _CancelledError(Exception):
    """Raised inside the objective function when cancellation is requested."""


class _ConvergedError(Exception):
    """Raised when the metric has stopped improving."""


class RegistrationRunner:
    """Single-view 6-DOF registration against a target X-ray image."""

    def __init__(
        self,
        engine: DRREngine,
        metric_name: str,
        preset: str,
        threshold: float | None,
        initial_pose: dict,
        report_every_n: int = 5,
    ):
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric: {metric_name!r}. Available: {list(METRIC_REGISTRY)}")
        if engine._target is None:
            raise ValueError("No target image set on the engine")

        self.engine = engine
        self.metric_fn = METRIC_REGISTRY[metric_name]
        self.preset = preset
        self.threshold = threshold
        self.initial_pose = initial_pose
        self.report_every_n = report_every_n

        self.event_queue: queue.Queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._iteration = 0
        self._best_value = float("inf")
        self._best_x: np.ndarray | None = None
        self._stale_count = 0
        self._convergence_tol = 1e-5
        self._convergence_patience = 30  # stop after N evals with no meaningful improvement

    def start(self) -> None:
        """Launch the optimisation in a daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Signal cancellation."""
        self._cancel_event.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run(self) -> None:
        x0 = np.array([
            self.initial_pose["tx"], self.initial_pose["ty"], self.initial_pose["tz"],
            self.initial_pose["rx"], self.initial_pose["ry"], self.initial_pose["rz"],
        ])

        try:
            result = minimize(
                self._objective,
                x0,
                method="Powell",
                options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 200, "disp": False},
            )

            final_pose = self._x_to_pose(result.x)
            drr_img = self.engine.render(**final_pose, preset=self.preset, threshold=self.threshold)
            drr_b64 = DRREngine._encode_png_base64(drr_img)

            self.event_queue.put({
                "event": "complete",
                "data": {
                    "pose": final_pose,
                    "metric_value": float(result.fun),
                    "drr": drr_b64,
                    "iterations": self._iteration,
                    "success": bool(result.success),
                    "message": str(result.message),
                },
            })
            logger.info(
                "Registration complete: %d iterations, metric=%.6f, success=%s",
                self._iteration, result.fun, result.success,
            )

        except _ConvergedError:
            final_pose = self._x_to_pose(self._best_x)
            drr_img = self.engine.render(**final_pose, preset=self.preset, threshold=self.threshold)
            drr_b64 = DRREngine._encode_png_base64(drr_img)
            self.event_queue.put({
                "event": "complete",
                "data": {
                    "pose": final_pose,
                    "metric_value": float(self._best_value),
                    "drr": drr_b64,
                    "iterations": self._iteration,
                    "success": True,
                    "message": f"Converged: metric stable for {self._convergence_patience} evaluations",
                },
            })
            logger.info(
                "Registration converged early: %d iterations, metric=%.6f",
                self._iteration, self._best_value,
            )

        except _CancelledError:
            self.event_queue.put({
                "event": "cancelled",
                "data": {"iterations": self._iteration},
            })
            logger.info("Registration cancelled at iteration %d", self._iteration)

        except Exception as exc:
            self.event_queue.put({
                "event": "error",
                "data": {"message": str(exc)},
            })
            logger.exception("Registration failed")

        finally:
            self.event_queue.put(None)  # sentinel: stream is done

    def _objective(self, x: np.ndarray) -> float:
        if self._cancel_event.is_set():
            raise _CancelledError()

        self._iteration += 1
        pose = self._x_to_pose(x)

        drr = self.engine.render(**pose, preset=self.preset, threshold=self.threshold)
        value = float(self.metric_fn(self.engine._target, drr))

        # Convergence: stop if no meaningful improvement for N evaluations
        if value < self._best_value - self._convergence_tol:
            self._best_value = value
            self._best_x = x.copy()
            self._stale_count = 0
        else:
            self._stale_count += 1
            if self._stale_count >= self._convergence_patience:
                raise _ConvergedError()

        # Stream progress every N function evaluations
        if self._iteration % self.report_every_n == 0:
            drr_b64 = DRREngine._encode_png_base64(drr)
            self.event_queue.put({
                "event": "progress",
                "data": {
                    "iteration": self._iteration,
                    "metric_value": value,
                    "pose": pose,
                    "drr": drr_b64,
                },
            })

        return value

    @staticmethod
    def _x_to_pose(x: np.ndarray) -> dict:
        return {
            "tx": float(x[0]), "ty": float(x[1]), "tz": float(x[2]),
            "rx": float(x[3]), "ry": float(x[4]), "rz": float(x[5]),
        }


class TorchRegistrationRunner:
    """Gradient-based 6-DOF registration using PyTorch optimizers."""

    OPTIMIZERS = {
        "adamw": lambda params, lr: torch.optim.AdamW(params, lr=lr, weight_decay=0),
        "sgd": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True),
    }

    # Default learning rates per optimizer (tuned for pose-parameter scale)
    DEFAULT_LR = {"adamw": 0.5, "sgd": 0.05}

    def __init__(
        self,
        engine: DRREngine,
        metric_name: str,
        optimizer_name: str,
        preset: str,
        threshold: float | None,
        initial_pose: dict,
        lr: float | None = None,
        max_iterations: int = 200,
        report_every_n: int = 5,
    ):
        if metric_name not in TORCH_METRIC_REGISTRY:
            available = list(TORCH_METRIC_REGISTRY)
            raise ValueError(
                f"Metric {metric_name!r} is not available for PyTorch optimizers. "
                f"Available: {available}"
            )
        if optimizer_name not in self.OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {optimizer_name!r}")
        if engine._target is None:
            raise ValueError("No target image set on the engine")

        self.engine = engine
        self.metric_fn = TORCH_METRIC_REGISTRY[metric_name]
        self.optimizer_name = optimizer_name
        self.preset = preset
        self.threshold = threshold
        self.lr = lr if lr is not None else self.DEFAULT_LR[optimizer_name]
        self.max_iterations = max_iterations
        self.report_every_n = report_every_n

        self.event_queue: queue.Queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._iteration = 0
        self._best_value = float("inf")
        self._best_pose: torch.Tensor | None = None
        self._stale_count = 0
        self._convergence_tol = 1e-5
        self._convergence_patience = 30

        x0 = [
            initial_pose["tx"], initial_pose["ty"], initial_pose["tz"],
            initial_pose["rx"], initial_pose["ry"], initial_pose["rz"],
        ]
        self.pose_param = torch.tensor(
            x0, dtype=torch.float32, device=engine.device, requires_grad=True,
        )

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            target = torch.from_numpy(self.engine._target).to(self.engine.device)
            optimizer = self.OPTIMIZERS[self.optimizer_name](
                [self.pose_param], self.lr,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5, min_lr=1e-4,
            )

            for i in range(self.max_iterations):
                if self._cancel_event.is_set():
                    raise _CancelledError()

                self._iteration = i + 1
                optimizer.zero_grad()

                drr = self.engine.render_differentiable(
                    self.pose_param, self.preset, self.threshold,
                )
                loss = self.metric_fn(target, drr)
                loss.backward()

                torch.nn.utils.clip_grad_norm_([self.pose_param], max_norm=50.0)

                optimizer.step()
                value = loss.item()
                scheduler.step(value)

                # Convergence check
                if value < self._best_value - self._convergence_tol:
                    self._best_value = value
                    self._best_pose = self.pose_param.detach().clone()
                    self._stale_count = 0
                else:
                    self._stale_count += 1
                    if self._stale_count >= self._convergence_patience:
                        raise _ConvergedError()

                # Stream progress
                if self._iteration % self.report_every_n == 0:
                    with torch.no_grad():
                        drr_np = drr.detach().cpu().numpy()
                    drr_b64 = DRREngine._encode_png_base64(drr_np)
                    pose_dict = self._pose_to_dict(self.pose_param)
                    self.event_queue.put({
                        "event": "progress",
                        "data": {
                            "iteration": self._iteration,
                            "metric_value": value,
                            "pose": pose_dict,
                            "drr": drr_b64,
                        },
                    })

            # Completed all iterations
            final_pose = self._best_pose if self._best_pose is not None else self.pose_param.detach()
            self._emit_complete(final_pose, True, "Max iterations reached")

        except _ConvergedError:
            self._emit_complete(
                self._best_pose, True,
                f"Converged: metric stable for {self._convergence_patience} iterations",
            )
            logger.info(
                "PyTorch registration converged: %d iterations, metric=%.6f",
                self._iteration, self._best_value,
            )

        except _CancelledError:
            self.event_queue.put({
                "event": "cancelled",
                "data": {"iterations": self._iteration},
            })
            logger.info("PyTorch registration cancelled at iteration %d", self._iteration)

        except Exception as exc:
            self.event_queue.put({
                "event": "error",
                "data": {"message": str(exc)},
            })
            logger.exception("PyTorch registration failed")

        finally:
            self.event_queue.put(None)  # sentinel

    def _emit_complete(self, pose_tensor: torch.Tensor, success: bool, message: str) -> None:
        pose_dict = self._pose_to_dict(pose_tensor)
        with torch.no_grad():
            drr_img = self.engine.render(**pose_dict, preset=self.preset, threshold=self.threshold)
        drr_b64 = DRREngine._encode_png_base64(drr_img)
        self.event_queue.put({
            "event": "complete",
            "data": {
                "pose": pose_dict,
                "metric_value": float(self._best_value),
                "drr": drr_b64,
                "iterations": self._iteration,
                "success": success,
                "message": message,
            },
        })

    @staticmethod
    def _pose_to_dict(pose_tensor: torch.Tensor) -> dict:
        p = pose_tensor.detach().cpu().tolist()
        return {
            "tx": p[0], "ty": p[1], "tz": p[2],
            "rx": p[3], "ry": p[4], "rz": p[5],
        }
