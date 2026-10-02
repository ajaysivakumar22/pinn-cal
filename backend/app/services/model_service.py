"""
PINN-Cal Backend — Model Service

Loads the PINN checkpoint ONCE at startup and caches it.
All subsequent inference calls reuse the cached model.

The PINN model is kept in eval() mode permanently.
Inference is wrapped in torch.no_grad().

IMPORTANT:
    PINNCalV3 is imported from pinn_cal_final.py via the adapter.
    The model architecture is NOT duplicated or reimplemented here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Import via adapter — never import directly from frozen files in routes
from backend.app.scientific.adapter import (
    SCIENTIFIC_CORE_AVAILABLE,
    SCIENTIFIC_CORE_ERROR,
    PINNCalV4,
    PINNCalV3,
    DEFAULT_CONFIG,
    DEVICE,
    M,
)
from backend.app.core.config import (
    PINN_BEST_CHECKPOINT,
    MODEL_NAME,
    MODEL_VERSION,
    MODEL_PARAMS,
    MODEL_BEST_EPOCH,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    TEST_SAMPLES,
    TIKHONOV_DELTA,
    RESIDUAL_ALPHA,
)


class ModelService:
    """
    Singleton service that owns and caches the loaded PINN model.

    Thread safety: FastAPI uses an event loop per worker. The model
    is loaded once during application startup via the lifespan handler.
    """

    _instance: Optional["ModelService"] = None

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._loaded: bool = False
        self._load_error: Optional[str] = None
        self._checkpoint_path: Optional[Path] = None
        self._load_time_s: float = 0.0
        self._start_time: float = time.monotonic()

    @classmethod
    def get(cls) -> "ModelService":
        """Return the singleton, loading the model on first call."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    # -----------------------------------------------------------------------
    # Internal loader — called exactly once
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        if not SCIENTIFIC_CORE_AVAILABLE:
            self._load_error = (
                f"Scientific core unavailable: {SCIENTIFIC_CORE_ERROR}"
            )
            return

        checkpoint_path = PINN_BEST_CHECKPOINT

        if not checkpoint_path.exists():
            self._load_error = (
                f"PINN checkpoint not found at: {checkpoint_path}"
            )
            return

        try:
            t0 = time.monotonic()

            model = PINNCalV4(
                config=DEFAULT_CONFIG,
                bounded=True,
                alpha=RESIDUAL_ALPHA,
                delta=TIKHONOV_DELTA,
            ).to(DEVICE)

            state = torch.load(
                str(checkpoint_path),
                map_location=DEVICE,
                weights_only=True,
            )

            model.load_state_dict(state)
            model.eval()

            self._model = model
            self._loaded = True
            self._checkpoint_path = checkpoint_path
            self._load_time_s = time.monotonic() - t0

        except Exception as exc:
            self._load_error = f"Checkpoint load failed: {exc}"

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run PINN inference on a batch of model inputs.

        Parameters
        ----------
        x : np.ndarray
            Shape (N, 130) — float32-compatible.
            [real(64), imag(64), sin(theta), cos(theta)]

        Returns
        -------
        np.ndarray
            Shape (N, 128) — predicted correction coefficients c_hat.
            [real(C_1..C_64), imag(C_1..C_64)]

        Raises
        ------
        RuntimeError
            If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError(
                f"PINN model not loaded: {self._load_error}"
            )

        x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        out = self._model(x_tensor)
        if isinstance(out, tuple):
            c_hat = out[0]
            c_hat_np = c_hat.detach().cpu().numpy()
            return np.concatenate(
                [c_hat_np.real.astype(np.float32), c_hat_np.imag.astype(np.float32)],
                axis=-1,
            )
        else:
            return out.detach().cpu().numpy()

    @torch.no_grad()
    def predict_v4(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run PINN V4 inference returning the full decomposition:
        (c_hat, delta_c_real, x_prior).
        """
        if not self._loaded or self._model is None:
            raise RuntimeError(
                f"PINN model not loaded: {self._load_error}"
            )

        x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        c_hat, delta_c, x_prior = self._model(x_tensor)
        return (
            c_hat.detach().cpu().numpy(),
            delta_c.detach().cpu().numpy(),
            x_prior.detach().cpu().numpy(),
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def total_parameters(self) -> int:
        if self._model is not None:
            return sum(p.numel() for p in self._model.parameters())
        return MODEL_PARAMS

    @property
    def trainable_parameters(self) -> int:
        if self._model is not None:
            return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        return MODEL_PARAMS

    # -----------------------------------------------------------------------
    # Status dict (for /api/model-status endpoint)
    # -----------------------------------------------------------------------

    def status_dict(self) -> dict:
        return {
            "model_name": MODEL_NAME,
            "version": MODEL_VERSION,
            "architecture": "PINNCalV4 (PhysicsPreconditioner + BoundedResidualDenoiser)",
            "tikhonov_delta": TIKHONOV_DELTA,
            "residual_alpha": RESIDUAL_ALPHA,
            "bounded": True,
            "input_dim": 130,
            "output_dim": 128,
            "hidden_dims": [128, 64, 128],
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "checkpoint": str(PINN_BEST_CHECKPOINT),
            "checkpoint_path": str(PINN_BEST_CHECKPOINT),
            "checkpoint_available": PINN_BEST_CHECKPOINT.exists(),
            "best_epoch": MODEL_BEST_EPOCH,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "device": str(DEVICE) if DEVICE is not None else "unknown",
            "loaded": self._loaded,
            "load_error": self._load_error,
            "load_time_s": round(self._load_time_s, 3),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "pinn_trained_for": {
                "num_elements": 64,
                "frequency_hz": 28e9,
                "frequency_label": "28 GHz",
                "note": (
                    "PINN-Cal V4 calibration requires exactly 64 elements at 28 GHz. "
                    "Other configurations run physics simulation only."
                ),
            },
        }
