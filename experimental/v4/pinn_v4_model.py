"""
PINN-Cal v4: Physics-Embedded Linear Preconditioner + Bounded Neural Residual Denoiser

Architecture:
1. Fixed Physics Preconditioner (Non-learned, deterministic linear impedance inversion):
   Input: X in R^(B, 130) -> [Re(I_obs), Im(I_obs), sin(theta), cos(theta)]
   V_eff_prior = I_obs @ Z_nom^T
   V_hw_prior  = V_eff_prior / Z_self
   V_ideal     = steering_vector(theta, CONFIG)
   C_prior     = V_ideal / (V_hw_prior + 1e-12)
   Output: X_prior in R^(B, 128) -> [Re(C_prior), Im(C_prior)]

2. Bounded Neural Residual Denoiser:
   Input: X_prior in R^(B, 128)
   Hidden: Linear(128, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 128)
   Output: Delta_C = alpha * tanh(r)  [Bounded] or r [Unbounded]

3. Final Correction:
   C_hat = C_prior + Delta_C
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from physics_core import (
    DEFAULT_CONFIG,
    ArrayConfig,
    mutual_impedance_matrix,
    steering_vector,
)

CONFIG = DEFAULT_CONFIG
M = CONFIG.num_elements  # 64

class FixedPhysicsPreconditioner(nn.Module):
    """
    Fixed, non-learned, physically exact nominal linear impedance inversion layer.
    Computes C_prior from observed currents I_obs and look direction theta.
    Uses ONLY nominal quantities available at inference time (Z_nom, Z_self, theta).
    Contains zero learned weights and zero hidden ground-truth leakage.
    Supports Tikhonov regularization delta >= 0 to suppress low-SNR division singularities.
    """
    def __init__(self, config: ArrayConfig = DEFAULT_CONFIG, delta: float = 0.0):
        super().__init__()
        self.config = config
        self.delta = float(delta)
        Z = mutual_impedance_matrix(config)
        self.register_buffer("Z_nom", Z.to(torch.complex128))
        self.register_buffer("Z_self", Z.diagonal().mean().to(torch.complex128))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, 130)
            [Re(I_obs)(64), Im(I_obs)(64), sin(theta)(1), cos(theta)(1)]

        Returns
        -------
        C_prior : torch.Tensor of shape (B, 64), complex128
        X_prior : torch.Tensor of shape (B, 128), float32 [Re(C_prior), Im(C_prior)]
        """
        # 1. Decode observed currents
        i_obs = torch.complex(
            x[:, :M].to(torch.float64),
            x[:, M:2*M].to(torch.float64),
        )

        # 2. Decode look direction theta in degrees
        sin_theta = x[:, 2*M]
        cos_theta = x[:, 2*M + 1]
        theta_deg = torch.rad2deg(torch.atan2(sin_theta, cos_theta)).to(torch.float64)

        # 3. Vectorized nominal effective voltage reconstruction: V_eff = I_obs @ Z_nom^T
        v_eff = i_obs @ self.Z_nom.T

        # 4. Recover estimated hardware excitation: V_hw = V_eff / Z_self
        v_hw = v_eff / self.Z_self

        # 5. Ideal steering excitation
        v_ideal = steering_vector(theta_deg, self.config).to(torch.complex128)

        # 6. Physical analytical calibration prior:
        if self.delta == 0.0:
            c_prior = v_ideal / (v_hw + 1e-12)
        else:
            denom = torch.abs(v_hw)**2 + (self.delta ** 2)
            c_prior = (v_ideal * torch.conj(v_hw)) / denom

        # 7. Convert to real representation for neural network input (B, 128)
        x_prior = torch.cat(
            [c_prior.real.to(torch.float32), c_prior.imag.to(torch.float32)],
            dim=-1,
        )

        return c_prior, x_prior


class BoundedResidualDenoiser(nn.Module):
    """
    Compact continuous-regression neural residual denoiser.
    Predicts Delta_C bounded by alpha * tanh(r) to prevent destabilizing the physics prior.
    Supports fixed alpha or observable SNR-aware adaptive alpha.
    """
    def __init__(
        self,
        in_features: int = 128,
        hidden_dims: tuple[int, ...] = (128, 64),
        out_features: int = 128,
        bounded: bool = True,
        alpha: float = 0.5,
        alpha_mode: str = "fixed",  # "fixed", "mild", "strong"
        alpha_bounds: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.bounded = bounded
        self.alpha_mode = alpha_mode

        if alpha_bounds is not None:
            self.alpha_bounds = alpha_bounds
            self.alpha = float(alpha_bounds[0]) if alpha_mode == "fixed" else alpha
        elif alpha_mode == "mild":
            self.alpha_bounds = (0.30, 0.80)
            self.alpha = 0.5
        elif alpha_mode == "strong":
            self.alpha_bounds = (0.25, 1.00)
            self.alpha = 0.5
        else:
            self.alpha_bounds = (alpha, alpha)
            self.alpha = alpha

        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_features))

        self.net = nn.Sequential(*layers)

        # Initialize final layer weights close to zero so initial residual is ~0
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)

    def compute_alpha(self, rho: torch.Tensor | None = None) -> torch.Tensor:
        if self.alpha_mode == "fixed" or rho is None:
            return torch.tensor(self.alpha, dtype=torch.float32)

        # Hermite cubic smoothstep transition over array spatial coherence rho in [0.80, 0.92]
        u = torch.clamp((rho - 0.80) / 0.12, 0.0, 1.0)
        w = u * u * (3.0 - 2.0 * u)
        alpha_min, alpha_max = self.alpha_bounds
        alpha = alpha_min + (alpha_max - alpha_min) * w
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(-1)  # (B, 1) for broadcasting
        return alpha

    def forward(
        self,
        x_prior: torch.Tensor,
        rho: torch.Tensor | None = None,
        return_alpha: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r = self.net(x_prior)
        if not self.bounded:
            if return_alpha:
                return r, torch.tensor(1.0, device=r.device)
            return r

        alpha = self.compute_alpha(rho)
        if isinstance(alpha, torch.Tensor) and alpha.device != r.device:
            alpha = alpha.to(r.device)

        delta_c = alpha * torch.tanh(r)
        if return_alpha:
            return delta_c, alpha
        return delta_c


class PINNCalV4(nn.Module):
    """
    PINN-Cal v4 Complete Model:
    Physics Preconditioner + Bounded Neural Residual Denoiser
    """
    def __init__(
        self,
        config: ArrayConfig = DEFAULT_CONFIG,
        bounded: bool = True,
        alpha: float = 0.5,
        alpha_mode: str = "fixed",  # "fixed", "mild", "strong"
        alpha_bounds: tuple[float, float] | None = None,
        delta: float = 0.0,
        hidden_dims: tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        self.delta = float(delta)
        self.physics_preconditioner = FixedPhysicsPreconditioner(config, delta=self.delta)
        self.residual_denoiser = BoundedResidualDenoiser(
            in_features=128,
            hidden_dims=hidden_dims,
            out_features=128,
            bounded=bounded,
            alpha=alpha,
            alpha_mode=alpha_mode,
            alpha_bounds=alpha_bounds,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_alpha: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        c_hat : torch.Tensor of shape (B, 64), complex128 (final calibration correction)
        delta_c_real : torch.Tensor of shape (B, 128), float32 (neural residual)
        x_prior : torch.Tensor of shape (B, 128), float32 (physics prior real)
        (optional) alpha : torch.Tensor of shape (B, 1) or scalar tensor
        """
        c_prior, x_prior = self.physics_preconditioner(x)

        # If adaptive alpha is active, compute observable spatial coherence proxy rho
        rho = None
        if self.residual_denoiser.alpha_mode != "fixed":
            # Reconstruct V_eff and V_ideal strictly from observable deployment quantities
            i_obs = torch.complex(
                x[:, :M].to(torch.float64),
                x[:, M:2*M].to(torch.float64),
            )
            sin_theta = x[:, 2*M]
            cos_theta = x[:, 2*M + 1]
            theta_deg = torch.rad2deg(torch.atan2(sin_theta, cos_theta)).to(torch.float64)

            v_eff = i_obs @ self.physics_preconditioner.Z_nom.T
            v_ideal = steering_vector(theta_deg, self.physics_preconditioner.config).to(torch.complex128)

            dot = torch.sum(v_eff * torch.conj(v_ideal), dim=-1)
            v_eff_norm = torch.linalg.norm(v_eff, dim=-1)
            v_ideal_norm = torch.linalg.norm(v_ideal, dim=-1)
            rho = (torch.abs(dot) / (v_eff_norm * v_ideal_norm + 1e-12)).to(torch.float32)

        if return_alpha:
            delta_c_real, eff_alpha = self.residual_denoiser(x_prior, rho=rho, return_alpha=True)
        else:
            delta_c_real = self.residual_denoiser(x_prior, rho=rho, return_alpha=False)

        # Reconstruct complex correction
        delta_c_complex = torch.complex(
            delta_c_real[:, :M].to(torch.float64),
            delta_c_real[:, M:].to(torch.float64),
        )

        c_hat = c_prior + delta_c_complex

        if return_alpha:
            return c_hat, delta_c_real, x_prior, eff_alpha
        return c_hat, delta_c_real, x_prior
