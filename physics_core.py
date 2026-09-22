"""
PINN-Cal — Physics Core

Differentiable physical model for a 64-element mmWave antenna array.

This module will become the single source of truth for:
    1. Array geometry
    2. Steering vectors
    3. Hardware impairments
    4. Mutual coupling
    5. Array response / beam pattern
    6. Physical quantities used by PINN-Cal

No neural-network code belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class ArrayConfig:
    """Physical configuration of the antenna array."""

    num_elements: int = 64
    frequency_hz: float = 28e9
    element_spacing_lambda: float = 0.5
    c: float = 299_792_458.0

    @property
    def wavelength_m(self) -> float:
        return self.c / self.frequency_hz

    @property
    def spacing_m(self) -> float:
        return self.element_spacing_lambda * self.wavelength_m

    @property
    def wavenumber(self) -> float:
        return 2.0 * np.pi / self.wavelength_m


DEFAULT_CONFIG = ArrayConfig()


def element_positions(config: ArrayConfig = DEFAULT_CONFIG) -> torch.Tensor:
    """
    Return 1-D ULA element positions in metres.

    Positions are centered around zero so that the array geometry is
    symmetric about the origin.
    """
    n = config.num_elements
    d = config.spacing_m

    positions = (
        torch.arange(n, dtype=torch.float64)
        - (n - 1) / 2.0
    ) * d

    return positions


def steering_vector(
    angle_deg: torch.Tensor | float,
    config: ArrayConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """
    Compute the complex steering vector for a ULA.

    Convention:
        a_n(theta) = exp(j k x_n sin(theta))

    Parameters
    ----------
    angle_deg:
        Scalar or tensor of look directions in degrees.

    Returns
    -------
    torch.Tensor
        Complex steering vector(s).
    """
    angle = torch.as_tensor(angle_deg, dtype=torch.float64)

    theta = torch.deg2rad(angle)
    positions = element_positions(config).to(angle.device)

    phase = (
        config.wavenumber
        * positions
        * torch.sin(theta)[..., None]
    )

    return torch.exp(1j * phase)


def normalize_power(weights: torch.Tensor) -> torch.Tensor:
    """
    Normalize complex array weights so total excitation power is one.

    The operation preserves relative amplitude and phase.
    """
    power = torch.sum(torch.abs(weights) ** 2, dim=-1, keepdim=True)
    return weights / torch.sqrt(power.clamp_min(1e-12))


def array_factor(
    weights: torch.Tensor,
    angles_deg: torch.Tensor,
    config: ArrayConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """
    Compute complex array factor for arbitrary complex excitation weights.

    Parameters
    ----------
    weights:
        Shape (..., M)
    angles_deg:
        Shape (K,)

    Returns
    -------
    torch.Tensor
        Complex array response with shape (..., K)
    """
    weights = torch.as_tensor(weights)

    angles = torch.as_tensor(
        angles_deg,
        dtype=torch.float64,
        device=weights.device,
    )

    steering = steering_vector(angles, config).to(weights.device)
    # steering shape: (K, M)

    return torch.sum(
        weights[..., None, :] * torch.conj(steering),
        dim=-1,
    )


def power_pattern(
    weights: torch.Tensor,
    angles_deg: torch.Tensor,
    config: ArrayConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """
    Compute normalized power pattern |AF|^2 / max(|AF|^2).
    """
    af = array_factor(weights, angles_deg, config)
    power = torch.abs(af) ** 2

    maximum = power.amax(dim=-1, keepdim=True).clamp_min(1e-12)

    return power / maximum


def beam_peak_angle(
    weights: torch.Tensor,
    angles_deg: torch.Tensor,
    config: ArrayConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """
    Return the angle corresponding to the maximum beam power.
    """
    pattern = power_pattern(weights, angles_deg, config)
    peak_index = torch.argmax(pattern, dim=-1)

    angles = torch.as_tensor(
        angles_deg,
        dtype=torch.float64,
        device=weights.device,
    )

    return angles[peak_index]
def apply_hardware_errors(
    weights: torch.Tensor,
    phase_error_deg: torch.Tensor,
    gain_error_db: torch.Tensor,
) -> torch.Tensor:
    """
    Apply per-element gain and phase errors to complex array weights.

    Gain error is specified in dB.
    Phase error is specified in degrees.
    """
    phase_rad = torch.deg2rad(phase_error_deg)
    gain_linear = 10.0 ** (gain_error_db / 20.0)

    error_factor = gain_linear * torch.exp(1j * phase_rad)

    return weights * error_factor


def generate_hardware_errors(
    num_elements: int,
    phase_sigma_deg: float = 5.0,
    gain_sigma_db: float = 0.5,
    seed: int | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate independent Gaussian gain and phase errors.
    """
    generator = None

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    phase_error_deg = torch.randn(
        num_elements,
        generator=generator,
        device=device,
        dtype=torch.float64,
    ) * phase_sigma_deg

    gain_error_db = torch.randn(
        num_elements,
        generator=generator,
        device=device,
        dtype=torch.float64,
    ) * gain_sigma_db

    return phase_error_deg, gain_error_db


def mutual_impedance_matrix(
    config: ArrayConfig = DEFAULT_CONFIG,
    self_impedance_ohm: complex = 73.1 + 42.5j,
    nearest_neighbor_coupling: float = 0.12,
    decay: float = 0.55,
    phase_step_rad: float = 0.35,
) -> torch.Tensor:
    """
    Build a simplified reciprocal mutual-impedance matrix.

    Important:
        This is a compact network model, not a full-wave EM solution.

    Properties:
        - Z_ii = self impedance
        - Z_ij = Z_ji  (reciprocal network)
        - coupling decreases with element separation
        - complex phase is retained

    The matrix is deliberately isolated so that a measured or
    EM-solver-derived impedance/S-parameter matrix can replace it later.
    """
    m = config.num_elements

    Z = torch.zeros(
        (m, m),
        dtype=torch.complex128,
    )

    z_self = torch.tensor(
        self_impedance_ohm,
        dtype=torch.complex128,
    )

    Z.fill_diagonal_(z_self)

    z_nn = nearest_neighbor_coupling * z_self

    for i in range(m):
        for j in range(i + 1, m):
            separation = j - i

            magnitude = decay ** (separation - 1)
            phase = -phase_step_rad * (separation - 1)

            z_mutual = z_nn * magnitude * torch.exp(
                torch.tensor(1j * phase, dtype=torch.complex128)
            )

            # Reciprocal network: Z_ij = Z_ji
            Z[i, j] = z_mutual
            Z[j, i] = z_mutual

    return Z


def apply_mutual_coupling(
    weights: torch.Tensor,
    config: ArrayConfig = DEFAULT_CONFIG,
    coupling_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply the mutual-coupling network to array excitations.

    Interpretation:
        V = Z I

    For a desired excitation voltage V, the coupled current is:

        I = Z^{-1} V

    We scale by the self impedance so that the uncoupled case
    approaches the original excitation vector.
    """
    if coupling_matrix is None:
        Z = mutual_impedance_matrix(config)
    else:
        Z = coupling_matrix

    Z = Z.to(device=weights.device)

    z_self = Z.diagonal().mean()

    excitation = weights.to(torch.complex128)

    # Coupled current response.
    currents = torch.linalg.solve(
        Z,
        excitation * z_self,
    )

    return currents.to(weights.dtype)
