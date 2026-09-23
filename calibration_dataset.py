"""
PINN-Cal — Physically Grounded Calibration Dataset

Generates training samples from the differentiable physics core.

Input:
    - Real + imaginary parts of the coupled complex array current
    - sin(target_angle)
    - cos(target_angle)

Target:
    - Real + imaginary parts of the exact inverse hardware correction

The target correction is:
    C = V_ideal / V_hardware

The physical engine is then used by the PINN to verify whether the
predicted correction restores the desired coupled array state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from physics_core import (
    ArrayConfig,
    DEFAULT_CONFIG,
    apply_hardware_errors,
    generate_hardware_errors,
    mutual_impedance_matrix,
    steering_vector,
)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for calibration dataset generation."""

    num_samples: int = 1000

    phase_sigma_deg: float = 20.0
    gain_sigma_db: float = 2.0

    # Measurement SNR in dB.
    # None disables measurement noise.
    snr_db: float | None = 20.0

    min_angle_deg: float = -60.0
    max_angle_deg: float = 60.0

    seed: int = 42


def generate_sample(
    angle_deg: float,
    seed: int,
    config: ArrayConfig = DEFAULT_CONFIG,
    dataset_config: DatasetConfig = DatasetConfig(),
    coupling_matrix: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one physically grounded calibration sample.

    Returns
    -------
    x : np.ndarray
        Input vector of shape (130,):
            [Re(I_obs), Im(I_obs), sin(theta), cos(theta)]

    y : np.ndarray
        Target vector of shape (128,):
            [Re(C), Im(C)]
    """

    if coupling_matrix is None:
        coupling_matrix = mutual_impedance_matrix(config)

    coupling_matrix = coupling_matrix.to(torch.complex128)

    z_self = coupling_matrix.diagonal().mean()

    # ------------------------------------------------------------------
    # Ideal desired excitation for the requested beam direction.
    # ------------------------------------------------------------------
    v_ideal = steering_vector(
        angle_deg,
        config,
    ).to(torch.complex128)

    # ------------------------------------------------------------------
    # Hardware impairment.
    # ------------------------------------------------------------------
    phase_error_deg, gain_error_db = generate_hardware_errors(
        config.num_elements,
        phase_sigma_deg=dataset_config.phase_sigma_deg,
        gain_sigma_db=dataset_config.gain_sigma_db,
        seed=seed,
        device=torch.device("cpu"),
    )

    v_hardware = apply_hardware_errors(
        v_ideal,
        phase_error_deg,
        gain_error_db,
    ).to(torch.complex128)

    # ------------------------------------------------------------------
    # Physical observation after mutual coupling.
    # ------------------------------------------------------------------
    i_observed = torch.linalg.solve(
        coupling_matrix,
        v_hardware * z_self,
    )

    # ------------------------------------------------------------------
    # Optional complex AWGN measurement noise.
    #
    # The target remains the true hardware correction.
    # Only the observed input is corrupted.
    # ------------------------------------------------------------------
    if dataset_config.snr_db is not None:
        signal_power = torch.mean(
            torch.abs(i_observed) ** 2
        )

        noise_power = (
            signal_power
            / (10.0 ** (dataset_config.snr_db / 10.0))
        )

        noise_std = torch.sqrt(
            noise_power / 2.0
        )

        noise_generator = torch.Generator(
            device=i_observed.device
        )

        noise_generator.manual_seed(
            seed + 1_000_000
        )

        noise = torch.complex(
            torch.randn(
                i_observed.shape,
                generator=noise_generator,
                dtype=torch.float64,
                device=i_observed.device,
            ),
            torch.randn(
                i_observed.shape,
                generator=noise_generator,
                dtype=torch.float64,
                device=i_observed.device,
            ),
        ) * noise_std

        i_observed = i_observed + noise

    # ------------------------------------------------------------------
    # Exact hardware correction.
    # ------------------------------------------------------------------
    correction = v_ideal / v_hardware

    # ------------------------------------------------------------------
    # Model input:
    #   64 real + 64 imaginary coupled-current values
    #   + target angle encoded as sin/cos
    # ------------------------------------------------------------------
    x = torch.cat(
        [
            i_observed.real,
            i_observed.imag,
            torch.tensor(
                [
                    np.sin(np.deg2rad(angle_deg)),
                    np.cos(np.deg2rad(angle_deg)),
                ],
                dtype=torch.float64,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Training target:
    #   64 real + 64 imaginary correction coefficients
    # ------------------------------------------------------------------
    y = torch.cat(
        [
            correction.real,
            correction.imag,
        ]
    )

    return (
        x.numpy().astype(np.float32),
        y.numpy().astype(np.float32),
    )


def generate_dataset(
    config: DatasetConfig = DatasetConfig(),
    array_config: ArrayConfig = DEFAULT_CONFIG,
) -> dict[str, np.ndarray]:
    """
    Generate a complete calibration dataset.
    """

    rng = np.random.default_rng(config.seed)

    # One fixed physical coupling matrix is shared by the dataset.
    coupling_matrix = mutual_impedance_matrix(
        array_config
    ).cpu()

    X = np.zeros(
        (
            config.num_samples,
            2 * array_config.num_elements + 2,
        ),
        dtype=np.float32,
    )

    Y = np.zeros(
        (
            config.num_samples,
            2 * array_config.num_elements,
        ),
        dtype=np.float32,
    )

    angles = np.zeros(
        config.num_samples,
        dtype=np.float32,
    )

    for idx in range(config.num_samples):
        angle = float(
            rng.uniform(
                config.min_angle_deg,
                config.max_angle_deg,
            )
        )

        x, y = generate_sample(
            angle_deg=angle,
            seed=config.seed + idx,
            config=array_config,
            dataset_config=config,
            coupling_matrix=coupling_matrix,
        )

        X[idx] = x
        Y[idx] = y
        angles[idx] = angle

    return {
        "X": X,
        "Y": Y,
        "steering_angles": angles,
    }