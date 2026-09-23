"""
PINN-Cal — Physically Grounded Calibration Dataset

Generates training samples from the differentiable physics core.

Physical generation chain:

    Ideal excitation
        ↓
    Hardware gain/phase errors
        ↓
    TRUE mutual-coupling network
        ↓
    Coupled current
        ↓
    Measurement noise
        ↓
    Observed current

The PINN receives only the noisy observation and target steering angle.

The true coupling matrix is hidden from the model.
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

from coupling_model import (
    generate_true_coupling_matrix,
)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for calibration dataset generation."""

    num_samples: int = 1000

    # Hardware impairments
    phase_sigma_deg: float = 20.0
    gain_sigma_db: float = 2.0

    # Measurement noise
    snr_db: float | None = 20.0

    # Coupling uncertainty
    # ±20% parameter mismatch around nominal coupling.
    coupling_mismatch_fraction: float = 0.20

    # Steering range
    min_angle_deg: float = -60.0
    max_angle_deg: float = 60.0

    # Reproducibility
    seed: int = 42


def generate_sample(
    angle_deg: float,
    seed: int,
    config: ArrayConfig = DEFAULT_CONFIG,
    dataset_config: DatasetConfig = DatasetConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one physically grounded calibration sample.

    Input:
        130 values:
            64 real observed-current values
            64 imaginary observed-current values
            sin(theta)
            cos(theta)

    Target:
        128 values:
            64 real correction values
            64 imaginary correction values
    """

    # ---------------------------------------------------------------
    # Nominal coupling model.
    #
    # This represents the model available to PINN-Cal.
    # ---------------------------------------------------------------
    nominal_coupling = mutual_impedance_matrix(
        config
    ).to(torch.complex128).cpu()

    # ---------------------------------------------------------------
    # TRUE coupling matrix.
    #
    # This is deliberately different from the nominal matrix and is
    # hidden from the model.
    #
    # Use a separate deterministic seed so hardware errors and
    # coupling uncertainty are statistically independent.
    # ---------------------------------------------------------------
    true_coupling = generate_true_coupling_matrix(
        config=config,
        mismatch_fraction=(
            dataset_config.coupling_mismatch_fraction
        ),
        seed=seed + 2_000_000,
        nominal_matrix=nominal_coupling,
    ).to(torch.complex128)

    z_self_true = true_coupling.diagonal().mean()

    # ---------------------------------------------------------------
    # Ideal excitation.
    # ---------------------------------------------------------------
    v_ideal = steering_vector(
        angle_deg,
        config,
    ).to(torch.complex128)

    # ---------------------------------------------------------------
    # Hardware impairment.
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # TRUE physical array state.
    #
    # Z_true I_true = V_hardware * Z_self,true
    # ---------------------------------------------------------------
    i_true = torch.linalg.solve(
        true_coupling,
        v_hardware * z_self_true,
    )

    i_observed = i_true

    # ---------------------------------------------------------------
    # Measurement noise.
    # ---------------------------------------------------------------
    if dataset_config.snr_db is not None:
        signal_power = torch.mean(
            torch.abs(i_true) ** 2
        )

        noise_power = (
            signal_power
            / (
                10.0
                ** (
                    dataset_config.snr_db
                    / 10.0
                )
            )
        )

        noise_std = torch.sqrt(
            noise_power / 2.0
        )

        noise_generator = torch.Generator(
            device=i_true.device
        )

        noise_generator.manual_seed(
            seed + 1_000_000
        )

        noise = torch.complex(
            torch.randn(
                i_true.shape,
                generator=noise_generator,
                dtype=torch.float64,
                device=i_true.device,
            ),
            torch.randn(
                i_true.shape,
                generator=noise_generator,
                dtype=torch.float64,
                device=i_true.device,
            ),
        ) * noise_std

        i_observed = i_true + noise

    # ---------------------------------------------------------------
    # Exact hardware correction.
    #
    # Important:
    # The target depends only on the hardware impairment.
    #
    # The coupling matrix and measurement noise affect the
    # observation, not the ground-truth hardware correction.
    # ---------------------------------------------------------------
    correction = (
        v_ideal / v_hardware
    )

    # ---------------------------------------------------------------
    # Model input:
    #
    #   64 real observed-current values
    #   64 imaginary observed-current values
    #   sin(theta)
    #   cos(theta)
    # ---------------------------------------------------------------
    x = torch.cat(
        [
            i_observed.real,
            i_observed.imag,
            torch.tensor(
                [
                    np.sin(
                        np.deg2rad(angle_deg)
                    ),
                    np.cos(
                        np.deg2rad(angle_deg)
                    ),
                ],
                dtype=torch.float64,
            ),
        ]
    )

    # ---------------------------------------------------------------
    # Model target:
    #
    #   64 real correction values
    #   64 imaginary correction values
    # ---------------------------------------------------------------
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
    Generate a complete reproducible calibration dataset.
    """

    rng = np.random.default_rng(
        config.seed
    )

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

    for idx in range(
        config.num_samples
    ):
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
        )

        X[idx] = x
        Y[idx] = y
        angles[idx] = angle

    return {
        "X": X,
        "Y": Y,
        "steering_angles": angles,
    }