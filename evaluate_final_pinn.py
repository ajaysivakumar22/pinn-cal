from pathlib import Path

import numpy as np
import torch

from physics_core import (
    DEFAULT_CONFIG,
    mutual_impedance_matrix,
    steering_vector,
)

from pinn_cal_final import (
    PINNCalV3,
    DEVICE,
)


# =====================================================================
# Configuration
# =====================================================================

DATA_PATH = Path(
    "data/physics_test_200.npy"
)

MODEL_PATH = Path(
    "models/pinn_cal_final_best.pth"
)

CONFIG = DEFAULT_CONFIG

SELF_IMPEDANCE = 73.1 + 42.5j
COUPLING = 0.12
DECAY = 0.55
PHASE_STEP_RAD = 0.35


# =====================================================================
# Correction-domain metrics
# =====================================================================

def nmse_db(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    numerator = np.sum(
        np.abs(predicted - target) ** 2
    )

    denominator = np.sum(
        np.abs(target) ** 2
    )

    return float(
        10.0
        * np.log10(
            numerator / denominator
        )
    )


def phase_error_deg(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    phase_difference = np.angle(
        predicted * np.conj(target)
    )

    return float(
        np.degrees(
            np.mean(
                np.abs(phase_difference)
            )
        )
    )


def gain_error_db(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    predicted_db = (
        20.0
        * np.log10(
            np.maximum(
                np.abs(predicted),
                1e-12,
            )
        )
    )

    target_db = (
        20.0
        * np.log10(
            np.maximum(
                np.abs(target),
                1e-12,
            )
        )
    )

    return float(
        np.mean(
            np.abs(
                predicted_db - target_db
            )
        )
    )


def relative_error(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.abs(predicted - target)
            /
            (
                np.abs(target)
                + 1e-12
            )
        )
    )


# =====================================================================
# Nominal physics model
# =====================================================================

def build_nominal_impedance() -> torch.Tensor:
    """
    Construct the same nominal reciprocal impedance matrix used
    by the final PINN.
    """

    z_nom = mutual_impedance_matrix(
        config=CONFIG,
        self_impedance_ohm=SELF_IMPEDANCE,
        nearest_neighbor_coupling=COUPLING,
        decay=DECAY,
        phase_step_rad=PHASE_STEP_RAD,
    )

    return (
        z_nom
        .to(torch.complex128)
        .to(DEVICE)
    )


# =====================================================================
# Physics-domain quantities
# =====================================================================

def decode_observed_current(
    X: np.ndarray,
) -> torch.Tensor:
    """
    Decode the 64 complex observed-current values from X.
    """

    observed = (
        X[:, :64]
        + 1j * X[:, 64:128]
    )

    return torch.tensor(
        observed,
        dtype=torch.complex128,
        device=DEVICE,
    )


def get_ideal_coupled_current(
    steering_angles_deg: np.ndarray,
    z_nom: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ideal coupled current under the nominal impedance model.
    """

    angles = torch.tensor(
        steering_angles_deg,
        dtype=torch.float64,
        device=DEVICE,
    )

    v_ideal = steering_vector(
        angles,
        CONFIG,
    ).to(torch.complex128)

    z_self = (
        z_nom.diagonal()
        .mean()
        .to(torch.complex128)
    )

    rhs = (
        v_ideal
        * z_self
    )

    i_ideal = torch.linalg.solve(
        z_nom,
        rhs.T,
    ).T

    return i_ideal


def apply_correction_through_nominal_physics(
    observed_current: torch.Tensor,
    correction: torch.Tensor,
    z_nom: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a predicted correction through the same nominal physical
    network used by the PINN.

        I_obs
          -> V_obs = Z_nom I_obs
          -> V_corr = C V_obs
          -> I_corr = Z_nom^-1 V_corr
    """

    v_observed = (
        observed_current
        @ z_nom.T
    )

    v_corrected = (
        correction
        * v_observed
    )

    corrected_current = (
        torch.linalg.solve(
            z_nom,
            v_corrected.T,
        ).T
    )

    return corrected_current


# =====================================================================
# Current-domain metrics
# =====================================================================

def current_nmse_db(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    numerator = torch.sum(
        torch.abs(
            predicted - target
        ) ** 2
    )

    denominator = torch.sum(
        torch.abs(target) ** 2
    )

    return float(
        10.0
        * torch.log10(
            numerator
            /
            denominator
        )
        .detach()
        .cpu()
    )


def current_relative_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    error = (
        torch.abs(
            predicted - target
        )
        /
        (
            torch.abs(target)
            + 1e-12
        )
    )

    return float(
        torch.mean(error)
        .detach()
        .cpu()
    )


def current_phase_error_deg(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    phase_difference = torch.angle(
        predicted
        * torch.conj(target)
    )

    return float(
        torch.rad2deg(
            torch.mean(
                torch.abs(
                    phase_difference
                )
            )
        )
        .detach()
        .cpu()
    )


def current_gain_error_db(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    predicted_db = (
        20.0
        * torch.log10(
            torch.clamp(
                torch.abs(predicted),
                min=1e-12,
            )
        )
    )

    target_db = (
        20.0
        * torch.log10(
            torch.clamp(
                torch.abs(target),
                min=1e-12,
            )
        )
    )

    return float(
        torch.mean(
            torch.abs(
                predicted_db
                - target_db
            )
        )
        .detach()
        .cpu()
    )


# =====================================================================
# Reporting helpers
# =====================================================================

def print_correction_metrics(
    name: str,
    predicted: np.ndarray,
    target: np.ndarray,
) -> None:

    print()
    print(name)
    print("-" * 70)

    print(
        f"NMSE               : "
        f"{nmse_db(predicted, target):.4f} dB"
    )

    print(
        f"Mean phase error   : "
        f"{phase_error_deg(predicted, target):.4f}°"
    )

    print(
        f"Mean gain error    : "
        f"{gain_error_db(predicted, target):.4f} dB"
    )

    print(
        f"Mean relative err. : "
        f"{100.0 * relative_error(predicted, target):.4f}%"
    )


def print_current_metrics(
    name: str,
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> None:

    print()
    print(name)
    print("-" * 70)

    print(
        f"Current NMSE       : "
        f"{current_nmse_db(predicted, target):.4f} dB"
    )

    print(
        f"Current phase err. : "
        f"{current_phase_error_deg(predicted, target):.4f}°"
    )

    print(
        f"Current gain err.  : "
        f"{current_gain_error_db(predicted, target):.4f} dB"
    )

    print(
        f"Current rel. err.  : "
        f"{100.0 * current_relative_error(predicted, target):.4f}%"
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print("=" * 70)
    print(
        "PINN-Cal — Final Calibration Evaluation"
    )
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load untouched test set
    # -----------------------------------------------------------------

    data = np.load(
        DATA_PATH,
        allow_pickle=True,
    ).item()

    X = data["X"]
    Y = data["Y"]

    steering_angles_deg = (
        data["steering_angles"]
    )

    target_correction = (
        Y[:, :64]
        + 1j * Y[:, 64:]
    )

    # -----------------------------------------------------------------
    # Load nominal physics
    # -----------------------------------------------------------------

    z_nom = build_nominal_impedance()

    # -----------------------------------------------------------------
    # Decode observed current
    # -----------------------------------------------------------------

    observed_current = (
        decode_observed_current(X)
    )

    # -----------------------------------------------------------------
    # Ideal coupled current
    # -----------------------------------------------------------------

    ideal_current = (
        get_ideal_coupled_current(
            steering_angles_deg,
            z_nom,
        )
    )

    # -----------------------------------------------------------------
    # Analytical correction
    # -----------------------------------------------------------------

    steering_angles_tensor = torch.tensor(
        steering_angles_deg,
        dtype=torch.float64,
        device=DEVICE,
    )

    v_ideal = steering_vector(
        steering_angles_tensor,
        CONFIG,
    ).to(torch.complex128)

    estimated_effective_voltage = (
        observed_current
        @ z_nom.T
    )

    estimated_hardware = (
        estimated_effective_voltage
        / SELF_IMPEDANCE
    )

    analytical_correction = (
        v_ideal
        /
        (
            estimated_hardware
            + 1e-12
        )
    )

    # -----------------------------------------------------------------
    # Final PINN prediction
    # -----------------------------------------------------------------

    model = PINNCalV3().to(
        DEVICE
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        state
    )

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():

        prediction = model(
            X_tensor
        )

        pinn_correction = (
            torch.complex(
                prediction[:, :64]
                .to(torch.float64),

                prediction[:, 64:]
                .to(torch.float64),
            )
        )

    # -----------------------------------------------------------------
    # Uncalibrated current
    # -----------------------------------------------------------------

    uncalibrated_current = (
        observed_current
    )

    # -----------------------------------------------------------------
    # Apply analytical correction
    # -----------------------------------------------------------------

    analytical_corrected_current = (
        apply_correction_through_nominal_physics(
            observed_current,
            analytical_correction,
            z_nom,
        )
    )

    # -----------------------------------------------------------------
    # Apply PINN correction
    # -----------------------------------------------------------------

    pinn_corrected_current = (
        apply_correction_through_nominal_physics(
            observed_current,
            pinn_correction,
            z_nom,
        )
    )

    # -----------------------------------------------------------------
    # Convert PINN prediction to NumPy
    # -----------------------------------------------------------------

    pinn_prediction_np = (
        pinn_correction
        .detach()
        .cpu()
        .numpy()
    )

    # -----------------------------------------------------------------
    # Test-set information
    # -----------------------------------------------------------------

    print(
        f"Checkpoint         : "
        f"{MODEL_PATH}"
    )

    print(
        f"Test samples       : "
        f"{len(target_correction)}"
    )

    print(
        f"Input shape        : "
        f"{X.shape}"
    )

    print(
        f"Target shape       : "
        f"{target_correction.shape}"
    )

    print(
        f"Steering range     : "
        f"{steering_angles_deg.min():.3f}° "
        f"to "
        f"{steering_angles_deg.max():.3f}°"
    )

    # =================================================================
    # Correction-domain evaluation
    # =================================================================

    print()
    print("=" * 70)
    print(
        "CORRECTION-DOMAIN METRICS"
    )
    print("=" * 70)

    print_correction_metrics(
        "ANALYTICAL CALIBRATION",
        analytical_correction
        .detach()
        .cpu()
        .numpy(),
        target_correction,
    )

    print_correction_metrics(
        "PINN CALIBRATION",
        pinn_prediction_np,
        target_correction,
    )

    # =================================================================
    # Current-domain evaluation
    # =================================================================

    print()
    print("=" * 70)
    print(
        "NOMINAL COUPLED-CURRENT METRICS"
    )
    print("=" * 70)

    print_current_metrics(
        "UNCALIBRATED OBSERVATION",
        uncalibrated_current,
        ideal_current,
    )

    print_current_metrics(
        "ANALYTICAL CALIBRATION",
        analytical_corrected_current,
        ideal_current,
    )

    print_current_metrics(
        "PINN CALIBRATION",
        pinn_corrected_current,
        ideal_current,
    )

    # =================================================================
    # Sanity check
    # =================================================================

    print()
    print("=" * 70)
    print(
        "PHYSICS CONSISTENCY CHECK"
    )
    print("=" * 70)

    analytical_residual = (
        current_relative_error(
            analytical_corrected_current,
            ideal_current,
        )
    )

    pinn_residual = (
        current_relative_error(
            pinn_corrected_current,
            ideal_current,
        )
    )

    print(
        "Analytical current residual : "
        f"{100.0 * analytical_residual:.6f}%"
    )

    print(
        "PINN current residual       : "
        f"{100.0 * pinn_residual:.6f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()