from pathlib import Path

import numpy as np

from physics_core import (
    ArrayConfig,
    mutual_impedance_matrix,
    steering_vector,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CONFIG = ArrayConfig()

SELF_IMPEDANCE = 73.1 + 42.5j
COUPLING = 0.12
DECAY = 0.55
PHASE_STEP_RAD = 0.35

DATA_PATH = Path("data/physics_test_200.npy")


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_test_data():
    """
    Load the controlled test split.

    The split file contains:
        X
        Y
        steering_angles

    X:
        64 real observed-current values
        64 imaginary observed-current values
        sin(theta)
        cos(theta)

    Y:
        64 real correction values
        64 imaginary correction values

    steering_angles:
        Original steering angles in degrees.

    The original steering angles are used directly because
    physics_core.steering_vector() expects degrees.
    """

    data = np.load(
        DATA_PATH,
        allow_pickle=True,
    ).item()

    x = data["X"]
    y = data["Y"]
    steering_angles_deg = data["steering_angles"]

    # ---------------------------------------------------------------
    # Observed complex current
    # ---------------------------------------------------------------

    observed_current = (
        x[:, :64]
        + 1j * x[:, 64:128]
    )

    return (
        observed_current,
        steering_angles_deg,
        y,
    )


# ---------------------------------------------------------------------
# Analytical calibration
# ---------------------------------------------------------------------

def analytical_correction(
    observed_current: np.ndarray,
    steering_angles_deg: np.ndarray,
) -> np.ndarray:
    """
    Recover the hardware correction analytically using the
    nominal coupling matrix.

    Dataset forward model:

        Z_true I_true = V_hardware * Z_self,true

    Analytical inverse under the nominal model:

        V_eff_nom = Z_nom I_observed

        V_hardware_est =
            V_eff_nom / Z_self_nom

        correction =
            V_ideal / V_hardware_est
    """

    # ---------------------------------------------------------------
    # Nominal coupling matrix
    # ---------------------------------------------------------------

    z_nom = (
        mutual_impedance_matrix(
            config=CONFIG,
            self_impedance_ohm=SELF_IMPEDANCE,
            nearest_neighbor_coupling=COUPLING,
            decay=DECAY,
            phase_step_rad=PHASE_STEP_RAD,
        )
        .detach()
        .cpu()
        .numpy()
    )

    corrections = []

    # ---------------------------------------------------------------
    # Process each test sample
    # ---------------------------------------------------------------

    for i in range(observed_current.shape[0]):

        current = observed_current[i]

        # -----------------------------------------------------------
        # Reconstruct effective excitation using nominal Z.
        # -----------------------------------------------------------

        v_eff = z_nom @ current

        # -----------------------------------------------------------
        # Recover estimated hardware excitation.
        # -----------------------------------------------------------

        w_hardware = (
            v_eff / SELF_IMPEDANCE
        )

        # -----------------------------------------------------------
        # Reconstruct the exact ideal steering excitation.
        #
        # IMPORTANT:
        # steering_angles_deg is in degrees.
        # steering_vector() expects degrees.
        # -----------------------------------------------------------

        w_ideal = (
            steering_vector(
                steering_angles_deg[i],
                config=CONFIG,
            )
            .detach()
            .cpu()
            .numpy()
        )

        # -----------------------------------------------------------
        # Required calibration correction.
        # -----------------------------------------------------------

        correction = (
            w_ideal
            / (w_hardware + 1e-12)
        )

        corrections.append(correction)

    return np.asarray(
        corrections,
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def nmse_db(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Normalized mean squared error in dB.
    """

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
    """
    Mean absolute wrapped phase error in degrees.
    """

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
    """
    Mean absolute magnitude error in dB.
    """

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
    """
    Mean relative complex correction error.
    """

    return float(
        np.mean(
            np.abs(
                predicted - target
            )
            /
            (
                np.abs(target)
                + 1e-12
            )
        )
    )


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main():

    # ---------------------------------------------------------------
    # Load test set
    # ---------------------------------------------------------------

    (
        observed_current,
        steering_angles_deg,
        y,
    ) = load_test_data()

    # ---------------------------------------------------------------
    # Decode target correction
    #
    # Y = [real(64), imag(64)]
    # ---------------------------------------------------------------

    target = (
        y[:, :64]
        + 1j * y[:, 64:]
    )

    # ---------------------------------------------------------------
    # Analytical prediction
    # ---------------------------------------------------------------

    predicted = analytical_correction(
        observed_current,
        steering_angles_deg,
    )

    # ---------------------------------------------------------------
    # Shape validation
    # ---------------------------------------------------------------

    if predicted.shape != target.shape:
        raise RuntimeError(
            "Prediction/target shape mismatch: "
            f"{predicted.shape} vs {target.shape}"
        )

    # ---------------------------------------------------------------
    # Finite-value validation
    # ---------------------------------------------------------------

    if not np.all(
        np.isfinite(predicted.real)
    ):
        raise RuntimeError(
            "Predicted real values contain "
            "non-finite values."
        )

    if not np.all(
        np.isfinite(predicted.imag)
    ):
        raise RuntimeError(
            "Predicted imaginary values contain "
            "non-finite values."
        )

    if not np.all(
        np.isfinite(target.real)
    ):
        raise RuntimeError(
            "Target real values contain "
            "non-finite values."
        )

    if not np.all(
        np.isfinite(target.imag)
    ):
        raise RuntimeError(
            "Target imaginary values contain "
            "non-finite values."
        )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print("=" * 70)
    print(
        "PINN-Cal — Analytical Calibration Baseline"
    )
    print("=" * 70)

    print(
        f"Test samples       : "
        f"{len(target)}"
    )

    print(
        f"Input current      : "
        f"{observed_current.shape}"
    )

    print(
        f"Target correction  : "
        f"{target.shape}"
    )

    print(
        f"Steering range     : "
        f"{steering_angles_deg.min():.3f}° "
        f"to "
        f"{steering_angles_deg.max():.3f}°"
    )

    print()

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

    print("=" * 70)


if __name__ == "__main__":
    main()
