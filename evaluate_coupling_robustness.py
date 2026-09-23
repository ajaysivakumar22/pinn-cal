from pathlib import Path

import numpy as np
import torch

from calibration_dataset import (
    DatasetConfig,
    generate_dataset,
)

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

MODEL_PATH = Path(
    "models/pinn_cal_final_best.pth"
)

# Coupling mismatch values to test.
# Example:
#   0.20 = ±20% mismatch
MISMATCH_LEVELS = (
    0.00,
    0.10,
    0.20,
    0.30,
    0.50,
)

NUM_SAMPLES = 200
ROBUSTNESS_SEED = 5200

# Keep every other impairment fixed.
SNR_DB = 20.0
PHASE_SIGMA_DEG = 20.0
GAIN_SIGMA_DB = 2.0

MIN_ANGLE_DEG = -60.0
MAX_ANGLE_DEG = 60.0

# Nominal coupling model known to the calibration methods.
SELF_IMPEDANCE = 73.1 + 42.5j
COUPLING = 0.12
DECAY = 0.55
PHASE_STEP_RAD = 0.35


# =====================================================================
# Metrics
# =====================================================================

def nmse_db(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:

    numerator = np.sum(
        np.abs(
            predicted - target
        ) ** 2
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
        predicted
        * np.conj(target)
    )

    return float(
        np.degrees(
            np.mean(
                np.abs(
                    phase_difference
                )
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
                predicted_db
                - target_db
            )
        )
    )


def relative_error(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:

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


# =====================================================================
# Analytical baseline
# =====================================================================

def analytical_correction(
    observed_current: np.ndarray,
    steering_angles_deg: np.ndarray,
) -> np.ndarray:
    """
    Recover calibration analytically using the nominal coupling matrix.

    The true test data may have a different hidden coupling matrix,
    but the analytical method only knows the nominal model.
    """

    z_nom = (
        mutual_impedance_matrix(
            config=DEFAULT_CONFIG,
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

    for i in range(
        observed_current.shape[0]
    ):

        current = observed_current[i]

        # -------------------------------------------------------------
        # Reconstruct nominal effective excitation.
        # -------------------------------------------------------------

        v_eff = (
            z_nom
            @ current
        )

        # -------------------------------------------------------------
        # Estimate hardware excitation.
        # -------------------------------------------------------------

        w_hardware = (
            v_eff
            / SELF_IMPEDANCE
        )

        # -------------------------------------------------------------
        # Ideal steering vector.
        # -------------------------------------------------------------

        angle = torch.tensor(
            steering_angles_deg[i],
            dtype=torch.float64,
        )

        w_ideal = (
            steering_vector(
                angle,
                DEFAULT_CONFIG,
            )
            .detach()
            .cpu()
            .numpy()
        )

        # -------------------------------------------------------------
        # Required analytical correction.
        # -------------------------------------------------------------

        correction = (
            w_ideal
            /
            (
                w_hardware
                + 1e-12
            )
        )

        corrections.append(
            correction
        )

    return np.asarray(
        corrections,
        dtype=np.complex128,
    )


# =====================================================================
# Dataset generation
# =====================================================================

def prepare_dataset(
    mismatch_fraction: float,
):
    """
    Generate one robustness dataset.

    Only the hidden coupling mismatch changes between experiments.

    The following remain fixed:
        - SNR
        - phase error distribution
        - gain error distribution
        - steering range
        - random seed
    """

    config = DatasetConfig(
        num_samples=NUM_SAMPLES,

        phase_sigma_deg=PHASE_SIGMA_DEG,

        gain_sigma_db=GAIN_SIGMA_DB,

        snr_db=SNR_DB,

        coupling_mismatch_fraction=mismatch_fraction,

        min_angle_deg=MIN_ANGLE_DEG,

        max_angle_deg=MAX_ANGLE_DEG,

        seed=ROBUSTNESS_SEED,
    )

    return generate_dataset(
        config=config,
        array_config=DEFAULT_CONFIG,
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print("=" * 90)
    print(
        "PINN-Cal — Coupling-Mismatch Robustness Evaluation"
    )
    print("=" * 90)

    print(
        f"Model checkpoint   : {MODEL_PATH}"
    )

    print(
        f"Samples per level  : {NUM_SAMPLES}"
    )

    print(
        f"Robustness seed    : {ROBUSTNESS_SEED}"
    )

    print(
        f"SNR                : {SNR_DB:.1f} dB"
    )

    print(
        f"Hardware phase σ   : {PHASE_SIGMA_DEG:.1f}°"
    )

    print(
        f"Hardware gain σ    : {GAIN_SIGMA_DB:.1f} dB"
    )

    print(
        f"Steering range     : "
        f"{MIN_ANGLE_DEG:.1f}° to "
        f"{MAX_ANGLE_DEG:.1f}°"
    )

    print()

    # -----------------------------------------------------------------
    # Load frozen PINN
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

    # -----------------------------------------------------------------
    # Generate datasets
    # -----------------------------------------------------------------

    datasets = {}

    for mismatch in MISMATCH_LEVELS:

        datasets[mismatch] = (
            prepare_dataset(
                mismatch
            )
        )

    # -----------------------------------------------------------------
    # Verify that hardware targets remain unchanged.
    #
    # The target correction depends on hardware errors, not coupling
    # mismatch. Therefore changing only coupling mismatch should not
    # change Y when the seed is fixed.
    # -----------------------------------------------------------------

    reference_mismatch = (
        MISMATCH_LEVELS[0]
    )

    reference_y = datasets[
        reference_mismatch
    ]["Y"]

    target_invariance_pass = True

    for mismatch in MISMATCH_LEVELS[1:]:

        current_y = datasets[
            mismatch
        ]["Y"]

        max_difference = np.max(
            np.abs(
                current_y
                - reference_y
            )
        )

        print(
            f"Target difference at "
            f"±{100.0 * mismatch:.0f}% : "
            f"{max_difference:.3e}"
        )

        if max_difference > 1e-6:
            target_invariance_pass = False

    print()

    if not target_invariance_pass:

        raise RuntimeError(
            "Ground-truth correction changed across coupling "
            "mismatch levels. The robustness setup is invalid."
        )

    print(
        "Target invariance check : PASS"
    )

    print(
        "Only the hidden coupling model changes."
    )

    print()

    # -----------------------------------------------------------------
    # Evaluate each mismatch level
    # -----------------------------------------------------------------

    results = []

    for mismatch in MISMATCH_LEVELS:

        data = datasets[
            mismatch
        ]

        X = data["X"]
        Y = data["Y"]
        steering_angles_deg = (
            data["steering_angles"]
        )

        # -------------------------------------------------------------
        # Decode observed current.
        # -------------------------------------------------------------

        observed_current = (
            X[:, :64]
            + 1j * X[:, 64:128]
        )

        # -------------------------------------------------------------
        # Ground-truth correction.
        # -------------------------------------------------------------

        target = (
            Y[:, :64]
            + 1j * Y[:, 64:]
        )

        # -------------------------------------------------------------
        # Analytical baseline.
        # -------------------------------------------------------------

        analytical = (
            analytical_correction(
                observed_current,
                steering_angles_deg,
            )
        )

        # -------------------------------------------------------------
        # Frozen PINN.
        # -------------------------------------------------------------

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=DEVICE,
        )

        with torch.no_grad():

            prediction = model(
                X_tensor
            )

        pinn = (
            prediction[:, :64]
            .detach()
            .cpu()
            .numpy()
            +
            1j
            * prediction[:, 64:]
            .detach()
            .cpu()
            .numpy()
        )

        # -------------------------------------------------------------
        # Metrics
        # -------------------------------------------------------------

        analytical_metrics = {
            "nmse":
                nmse_db(
                    analytical,
                    target,
                ),

            "phase":
                phase_error_deg(
                    analytical,
                    target,
                ),

            "gain":
                gain_error_db(
                    analytical,
                    target,
                ),

            "relative":
                relative_error(
                    analytical,
                    target,
                ),
        }

        pinn_metrics = {
            "nmse":
                nmse_db(
                    pinn,
                    target,
                ),

            "phase":
                phase_error_deg(
                    pinn,
                    target,
                ),

            "gain":
                gain_error_db(
                    pinn,
                    target,
                ),

            "relative":
                relative_error(
                    pinn,
                    target,
                ),
        }

        results.append(
            (
                mismatch,
                analytical_metrics,
                pinn_metrics,
            )
        )

    # -----------------------------------------------------------------
    # Analytical results
    # -----------------------------------------------------------------

    print(
        "=" * 90
    )

    print(
        "COUPLING-MISMATCH ROBUSTNESS RESULTS"
    )

    print(
        "=" * 90
    )

    print()

    print(
        "ANALYTICAL BASELINE"
    )

    print(
        f"{'Mismatch':>12}"
        f"{'NMSE(dB)':>14}"
        f"{'Phase(°)':>14}"
        f"{'Gain(dB)':>14}"
        f"{'Rel.Err(%)':>15}"
    )

    print(
        "-" * 75
    )

    for (
        mismatch,
        analytical_metrics,
        _,
    ) in results:

        print(
            f"{100.0 * mismatch:10.0f}%"
            f"{analytical_metrics['nmse']:14.4f}"
            f"{analytical_metrics['phase']:14.4f}"
            f"{analytical_metrics['gain']:14.4f}"
            f"{100.0 * analytical_metrics['relative']:15.4f}"
        )

    # -----------------------------------------------------------------
    # PINN results
    # -----------------------------------------------------------------

    print()

    print(
        "PINN"
    )

    print(
        f"{'Mismatch':>12}"
        f"{'NMSE(dB)':>14}"
        f"{'Phase(°)':>14}"
        f"{'Gain(dB)':>14}"
        f"{'Rel.Err(%)':>15}"
    )

    print(
        "-" * 75
    )

    for (
        mismatch,
        _,
        pinn_metrics,
    ) in results:

        print(
            f"{100.0 * mismatch:10.0f}%"
            f"{pinn_metrics['nmse']:14.4f}"
            f"{pinn_metrics['phase']:14.4f}"
            f"{pinn_metrics['gain']:14.4f}"
            f"{100.0 * pinn_metrics['relative']:15.4f}"
        )

    print()

    print(
        "=" * 90
    )

    print(
        "Coupling-mismatch robustness evaluation complete."
    )

    print(
        "The PINN checkpoint was not retrained."
    )

    print(
        "The original 800/200/200 training/test split was not modified."
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()
