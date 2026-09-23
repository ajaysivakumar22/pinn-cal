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

# Five steering regions covering the complete training range.
ANGLE_REGIONS = (
    (-60.0, -36.0),
    (-36.0, -12.0),
    (-12.0, 12.0),
    (12.0, 36.0),
    (36.0, 60.0),
)

NUM_SAMPLES_PER_REGION = 200

ROBUSTNESS_SEED = 6200

# Keep all non-angle physical conditions fixed.
SNR_DB = 20.0
PHASE_SIGMA_DEG = 20.0
GAIN_SIGMA_DB = 2.0
COUPLING_MISMATCH_FRACTION = 0.20


# Nominal coupling model used by the analytical baseline.
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
    Recover calibration using the nominal coupling matrix.

    The hidden true coupling can differ from the nominal model,
    but the analytical baseline only has access to Z_nom.
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
        # Reconstruct nominal effective voltage.
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
        # Analytical calibration correction.
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

def prepare_region_dataset(
    min_angle_deg: float,
    max_angle_deg: float,
):
    """
    Generate one steering-region dataset.

    All non-angle physical conditions remain fixed.

    The same robustness seed is used for every region so the
    hardware-error and hidden-coupling realizations are generated
    consistently by sample index. Only the steering-angle distribution
    changes.
    """

    config = DatasetConfig(
        num_samples=NUM_SAMPLES_PER_REGION,

        phase_sigma_deg=PHASE_SIGMA_DEG,

        gain_sigma_db=GAIN_SIGMA_DB,

        snr_db=SNR_DB,

        coupling_mismatch_fraction=(
            COUPLING_MISMATCH_FRACTION
        ),

        min_angle_deg=min_angle_deg,

        max_angle_deg=max_angle_deg,

        seed=ROBUSTNESS_SEED,
    )

    return generate_dataset(
        config=config,
        array_config=DEFAULT_CONFIG,
    )


# =====================================================================
# Region label
# =====================================================================

def region_label(
    min_angle_deg: float,
    max_angle_deg: float,
) -> str:

    return (
        f"{min_angle_deg:+.0f}° "
        f"to "
        f"{max_angle_deg:+.0f}°"
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print("=" * 90)
    print(
        "PINN-Cal — Steering Robustness Evaluation"
    )
    print("=" * 90)

    print(
        f"Model checkpoint       : {MODEL_PATH}"
    )

    print(
        f"Samples per region     : "
        f"{NUM_SAMPLES_PER_REGION}"
    )

    print(
        f"Robustness seed        : "
        f"{ROBUSTNESS_SEED}"
    )

    print(
        f"SNR                    : "
        f"{SNR_DB:.1f} dB"
    )

    print(
        f"Hardware phase σ       : "
        f"{PHASE_SIGMA_DEG:.1f}°"
    )

    print(
        f"Hardware gain σ        : "
        f"{GAIN_SIGMA_DB:.1f} dB"
    )

    print(
        f"Coupling mismatch      : "
        f"±{100.0 * COUPLING_MISMATCH_FRACTION:.0f}%"
    )

    print()

    # -----------------------------------------------------------------
    # Load frozen PINN checkpoint.
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
    # Generate one dataset per steering region.
    # -----------------------------------------------------------------

    datasets = {}

    for (
        min_angle_deg,
        max_angle_deg,
    ) in ANGLE_REGIONS:

        label = region_label(
            min_angle_deg,
            max_angle_deg,
        )

        datasets[label] = (
            prepare_region_dataset(
                min_angle_deg,
                max_angle_deg,
            )
        )

    # -----------------------------------------------------------------
    # Verify that each dataset stayed inside its assigned region.
    # -----------------------------------------------------------------

    for (
        min_angle_deg,
        max_angle_deg,
    ) in ANGLE_REGIONS:

        label = region_label(
            min_angle_deg,
            max_angle_deg,
        )

        angles = datasets[
            label
        ]["steering_angles"]

        observed_min = float(
            np.min(angles)
        )

        observed_max = float(
            np.max(angles)
        )

        if (
            observed_min < min_angle_deg
            or observed_max > max_angle_deg
        ):

            raise RuntimeError(
                "Steering-angle range violation "
                f"in region {label}: "
                f"{observed_min:.4f}° "
                f"to "
                f"{observed_max:.4f}°"
            )

    print(
        "Steering-region bounds check : PASS"
    )

    print()

    # -----------------------------------------------------------------
    # Evaluate each steering region.
    # -----------------------------------------------------------------

    results = []

    for (
        min_angle_deg,
        max_angle_deg,
    ) in ANGLE_REGIONS:

        label = region_label(
            min_angle_deg,
            max_angle_deg,
        )

        data = datasets[
            label
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
        # Compute metrics.
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
                min_angle_deg,
                max_angle_deg,
                analytical_metrics,
                pinn_metrics,
            )
        )

    # -----------------------------------------------------------------
    # Print analytical results.
    # -----------------------------------------------------------------

    print("=" * 90)

    print(
        "STEERING ROBUSTNESS RESULTS"
    )

    print("=" * 90)

    print()

    print(
        "ANALYTICAL BASELINE"
    )

    print(
        f"{'Region':>18}"
        f"{'NMSE(dB)':>14}"
        f"{'Phase(°)':>14}"
        f"{'Gain(dB)':>14}"
        f"{'Rel.Err(%)':>15}"
    )

    print(
        "-" * 80
    )

    for (
        min_angle_deg,
        max_angle_deg,
        analytical_metrics,
        _,
    ) in results:

        label = region_label(
            min_angle_deg,
            max_angle_deg,
        )

        print(
            f"{label:>18}"
            f"{analytical_metrics['nmse']:14.4f}"
            f"{analytical_metrics['phase']:14.4f}"
            f"{analytical_metrics['gain']:14.4f}"
            f"{100.0 * analytical_metrics['relative']:15.4f}"
        )

    # -----------------------------------------------------------------
    # Print PINN results.
    # -----------------------------------------------------------------

    print()

    print(
        "PINN"
    )

    print(
        f"{'Region':>18}"
        f"{'NMSE(dB)':>14}"
        f"{'Phase(°)':>14}"
        f"{'Gain(dB)':>14}"
        f"{'Rel.Err(%)':>15}"
    )

    print(
        "-" * 80
    )

    for (
        min_angle_deg,
        max_angle_deg,
        _,
        pinn_metrics,
    ) in results:

        label = region_label(
            min_angle_deg,
            max_angle_deg,
        )

        print(
            f"{label:>18}"
            f"{pinn_metrics['nmse']:14.4f}"
            f"{pinn_metrics['phase']:14.4f}"
            f"{pinn_metrics['gain']:14.4f}"
            f"{100.0 * pinn_metrics['relative']:15.4f}"
        )

    # -----------------------------------------------------------------
    # Aggregate statistics.
    # -----------------------------------------------------------------

    analytical_relative_errors = [
        metrics["relative"]
        for (
            _,
            _,
            metrics,
            _,
        ) in results
    ]

    pinn_relative_errors = [
        metrics["relative"]
        for (
            _,
            _,
            _,
            metrics,
        ) in results
    ]

    print()

    print(
        "=" * 90
    )

    print(
        "AGGREGATE STEERING STATISTICS"
    )

    print(
        "=" * 90
    )

    print(
        "Analytical mean relative error : "
        f"{100.0 * np.mean(analytical_relative_errors):.4f}%"
    )

    print(
        "PINN mean relative error       : "
        f"{100.0 * np.mean(pinn_relative_errors):.4f}%"
    )

    print()

    print(
        "Steering robustness evaluation complete."
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
