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

SNR_LEVELS = (
    40.0,
    30.0,
    20.0,
    15.0,
    10.0,
)

NUM_SAMPLES = 200
ROBUSTNESS_SEED = 4200

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
# Nominal analytical baseline
# =====================================================================

def analytical_correction(
    observed_current: np.ndarray,
    steering_angles_deg: np.ndarray,
) -> np.ndarray:

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

        v_eff = (
            z_nom
            @ current
        )

        w_hardware = (
            v_eff
            / SELF_IMPEDANCE
        )

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

        correction = (
            w_ideal
            / (
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
# Dataset preparation
# =====================================================================

def prepare_dataset(
    snr_db: float,
):
    config = DatasetConfig(
        num_samples=NUM_SAMPLES,
        phase_sigma_deg=20.0,
        gain_sigma_db=2.0,
        snr_db=snr_db,
        coupling_mismatch_fraction=0.20,
        min_angle_deg=-60.0,
        max_angle_deg=60.0,
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
        "PINN-Cal — SNR Robustness Evaluation"
    )
    print("=" * 90)

    print(
        f"Model checkpoint   : {MODEL_PATH}"
    )

    print(
        f"Samples per SNR    : {NUM_SAMPLES}"
    )

    print(
        f"Robustness seed    : {ROBUSTNESS_SEED}"
    )

    print(
        "Hardware phase σ   : 20.0°"
    )

    print(
        "Hardware gain σ    : 2.0 dB"
    )

    print(
        "Coupling mismatch  : ±20%"
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
    # Generate the first dataset.
    #
    # The same seed will be used for every SNR so that:
    #
    #     angles
    #     hardware errors
    #     coupling realization
    #
    # remain identical while only measurement noise changes.
    # -----------------------------------------------------------------

    datasets = {}

    for snr in SNR_LEVELS:

        datasets[snr] = (
            prepare_dataset(snr)
        )

    # -----------------------------------------------------------------
    # Verify that targets remain unchanged across SNR.
    # -----------------------------------------------------------------

    reference_y = datasets[
        SNR_LEVELS[0]
    ]["Y"]

    for snr in SNR_LEVELS[1:]:

        current_y = datasets[
            snr
        ]["Y"]

        max_target_difference = np.max(
            np.abs(
                current_y
                - reference_y
            )
        )

        if max_target_difference > 1e-6:

            raise RuntimeError(
                "Target correction changed "
                f"across SNR levels. "
                f"SNR={snr}, "
                f"max difference="
                f"{max_target_difference}"
            )

    print(
        "Target invariance check : PASS"
    )

    print(
        "Only measurement noise changes across SNR levels."
    )

    print()

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    results = []

    for snr in SNR_LEVELS:

        data = datasets[snr]

        X = data["X"]
        Y = data["Y"]
        steering_angles = (
            data["steering_angles"]
        )

        observed_current = (
            X[:, :64]
            + 1j * X[:, 64:128]
        )

        target = (
            Y[:, :64]
            + 1j * Y[:, 64:]
        )

        # -------------------------------------------------------------
        # Analytical baseline
        # -------------------------------------------------------------

        analytical = (
            analytical_correction(
                observed_current,
                steering_angles,
            )
        )

        # -------------------------------------------------------------
        # PINN
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
        # Compute metrics
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
                snr,
                analytical_metrics,
                pinn_metrics,
            )
        )

    # -----------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------

    print(
        "=" * 90
    )

    print(
        "SNR ROBUSTNESS RESULTS"
    )

    print(
        "=" * 90
    )

    print()

    print(
        "ANALYTICAL BASELINE"
    )

    print(
        f"{'SNR':>8}"
        f"{'NMSE(dB)':>14}"
        f"{'Phase(°)':>14}"
        f"{'Gain(dB)':>14}"
        f"{'Rel.Err(%)':>15}"
    )

    print(
        "-" * 70
    )

    for snr, analytical_metrics, _ in results:

        print(
            f"{snr:8.1f}"
            f"{analytical_metrics['nmse']:14.4f}"
            f"{analytical_metrics['phase']:14.4f}"
            f"{analytical_metrics['gain']:14.4f}"
            f"{100.0 * analytical_metrics['relative']:15.4f}"
        )

    print()

    print(
        "PINN"
    )

    print(
        f"{'SNR':>8}"
        f"{'NMSE(dB)':>14}"
        f"{'Phase(°)':>14}"
        f"{'Gain(dB)':>14}"
        f"{'Rel.Err(%)':>15}"
    )

    print(
        "-" * 70
    )

    for snr, _, pinn_metrics in results:

        print(
            f"{snr:8.1f}"
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
        "SNR robustness evaluation complete."
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
