from pathlib import Path

import numpy as np
import torch

from physics_core import (
    ArrayConfig,
    mutual_impedance_matrix,
    power_pattern,
    steering_vector,
)

from pinn_cal_final import (
    PINNCalV3,
    DEVICE,
)


# =====================================================================
# Configuration
# =====================================================================

CONFIG = ArrayConfig()

DATA_PATH = Path(
    "data/physics_test_200.npy"
)

MODEL_PATH = Path(
    "models/pinn_cal_final_best.pth"
)

SELF_IMPEDANCE = 73.1 + 42.5j
COUPLING = 0.12
DECAY = 0.55
PHASE_STEP_RAD = 0.35

# 0.1 degree resolution over the visible angular region.
ANGLE_GRID_DEG = np.linspace(
    -90.0,
    90.0,
    1801,
)

ANGLE_GRID = torch.tensor(
    ANGLE_GRID_DEG,
    dtype=torch.float64,
    device=DEVICE,
)


# =====================================================================
# Metric helpers
# =====================================================================

def beam_correlation(
    pattern: np.ndarray,
    ideal_pattern: np.ndarray,
) -> float:
    """
    Cosine similarity between two normalized power patterns.
    """

    numerator = np.sum(
        pattern * ideal_pattern
    )

    denominator = (
        np.linalg.norm(pattern)
        * np.linalg.norm(ideal_pattern)
    )

    if denominator <= 1e-12:
        return float("nan")

    return float(
        numerator / denominator
    )


def interpolate_crossing(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    threshold: float,
) -> float:
    """
    Linear interpolation for a threshold crossing.
    """

    denominator = y2 - y1

    if abs(denominator) <= 1e-15:
        return float(
            0.5 * (x1 + x2)
        )

    fraction = (
        threshold - y1
    ) / denominator

    return float(
        x1 + fraction * (x2 - x1)
    )


def beamwidth_3db(
    pattern: np.ndarray,
    peak_index: int,
) -> float:
    """
    Compute the -3 dB beamwidth.

    Since the power pattern is normalized to 1,
    the -3 dB power threshold is approximately 0.5.

    The nearest 0.5 crossings around the main peak
    are located using linear interpolation.
    """

    threshold = 0.5

    # -------------------------------------------------------------
    # Search left from the main peak.
    # -------------------------------------------------------------

    left_crossing = None

    for i in range(
        peak_index - 1,
        -1,
        -1,
    ):

        if (
            pattern[i] < threshold
            <= pattern[i + 1]
        ):

            left_crossing = (
                interpolate_crossing(
                    ANGLE_GRID_DEG[i],
                    pattern[i],
                    ANGLE_GRID_DEG[i + 1],
                    pattern[i + 1],
                    threshold,
                )
            )

            break

    # -------------------------------------------------------------
    # Search right from the main peak.
    # -------------------------------------------------------------

    right_crossing = None

    for i in range(
        peak_index,
        len(pattern) - 1,
    ):

        if (
            pattern[i] >= threshold
            > pattern[i + 1]
        ):

            right_crossing = (
                interpolate_crossing(
                    ANGLE_GRID_DEG[i],
                    pattern[i],
                    ANGLE_GRID_DEG[i + 1],
                    pattern[i + 1],
                    threshold,
                )
            )

            break

    if (
        left_crossing is None
        or right_crossing is None
    ):
        return float("nan")

    return float(
        right_crossing
        - left_crossing
    )


def find_left_mainlobe_boundary(
    pattern: np.ndarray,
    peak_index: int,
):
    """
    Find the nearest local minimum to the left
    of the main beam peak.
    """

    for i in range(
        peak_index - 1,
        0,
        -1,
    ):

        if (
            pattern[i]
            <= pattern[i - 1]
            and pattern[i]
            <= pattern[i + 1]
        ):
            return i

    return None


def find_right_mainlobe_boundary(
    pattern: np.ndarray,
    peak_index: int,
):
    """
    Find the nearest local minimum to the right
    of the main beam peak.
    """

    for i in range(
        peak_index + 1,
        len(pattern) - 1,
    ):

        if (
            pattern[i]
            <= pattern[i - 1]
            and pattern[i]
            <= pattern[i + 1]
        ):
            return i

    return None


def peak_sidelobe_level_db(
    pattern: np.ndarray,
    peak_index: int,
) -> float:
    """
    Estimate peak sidelobe level relative to the main-beam peak.

    The main lobe is excluded between the nearest local minima
    surrounding the main peak.
    """

    left_boundary = (
        find_left_mainlobe_boundary(
            pattern,
            peak_index,
        )
    )

    right_boundary = (
        find_right_mainlobe_boundary(
            pattern,
            peak_index,
        )
    )

    if (
        left_boundary is None
        or right_boundary is None
    ):
        return float("nan")

    sidelobe_values = np.concatenate(
        [
            pattern[:left_boundary],
            pattern[right_boundary + 1:],
        ]
    )

    if sidelobe_values.size == 0:
        return float("nan")

    peak_sidelobe_power = np.max(
        sidelobe_values
    )

    return float(
        10.0
        * np.log10(
            max(
                peak_sidelobe_power,
                1e-12,
            )
        )
    )


# =====================================================================
# Pattern generation
# =====================================================================

def calculate_patterns(
    weights: torch.Tensor,
) -> np.ndarray:
    """
    Calculate normalized power patterns.
    """

    patterns = power_pattern(
        weights,
        ANGLE_GRID,
        CONFIG,
    )

    return (
        patterns
        .detach()
        .cpu()
        .numpy()
    )


# =====================================================================
# Per-sample beam metrics
# =====================================================================

def evaluate_patterns(
    weights: torch.Tensor,
    ideal_patterns: np.ndarray,
    target_angles_deg: np.ndarray,
) -> dict:
    """
    Evaluate beam metrics for a batch of complex excitation weights.
    """

    patterns = calculate_patterns(
        weights
    )

    peak_indices = np.argmax(
        patterns,
        axis=1,
    )

    peak_angles = (
        ANGLE_GRID_DEG[
            peak_indices
        ]
    )

    pointing_errors = (
        peak_angles
        - target_angles_deg
    )

    correlations = []
    beamwidths = []
    psll_values = []

    for sample_index in range(
        patterns.shape[0]
    ):

        pattern = patterns[
            sample_index
        ]

        peak_index = (
            peak_indices[
                sample_index
            ]
        )

        correlations.append(
            beam_correlation(
                pattern,
                ideal_patterns[
                    sample_index
                ],
            )
        )

        beamwidths.append(
            beamwidth_3db(
                pattern,
                peak_index,
            )
        )

        psll_values.append(
            peak_sidelobe_level_db(
                pattern,
                peak_index,
            )
        )

    return {
        "patterns": patterns,
        "peak_angles": peak_angles,
        "pointing_errors": pointing_errors,
        "correlations": np.asarray(
            correlations,
            dtype=np.float64,
        ),
        "beamwidths": np.asarray(
            beamwidths,
            dtype=np.float64,
        ),
        "psll": np.asarray(
            psll_values,
            dtype=np.float64,
        ),
    }


# =====================================================================
# Summary reporting
# =====================================================================

def summarize_metrics(
    name: str,
    metrics: dict,
) -> None:
    """
    Print aggregate beam metrics.
    """

    print()
    print(
        name
    )
    print("-" * 70)

    print(
        "Mean pointing error : "
        f"{np.nanmean(np.abs(metrics['pointing_errors'])):.4f}°"
    )

    print(
        "Mean beam corr.     : "
        f"{np.nanmean(metrics['correlations']):.6f}"
    )

    print(
        "Mean -3 dB BW       : "
        f"{np.nanmean(metrics['beamwidths']):.4f}°"
    )

    print(
        "Mean PSLL           : "
        f"{np.nanmean(metrics['psll']):.4f} dB"
    )


# =====================================================================
# Main
# =====================================================================

def main() -> None:

    print("=" * 70)
    print(
        "PINN-Cal — Proper Beam Pattern Evaluation"
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

    # -----------------------------------------------------------------
    # Decode observed current
    # -----------------------------------------------------------------

    observed_current = (
        X[:, :64]
        + 1j * X[:, 64:128]
    )

    # -----------------------------------------------------------------
    # Decode ground-truth correction
    #
    # C_true = V_ideal / V_hardware
    # -----------------------------------------------------------------

    target_correction = (
        Y[:, :64]
        + 1j * Y[:, 64:]
    )

    # -----------------------------------------------------------------
    # Ideal steering excitation
    # -----------------------------------------------------------------

    steering_angles_tensor = (
        torch.tensor(
            steering_angles_deg,
            dtype=torch.float64,
            device=DEVICE,
        )
    )

    ideal_weights = (
        steering_vector(
            steering_angles_tensor,
            CONFIG,
        )
        .to(torch.complex128)
    )

    # -----------------------------------------------------------------
    # Recover TRUE hardware excitation.
    #
    # This uses the ground-truth correction only to reconstruct
    # the underlying physical test case:
    #
    #     V_hardware = V_ideal / C_true
    #
    # It is NOT a predicted calibration result.
    # -----------------------------------------------------------------

    target_correction_tensor = (
        torch.tensor(
            target_correction,
            dtype=torch.complex128,
            device=DEVICE,
        )
    )

    true_hardware = (
        ideal_weights
        /
        (
            target_correction_tensor
            + 1e-12
        )
    )

    # -----------------------------------------------------------------
    # Nominal coupling matrix
    # -----------------------------------------------------------------

    z_nom = (
        mutual_impedance_matrix(
            config=CONFIG,
            self_impedance_ohm=SELF_IMPEDANCE,
            nearest_neighbor_coupling=COUPLING,
            decay=DECAY,
            phase_step_rad=PHASE_STEP_RAD,
        )
        .to(torch.complex128)
        .to(DEVICE)
    )

    # -----------------------------------------------------------------
    # Observed current tensor
    # -----------------------------------------------------------------

    observed_current_tensor = (
        torch.tensor(
            observed_current,
            dtype=torch.complex128,
            device=DEVICE,
        )
    )

    # -----------------------------------------------------------------
    # Analytical calibration
    #
    # Reconstruct nominal effective voltage:
    #
    #     V_eff_est = Z_nom I_obs
    #
    # Then estimate hardware excitation:
    #
    #     V_hardware_est = V_eff_est / Z_self
    #
    # Then derive correction:
    #
    #     C_analytical = V_ideal / V_hardware_est
    # -----------------------------------------------------------------

    estimated_effective_voltage = (
        observed_current_tensor
        @ z_nom.T
    )

    estimated_hardware = (
        estimated_effective_voltage
        / SELF_IMPEDANCE
    )

    analytical_correction = (
        ideal_weights
        /
        (
            estimated_hardware
            + 1e-12
        )
    )

    # -----------------------------------------------------------------
    # Final PINN calibration
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

        pinn_prediction = model(
            X_tensor
        )

    pinn_correction = (
        torch.complex(
            pinn_prediction[:, :64]
            .to(torch.float64),

            pinn_prediction[:, 64:]
            .to(torch.float64),
        )
    )

    # -----------------------------------------------------------------
    # Apply predictions to the ACTUAL hardware.
    #
    # This is the physically meaningful post-calibration comparison:
    #
    #     V_corrected =
    #         V_hardware_true * C_pred
    #
    # The correction quality is therefore evaluated against the
    # actual hidden hardware state, including the effects represented
    # by the generated test sample.
    # -----------------------------------------------------------------

    uncalibrated_weights = (
        true_hardware
    )

    analytical_corrected_weights = (
        true_hardware
        * analytical_correction
    )

    pinn_corrected_weights = (
        true_hardware
        * pinn_correction
    )

    # -----------------------------------------------------------------
    # Ideal reference pattern
    # -----------------------------------------------------------------

    ideal_patterns = calculate_patterns(
        ideal_weights
    )

    # -----------------------------------------------------------------
    # Evaluate methods
    # -----------------------------------------------------------------

    ideal_metrics = evaluate_patterns(
        ideal_weights,
        ideal_patterns,
        steering_angles_deg,
    )

    uncalibrated_metrics = evaluate_patterns(
        uncalibrated_weights,
        ideal_patterns,
        steering_angles_deg,
    )

    analytical_metrics = evaluate_patterns(
        analytical_corrected_weights,
        ideal_patterns,
        steering_angles_deg,
    )

    pinn_metrics = evaluate_patterns(
        pinn_corrected_weights,
        ideal_patterns,
        steering_angles_deg,
    )

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------

    print(
        f"Test samples        : "
        f"{len(steering_angles_deg)}"
    )

    print(
        "Angular grid        : "
        f"{ANGLE_GRID_DEG[0]:.1f}° "
        f"to "
        f"{ANGLE_GRID_DEG[-1]:.1f}°"
    )

    print(
        f"Angular resolution  : "
        f"{ANGLE_GRID_DEG[1] - ANGLE_GRID_DEG[0]:.2f}°"
    )

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    summarize_metrics(
        "IDEAL REFERENCE",
        ideal_metrics,
    )

    summarize_metrics(
        "UNCALIBRATED HARDWARE",
        uncalibrated_metrics,
    )

    summarize_metrics(
        "ANALYTICAL CALIBRATION",
        analytical_metrics,
    )

    summarize_metrics(
        "PINN CALIBRATION",
        pinn_metrics,
    )

    # -----------------------------------------------------------------
    # Additional pointing statistics
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "Pointing Error Statistics"
    )
    print(
        "=" * 70
    )

    methods = {
        "Uncalibrated":
            uncalibrated_metrics,

        "Analytical":
            analytical_metrics,

        "PINN":
            pinn_metrics,
    }

    for name, metrics in methods.items():

        errors = np.abs(
            metrics[
                "pointing_errors"
            ]
        )

        print(
            f"{name:15s} | "
            f"Mean={np.mean(errors):8.4f}° | "
            f"Median={np.median(errors):8.4f}° | "
            f"Max={np.max(errors):8.4f}°"
        )

    # -----------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "Beam evaluation complete."
    )

    print(
        "Predicted corrections were evaluated against the "
        "true underlying hardware excitations."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
