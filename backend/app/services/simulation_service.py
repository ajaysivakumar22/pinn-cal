"""
PINN-Cal Backend — Simulation Service

Runs the full physics simulation chain and returns structured results.

Physical chain (per calibration_dataset.py):
    ideal steering excitation
        → hardware gain/phase impairment
        → hidden true coupling matrix Z_true
        → true coupled current
        → complex measurement noise
        → observed current

Then applies analytical correction (analytical_baseline.py pattern)
and PINN correction (if compatible).

IMPORTANT:
    All physics is delegated to physics_core.py / calibration_dataset.py.
    No physics equations are written here.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from backend.app.scientific.adapter import (
    SCIENTIFIC_CORE_AVAILABLE,
    SCIENTIFIC_CORE_ERROR,
    DEFAULT_CONFIG,
    ArrayConfig,
    steering_vector,
    generate_hardware_errors,
    apply_hardware_errors,
    mutual_impedance_matrix,
    generate_true_coupling_matrix,
    power_pattern,
)
from backend.app.core.config import (
    SELF_IMPEDANCE,
    NEAREST_NEIGHBOR_COUPLING,
    DECAY,
    PHASE_STEP_RAD,
)
from backend.app.services.beam_service import compute_all_beam_patterns
from backend.app.services.model_service import ModelService
from backend.app.schemas.simulation import (
    SimulationRequest,
    ElementState,
)


def _check_core() -> None:
    if not SCIENTIFIC_CORE_AVAILABLE:
        raise RuntimeError(
            f"Scientific core unavailable: {SCIENTIFIC_CORE_ERROR}"
        )


# ---------------------------------------------------------------------------
# Metric helpers — match analytical_baseline.py implementations exactly
# ---------------------------------------------------------------------------

def _nmse_db(predicted: np.ndarray, target: np.ndarray) -> float:
    num = np.sum(np.abs(predicted - target) ** 2)
    den = np.sum(np.abs(target) ** 2)
    return float(10.0 * np.log10(num / (den + 1e-30)))


def _phase_error_deg(predicted: np.ndarray, target: np.ndarray) -> float:
    diff = np.angle(predicted * np.conj(target))
    return float(np.degrees(np.mean(np.abs(diff))))


def _gain_error_db(predicted: np.ndarray, target: np.ndarray) -> float:
    pred_db = 20.0 * np.log10(np.maximum(np.abs(predicted), 1e-12))
    tgt_db = 20.0 * np.log10(np.maximum(np.abs(target), 1e-12))
    return float(np.mean(np.abs(pred_db - tgt_db)))


def _relative_error(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - target) / (np.abs(target) + 1e-12)))


def _current_relative_error(
    predicted: torch.Tensor, target: torch.Tensor
) -> float:
    err = torch.abs(predicted - target) / (torch.abs(target) + 1e-12)
    return float(torch.mean(err).item())


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_simulation(req: SimulationRequest) -> dict:
    """
    Execute the complete simulation + calibration pipeline for one request.

    Returns a dict with keys: elements, beam, calibration, metrics.
    """
    _check_core()

    # ------------------------------------------------------------------
    # Array configuration
    # ------------------------------------------------------------------
    from physics_core import ArrayConfig  # imported via sys.path set by adapter

    config = ArrayConfig(
        num_elements=req.num_elements,
        frequency_hz=req.frequency_hz,
        element_spacing_lambda=req.element_spacing_lambda,
    )

    pinn_available = req.pinn_compatible

    # ------------------------------------------------------------------
    # Nominal coupling matrix Z_nom
    # ------------------------------------------------------------------
    z_nom = mutual_impedance_matrix(
        config=config,
        self_impedance_ohm=SELF_IMPEDANCE,
        nearest_neighbor_coupling=NEAREST_NEIGHBOR_COUPLING,
        decay=DECAY,
        phase_step_rad=PHASE_STEP_RAD,
    ).to(torch.complex128)

    z_self_nom = z_nom.diagonal().mean()

    # ------------------------------------------------------------------
    # Hidden true coupling matrix Z_true
    # ------------------------------------------------------------------
    z_true = generate_true_coupling_matrix(
        config=config,
        mismatch_fraction=req.coupling_mismatch_fraction,
        seed=req.seed + 2_000_000,
        nominal_matrix=z_nom,
    ).to(torch.complex128)

    z_self_true = z_true.diagonal().mean()

    # ------------------------------------------------------------------
    # Ideal steering excitation
    # ------------------------------------------------------------------
    angle_tensor = torch.tensor(
        [req.target_angle_deg], dtype=torch.float64
    )
    v_ideal = steering_vector(angle_tensor, config).squeeze(0).to(torch.complex128)

    # ------------------------------------------------------------------
    # Hardware impairments
    # ------------------------------------------------------------------
    phase_errors_deg, gain_errors_db = generate_hardware_errors(
        num_elements=req.num_elements,
        phase_sigma_deg=req.phase_sigma_deg,
        gain_sigma_db=req.gain_sigma_db,
        seed=req.seed,
    )

    v_hardware = apply_hardware_errors(
        v_ideal, phase_errors_deg, gain_errors_db
    ).to(torch.complex128)

    # ------------------------------------------------------------------
    # True physical coupled current
    #   Z_true I_true = V_hardware * Z_self_true
    # ------------------------------------------------------------------
    i_true = torch.linalg.solve(
        z_true,
        v_hardware * z_self_true,
    )

    # ------------------------------------------------------------------
    # Observed current (with noise)
    # ------------------------------------------------------------------
    snr_linear = 10.0 ** (req.snr_db / 10.0)
    signal_power = float(torch.mean(torch.abs(i_true) ** 2))
    noise_power = signal_power / snr_linear
    noise_std = math.sqrt(noise_power / 2.0)

    rng = torch.Generator()
    rng.manual_seed(req.seed + 1_000_000)

    noise = torch.complex(
        torch.randn(req.num_elements, generator=rng, dtype=torch.float64) * noise_std,
        torch.randn(req.num_elements, generator=rng, dtype=torch.float64) * noise_std,
    )

    i_observed = i_true + noise

    # ------------------------------------------------------------------
    # Analytical correction (matches analytical_baseline.py)
    #   V_eff_nom = Z_nom I_obs
    #   V_hardware_est = V_eff_nom / Z_self_nom
    #   C_analytical = V_ideal / V_hardware_est
    # ------------------------------------------------------------------
    v_eff_nom = z_nom @ i_observed
    v_hardware_est = v_eff_nom / SELF_IMPEDANCE
    c_analytical = v_ideal / (v_hardware_est + 1e-30)

    # ------------------------------------------------------------------
    # Ground-truth correction
    #   C_true = V_ideal / V_hardware
    # ------------------------------------------------------------------
    c_true = v_ideal / (v_hardware + 1e-30)

    # ------------------------------------------------------------------
    # PINN correction
    # ------------------------------------------------------------------
    c_pinn: Optional[np.ndarray] = None
    pinn_error_msg: Optional[str] = None

    if pinn_available:
        model_svc = ModelService.get()
        if model_svc.loaded:
            angle_rad = math.radians(req.target_angle_deg)
            x_input = np.concatenate([
                i_observed.real.numpy().astype(np.float32),
                i_observed.imag.numpy().astype(np.float32),
                np.array([math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float32),
            ])[np.newaxis, :]  # shape (1, 130)

            pred = model_svc.predict(x_input)  # shape (1, 128)
            c_pinn = pred[0, :64] + 1j * pred[0, 64:]
        else:
            pinn_error_msg = model_svc.load_error or "PINN model not loaded"
            pinn_available = False
    else:
        pinn_error_msg = (
            "PINN calibration unavailable — "
            f"checkpoint trained for 64 elements @ 28 GHz; "
            f"current config: {req.num_elements} elements @ "
            f"{req.frequency_hz / 1e9:.1f} GHz"
        )

    # ------------------------------------------------------------------
    # Build compatibility message
    # ------------------------------------------------------------------
    if req.pinn_compatible:
        pinn_compat_msg = (
            "PINN calibration available — "
            "64 elements at 28 GHz matches trained checkpoint."
        )
    else:
        pinn_compat_msg = pinn_error_msg or "PINN calibration unavailable."

    # ------------------------------------------------------------------
    # Per-element states for antenna visualization
    # ------------------------------------------------------------------
    elements: list[ElementState] = []

    c_analytical_np = c_analytical.numpy()
    c_true_np = c_true.numpy()
    i_obs_np = i_observed.numpy()
    v_ideal_np = v_ideal.numpy()
    v_hw_np = v_hardware.numpy()
    phase_deg_np = phase_errors_deg.numpy()
    gain_db_np = gain_errors_db.numpy()

    for n in range(req.num_elements):
        vi = v_ideal_np[n]
        vh = v_hw_np[n]
        io = i_obs_np[n]
        ca = c_analytical_np[n]

        true_ph = float(phase_deg_np[n])
        true_gn = float(gain_db_np[n])

        # Analytical RF diagnostics:
        # C = V_ideal / V_hw => angle(C) = -delta_phi => est_phi = -angle(C)
        # |C| = 1/|H| => 20*log10(|C|) = -delta_g => est_gain = -20*log10(|C|)
        ca_mag = float(abs(ca))
        ca_phase = float(math.degrees(math.atan2(ca.imag, ca.real)))
        ana_est_phase = -ca_phase
        ana_est_gain = -20.0 * math.log10(max(ca_mag, 1e-12))
        ana_res_phase = ((true_ph - ana_est_phase + 180.0) % 360.0) - 180.0
        ana_res_gain = true_gn - ana_est_gain

        pinn_cr = pinn_ci = pinn_cm = pinn_cp = None
        pinn_est_phase = pinn_est_gain = pinn_res_phase = pinn_res_gain = None
        comp_phase = comp_gain = None

        if c_pinn is not None:
            cp = c_pinn[n]
            pinn_cr = float(cp.real)
            pinn_ci = float(cp.imag)
            pinn_cm = float(abs(cp))
            pinn_cp = float(math.degrees(math.atan2(cp.imag, cp.real)))

            # Derived PINN RF impairment estimation:
            pinn_est_phase = -pinn_cp
            pinn_est_gain = -20.0 * math.log10(max(pinn_cm, 1e-12))
            pinn_res_phase = ((true_ph - pinn_est_phase + 180.0) % 360.0) - 180.0
            pinn_res_gain = true_gn - pinn_est_gain

            # Compensation applied is the inverse of estimated impairment
            comp_phase = pinn_cp
            comp_gain = 20.0 * math.log10(max(pinn_cm, 1e-12))
        else:
            comp_phase = ca_phase
            comp_gain = 20.0 * math.log10(max(ca_mag, 1e-12))

        elements.append(ElementState(
            element_index=n,
            ideal_real=float(vi.real),
            ideal_imag=float(vi.imag),
            ideal_magnitude=float(abs(vi)),
            ideal_phase_deg=float(math.degrees(math.atan2(vi.imag, vi.real))),
            hardware_real=float(vh.real),
            hardware_imag=float(vh.imag),
            hardware_magnitude=float(abs(vh)),
            hardware_phase_deg=float(math.degrees(math.atan2(vh.imag, vh.real))),
            gain_error_db=true_gn,
            phase_error_deg=true_ph,
            true_gain_error_db=true_gn,
            true_phase_error_deg=true_ph,
            observed_real=float(io.real),
            observed_imag=float(io.imag),
            analytical_correction_real=float(ca.real),
            analytical_correction_imag=float(ca.imag),
            analytical_correction_magnitude=ca_mag,
            analytical_correction_phase_deg=ca_phase,
            analytical_estimated_phase_error_deg=ana_est_phase,
            analytical_estimated_gain_error_db=ana_est_gain,
            analytical_phase_residual_deg=ana_res_phase,
            analytical_gain_residual_db=ana_res_gain,
            pinn_correction_real=pinn_cr,
            pinn_correction_imag=pinn_ci,
            pinn_correction_magnitude=pinn_cm,
            pinn_correction_phase_deg=pinn_cp,
            pinn_estimated_phase_error_deg=pinn_est_phase,
            pinn_estimated_gain_error_db=pinn_est_gain,
            phase_error_residual_deg=pinn_res_phase,
            gain_error_residual_db=pinn_res_gain,
            compensation_phase_deg=comp_phase,
            compensation_gain_db=comp_gain,
        ))

    # ------------------------------------------------------------------
    # Calibration metrics
    # ------------------------------------------------------------------
    c_true_np = c_true_np  # shape (64,) complex

    # Analytical metrics
    ana_nmse = _nmse_db(c_analytical_np, c_true_np)
    ana_phase = _phase_error_deg(c_analytical_np, c_true_np)
    ana_gain = _gain_error_db(c_analytical_np, c_true_np)
    ana_rel = 100.0 * _relative_error(c_analytical_np, c_true_np)

    # Apply corrections through physics to compute current residuals
    # Analytical corrected current
    v_obs_nom = z_nom @ i_observed
    v_corr_ana = c_analytical * v_obs_nom
    i_corr_ana = torch.linalg.solve(z_nom, v_corr_ana)

    # Ideal coupled current under nominal model
    rhs_ideal = v_ideal * z_self_nom
    i_ideal_nom = torch.linalg.solve(z_nom, rhs_ideal)

    ana_current_residual = 100.0 * _current_relative_error(i_corr_ana, i_ideal_nom)

    ana_metrics = {
        "nmse_db": ana_nmse,
        "mean_phase_error_deg": ana_phase,
        "mean_gain_error_db": ana_gain,
        "mean_relative_error_pct": ana_rel,
        "current_residual_pct": ana_current_residual,
    }

    pinn_metrics = None
    if c_pinn is not None:
        c_pinn_tensor = torch.tensor(c_pinn, dtype=torch.complex128)
        pinn_nmse = _nmse_db(c_pinn, c_true_np)
        pinn_phase = _phase_error_deg(c_pinn, c_true_np)
        pinn_gain = _gain_error_db(c_pinn, c_true_np)
        pinn_rel = 100.0 * _relative_error(c_pinn, c_true_np)

        v_corr_pinn = c_pinn_tensor * v_obs_nom
        i_corr_pinn = torch.linalg.solve(z_nom, v_corr_pinn)
        pinn_current_residual = 100.0 * _current_relative_error(i_corr_pinn, i_ideal_nom)

        pinn_metrics = {
            "nmse_db": pinn_nmse,
            "mean_phase_error_deg": pinn_phase,
            "mean_gain_error_db": pinn_gain,
            "mean_relative_error_pct": pinn_rel,
            "current_residual_pct": pinn_current_residual,
        }

    # ------------------------------------------------------------------
    # Beam patterns
    # ------------------------------------------------------------------
    # Reconstruct true hardware excitation for beam evaluation
    c_true_tensor = torch.tensor(c_true_np, dtype=torch.complex128)
    true_hw_weights = v_ideal / (c_true_tensor + 1e-30)

    ideal_weights = v_ideal
    uncal_weights = true_hw_weights
    ana_corrected_weights = true_hw_weights * c_analytical
    pinn_corrected_weights = (
        true_hw_weights * torch.tensor(c_pinn, dtype=torch.complex128)
        if c_pinn is not None else None
    )

    # Use 64-element config if PINN compatible; otherwise general config
    beam_config = DEFAULT_CONFIG if req.pinn_compatible else config

    beam_result = compute_all_beam_patterns(
        ideal_weights=ideal_weights,
        uncalibrated_weights=uncal_weights,
        analytical_weights=ana_corrected_weights,
        pinn_weights=pinn_corrected_weights,
        target_angle_deg=req.target_angle_deg,
        config=beam_config,
    )

    # Summary scalar metrics (from analytical beam for the dashboard right panel)
    ana_beam = beam_result["analytical"]
    ideal_beam = beam_result["ideal"]

    return {
        "pinn_available": pinn_available,
        "pinn_compatibility_message": pinn_compat_msg,
        "elements": elements,
        "beam": beam_result,
        "calibration": {
            "analytical": ana_metrics,
            "pinn": pinn_metrics,
            "pinn_available": pinn_available,
        },
        "metrics": {
            "target_angle_deg": req.target_angle_deg,
            "beam_peak_angle_deg": ana_beam["peak_angle_deg"],
            "pointing_error_deg": ana_beam["pointing_error_deg"],
            "beam_correlation": ana_beam["beam_correlation"],
            "beamwidth_3db_deg": ana_beam["beamwidth_3db_deg"],
            "psll_db": ana_beam["psll_db"],
            "analytical_nmse_db": ana_nmse,
            "pinn_nmse_db": pinn_metrics["nmse_db"] if pinn_metrics else None,
        },
    }
