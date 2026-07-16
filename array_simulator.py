"""
PINN-Cal: Antenna Array Simulator
Simulates a 64-element ULA at 28 GHz with realistic hardware errors.
Generates dataset of (distorted_pattern, correction_matrix) pairs.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Constants ──────────────────────────────────────────────────────────────────
FREQ        = 28e9          # 28 GHz (mmWave)
C           = 3e8           # Speed of light
WAVELENGTH  = C / FREQ      # ~0.0107 m
M           = 64            # Number of antenna elements
D           = WAVELENGTH/2  # Half-wavelength spacing
K           = 2*np.pi / WAVELENGTH  # Wavenumber
ANGLES      = np.linspace(-90, 90, 360)  # Scan angles in degrees
ANGLES_RAD  = np.deg2rad(ANGLES)

# ── Steering Vector ────────────────────────────────────────────────────────────
def steering_vector(theta_rad, m=M, d=D):
    """Ideal steering vector for direction theta (radians)."""
    n = np.arange(m)
    return np.exp(1j * K * d * np.sin(theta_rad) * n)

# ── Ideal Beam Pattern ─────────────────────────────────────────────────────────
def ideal_beam_pattern(theta_0_deg):
    """Compute ideal beam pattern when steering toward theta_0."""
    theta_0 = np.deg2rad(theta_0_deg)
    w = steering_vector(theta_0)
    pattern = np.array([
        abs(np.dot(np.conj(w), steering_vector(t)))**2
        for t in ANGLES_RAD
    ])
    return pattern / pattern.max()

# ── Error Injection ────────────────────────────────────────────────────────────
def inject_errors(phase_sigma_deg=5.0, gain_sigma_db=0.5, seed=None):
    """
    Generate realistic hardware errors for each antenna element.
    Returns error matrix Gamma (diagonal, complex).
    """
    rng = np.random.default_rng(seed)
    phase_errors = rng.normal(0, phase_sigma_deg, M)
    gain_errors  = rng.normal(0, gain_sigma_db,  M)
    phase_rad    = np.deg2rad(phase_errors)
    gain_linear  = 10**(gain_errors / 20)
    gamma        = gain_linear * np.exp(1j * phase_rad)
    return gamma, phase_errors, gain_errors

# ── Distorted Beam Pattern ─────────────────────────────────────────────────────
def distorted_beam_pattern(theta_0_deg, gamma):
    """Compute beam pattern with hardware errors applied."""
    theta_0  = np.deg2rad(theta_0_deg)
    w_ideal  = steering_vector(theta_0)
    w_actual = gamma * w_ideal
    pattern  = np.array([
        abs(np.dot(np.conj(w_actual), steering_vector(t)))**2
        for t in ANGLES_RAD
    ])
    return pattern / pattern.max()

# ── Dataset Generation ─────────────────────────────────────────────────────────
def generate_dataset(n_samples=1200, phase_sigma=5.0, gain_sigma=0.5):
    """
    Generate dataset of (distorted_weights, correction) pairs.
    Steers randomly across [-60, 60] degrees.
    """
    print(f"Generating {n_samples} samples...")
    steering_angles = np.random.uniform(-60, 60, n_samples)

    X = np.zeros((n_samples, 2 * M), dtype=np.float32)
    Y = np.zeros((n_samples, 2 * M), dtype=np.float32)
    patterns_distorted = np.zeros((n_samples, len(ANGLES)), dtype=np.float32)
    patterns_ideal     = np.zeros((n_samples, len(ANGLES)), dtype=np.float32)
    gammas             = np.zeros((n_samples, M),           dtype=np.complex64)

    for i in range(n_samples):
        theta   = steering_angles[i]
        gamma, _, _ = inject_errors(phase_sigma, gain_sigma, seed=i)

        w_ideal  = steering_vector(np.deg2rad(theta))
        w_actual = gamma * w_ideal
        correction = w_ideal / w_actual

        X[i, :M] = w_actual.real
        X[i, M:] = w_actual.imag
        Y[i, :M] = correction.real
        Y[i, M:] = correction.imag

        patterns_distorted[i] = distorted_beam_pattern(theta, gamma)
        patterns_ideal[i]     = ideal_beam_pattern(theta)
        gammas[i]             = gamma

        if (i+1) % 200 == 0:
            print(f"  {i+1}/{n_samples} samples generated")

    return {
        'X': X, 'Y': Y,
        'patterns_distorted': patterns_distorted,
        'patterns_ideal':     patterns_ideal,
        'gammas':             gammas,
        'steering_angles':    steering_angles,
        'angles':             ANGLES
    }

# ── Quick Sanity Plot ──────────────────────────────────────────────────────────
def plot_beam_comparison(theta_0_deg=30, seed=42):
    """Plot ideal vs distorted beam pattern for a single example."""
    gamma, phase_err, gain_err = inject_errors(seed=seed)
    ideal     = ideal_beam_pattern(theta_0_deg)
    distorted = distorted_beam_pattern(theta_0_deg, gamma)

    print(f"\nBeam steering toward {theta_0_deg} degrees")
    print(f"Phase errors (deg): mean={phase_err.mean():.2f}, std={phase_err.std():.2f}")
    print(f"Gain errors  (dB):  mean={gain_err.mean():.2f},  std={gain_err.std():.2f}")

    ideal_dB     = 10 * np.log10(ideal     + 1e-10)
    distorted_dB = 10 * np.log10(distorted + 1e-10)

    plt.figure(figsize=(10, 5))
    plt.plot(ANGLES, ideal_dB,     'b-',  linewidth=2, label='Ideal beam')
    plt.plot(ANGLES, distorted_dB, 'r--', linewidth=2, label='Distorted beam (errors)')
    plt.axvline(x=theta_0_deg, color='g', linestyle=':', label=f'Target: {theta_0_deg} deg')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Power (dB)')
    plt.title('64-Element ULA at 28 GHz — Beam Pattern Comparison\nPhase sigma=5 deg, Gain sigma=0.5 dB')
    plt.legend()
    plt.ylim([-40, 5])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('beam_comparison.png', dpi=150)
    print("Plot saved: beam_comparison.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)

    plot_beam_comparison(theta_0_deg=30, seed=42)

    data = generate_dataset(n_samples=1200)

    train = {k: v[:1000] for k, v in data.items() if isinstance(v, np.ndarray)}
    test  = {k: v[1000:] for k, v in data.items() if isinstance(v, np.ndarray)}

    os.makedirs('data', exist_ok=True)
    np.save('data/train.npy', train)
    np.save('data/test.npy',  test)
    np.save('data/full.npy',  data)

    print(f"\nDataset saved to data/")
    print(f"  Train: {train['X'].shape[0]} samples, X={train['X'].shape}, Y={train['Y'].shape}")
    print(f"  Test:  {test['X'].shape[0]} samples,  X={test['X'].shape},  Y={test['Y'].shape}")
    print("\nStage 7 complete.")
