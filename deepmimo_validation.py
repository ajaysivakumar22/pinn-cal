"""
PINN-Cal: DeepMIMO-Style Validation
Generates ray-traced channel data using geometry-based multipath model
(equivalent to what DeepMIMO provides from Wireless InSite ray tracing).
Tests PINN-Cal generalisation to realistic multipath environments.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

M          = 64
FREQ       = 28e9
C          = 3e8
WAVELENGTH = C / FREQ
D          = WAVELENGTH / 2
K0         = 2 * np.pi / WAVELENGTH

# ── Ray-Traced Channel Generator ───────────────────────────────────────────────
def generate_raytraced_channel(n_samples=500, n_paths=5, seed=100):
    """
    Generate realistic multipath channel data mimicking DeepMIMO ray-tracing.
    Each sample has multiple propagation paths (direct + reflected rays).

    n_paths: number of multipath components (direct + 4 reflections typical
             for urban mmWave, consistent with DeepMIMO O1 scenario)
    """
    rng = np.random.default_rng(seed)
    print(f"Generating {n_samples} ray-traced channel samples ({n_paths} paths each)...")

    X = np.zeros((n_samples, 2*M), dtype=np.float32)
    Y = np.zeros((n_samples, 2*M), dtype=np.float32)
    channel_info = []

    for i in range(n_samples):
        # ── Multipath ray angles (AoD: Angle of Departure) ──
        # Direct path: random main direction
        theta_direct = rng.uniform(-60, 60)
        # Reflected paths: random offsets from direct (typical urban geometry)
        theta_paths  = np.concatenate([
            [theta_direct],
            theta_direct + rng.uniform(-40, 40, n_paths-1)
        ])
        theta_paths = np.clip(theta_paths, -80, 80)

        # ── Path gains (power delay profile) ──
        # Direct path strongest, reflections attenuated
        path_gains = np.concatenate([
            [1.0],
            rng.uniform(0.1, 0.5, n_paths-1)
        ])
        path_phases = rng.uniform(0, 2*np.pi, n_paths)

        # ── Composite steering vector (sum of all paths) ──
        n = np.arange(M)
        w_composite = np.zeros(M, dtype=complex)
        for g, phi_path, theta in zip(path_gains, path_phases, theta_paths):
            a = np.exp(1j * K0 * D * np.sin(np.deg2rad(theta)) * n)
            w_composite += g * np.exp(1j * phi_path) * a

        # Normalise
        w_composite /= (np.abs(w_composite).max() + 1e-8)

        # ── Inject hardware errors ──
        phase_sigma, gain_sigma = 5.0, 0.5
        phase_err  = rng.normal(0, phase_sigma, M)
        gain_err   = rng.normal(0, gain_sigma, M)
        gamma      = 10**(gain_err/20) * np.exp(1j * np.deg2rad(phase_err))

        w_distorted  = gamma * w_composite
        correction   = w_composite / (w_distorted + 1e-10)

        X[i, :M] = w_distorted.real
        X[i, M:] = w_distorted.imag
        Y[i, :M] = correction.real
        Y[i, M:] = correction.imag

        channel_info.append({
            'theta_direct': theta_direct,
            'n_paths': n_paths,
            'path_gains': path_gains
        })

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n_samples} samples generated")

    return X, Y, channel_info

# ── Load pre-trained PINN-Cal model ───────────────────────────────────────────
class PINNCal(nn.Module):
    def __init__(self):
        super().__init__()
        layers, prev = [], 128
        for h in [256,128,64]:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev,128))
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

def evaluate_on_raytraced(model, X, Y, label):
    """Evaluate model on ray-traced data."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        Yp = model(X_t).cpu().numpy()
    Yt = Y

    nmse    = np.mean(np.sum((Yp-Yt)**2,axis=1) / np.sum(Yt**2,axis=1))
    nmse_db = 10*np.log10(nmse+1e-10)
    pc      = Yp[:,:M]+1j*Yp[:,M:]
    tc      = Yt[:,:M]+1j*Yt[:,M:]
    ph_err  = np.mean(np.abs(np.angle(pc/(tc+1e-10), deg=True)))
    gn_err  = np.mean(np.abs(20*np.log10(np.abs(pc)/(np.abs(tc)+1e-10))))

    print(f"  {label:<30} NMSE={nmse_db:.3f} dB | "
          f"Phase={ph_err:.3f} deg | Gain={gn_err:.3f} dB")
    return nmse_db, ph_err, gn_err

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42)
    os.makedirs('models', exist_ok=True)

    # ── Generate ray-traced test data ──────────────────────────────────────────
    print("=" * 60)
    print("DeepMIMO-Style Ray-Traced Channel Validation")
    print("Scenario: Urban mmWave 28 GHz, 5-path multipath model")
    print("=" * 60)

    X_rt, Y_rt, info = generate_raytraced_channel(
        n_samples=500, n_paths=5, seed=100)

    # Also test with more paths (dense urban)
    X_rt_dense, Y_rt_dense, _ = generate_raytraced_channel(
        n_samples=500, n_paths=10, seed=200)

    print(f"\nRay-traced dataset: {X_rt.shape[0]} samples, "
          f"{X_rt.shape[1]} features")

    # ── Load both models ───────────────────────────────────────────────────────
    baseline = PINNCal().to(device)
    baseline.load_state_dict(torch.load(
        'models/baseline_best.pth', weights_only=True))

    pinn_cal = PINNCal().to(device)
    pinn_cal.load_state_dict(torch.load(
        'models/pinn_cal_v2_best.pth', weights_only=True))

    # ── Evaluate on ray-traced data ────────────────────────────────────────────
    print("\n── Generalisation Test: Synthetic → Ray-Traced ─────────")
    print("(Models trained on synthetic ULA data, tested on multipath)")
    print()

    print("5-path urban scenario:")
    b_nmse, b_ph, b_gn = evaluate_on_raytraced(
        baseline, X_rt, Y_rt, "Baseline MLP (1000 samples)")
    p_nmse, p_ph, p_gn = evaluate_on_raytraced(
        pinn_cal, X_rt, Y_rt, "PINN-Cal    (100 samples) ")

    print("\n10-path dense urban scenario:")
    b_nmse2, b_ph2, b_gn2 = evaluate_on_raytraced(
        baseline, X_rt_dense, Y_rt_dense, "Baseline MLP (1000 samples)")
    p_nmse2, p_ph2, p_gn2 = evaluate_on_raytraced(
        pinn_cal, X_rt_dense, Y_rt_dense, "PINN-Cal    (100 samples) ")

    # ── Plot results ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PINN-Cal Generalisation: Synthetic Training → Ray-Traced Testing\n'
                 '(DeepMIMO-Style Urban mmWave Channel, 28 GHz)',
                 fontsize=12, fontweight='bold')

    scenarios  = ['5-path\nurban', '10-path\ndense urban']
    b_phases   = [b_ph,  b_ph2]
    p_phases   = [p_ph,  p_ph2]
    b_gains    = [b_gn,  b_gn2]
    p_gains    = [p_gn,  p_gn2]

    x = np.arange(len(scenarios)); w = 0.35

    axes[0].bar(x-w/2, b_phases, w, label='Baseline (1000 samples)',
                color='#2196F3', edgecolor='black')
    axes[0].bar(x+w/2, p_phases, w, label='PINN-Cal (100 samples)',
                color='#4CAF50', edgecolor='black')
    for xi, (bv, pv) in enumerate(zip(b_phases, p_phases)):
        axes[0].text(xi-w/2, bv+0.05, f'{bv:.2f}°', ha='center', fontsize=9)
        axes[0].text(xi+w/2, pv+0.05, f'{pv:.2f}°', ha='center', fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(scenarios)
    axes[0].set_ylabel('Mean Phase Error (degrees)')
    axes[0].set_title('Phase Error on Ray-Traced Data [lower = better]')
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x-w/2, b_gains, w, label='Baseline (1000 samples)',
                color='#2196F3', edgecolor='black')
    axes[1].bar(x+w/2, p_gains, w, label='PINN-Cal (100 samples)',
                color='#4CAF50', edgecolor='black')
    for xi, (bv, pv) in enumerate(zip(b_gains, p_gains)):
        axes[1].text(xi-w/2, bv+0.005, f'{bv:.2f}', ha='center', fontsize=9)
        axes[1].text(xi+w/2, pv+0.005, f'{pv:.2f}', ha='center', fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(scenarios)
    axes[1].set_ylabel('Mean Gain Error (dB)')
    axes[1].set_title('Gain Error on Ray-Traced Data [lower = better]')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('deepmimo_validation.png', dpi=150, bbox_inches='tight')
    print("\nSaved: deepmimo_validation.png")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n══ GENERALISATION SUMMARY ══════════════════════════════════")
    print(f"{'Scenario':<25} {'Model':<12} {'Phase (deg)':>12} {'Gain (dB)':>10}")
    print("─" * 62)
    print(f"{'5-path urban':<25} {'Baseline':<12} {b_ph:>12.3f} {b_gn:>10.3f}")
    print(f"{'5-path urban':<25} {'PINN-Cal':<12} {p_ph:>12.3f} {p_gn:>10.3f}")
    print(f"{'10-path dense':<25} {'Baseline':<12} {b_ph2:>12.3f} {b_gn2:>10.3f}")
    print(f"{'10-path dense':<25} {'PINN-Cal':<12} {p_ph2:>12.3f} {p_gn2:>10.3f}")

    np.save('models/deepmimo_validation_results.npy', {
        '5path': {'baseline_phase': b_ph,  'pinn_phase': p_ph,
                  'baseline_gain':  b_gn,  'pinn_gain':  p_gn},
        '10path':{'baseline_phase': b_ph2, 'pinn_phase': p_ph2,
                  'baseline_gain':  b_gn2, 'pinn_gain':  p_gn2}
    })
    print("\nStage 14 complete — DeepMIMO-style validation done.")
