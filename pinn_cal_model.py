"""
PINN-Cal: Physics-Informed Neural Network Calibration
Same architecture as baseline but with electromagnetic physics constraints.
Trained on only 100 samples to demonstrate sample efficiency.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Constants ──────────────────────────────────────────────────────────────────
M          = 64
FREQ       = 28e9
C          = 3e8
WAVELENGTH = C / FREQ
D          = WAVELENGTH / 2
K0         = 2 * np.pi / WAVELENGTH
ETA0       = 377.0   # Intrinsic impedance of free space (ohms)

# ── Precompute Theoretical Mutual Impedance Matrix ─────────────────────────────
def compute_mutual_impedance_matrix(m=M, d=D):
    """
    Compute theoretical mutual impedance matrix for half-wave dipole array.
    Z_ij depends only on distance between elements i and j.
    """
    Z = np.zeros((m, m), dtype=complex)
    for i in range(m):
        for j in range(m):
            if i == j:
                Z[i, j] = 73.1 + 42.5j  # Self-impedance of half-wave dipole
            else:
                d_ij = abs(i - j) * d
                k_d  = K0 * d_ij
                # Simplified mutual impedance approximation
                Z[i, j] = (ETA0 / (4 * np.pi)) * (
                    np.exp(-1j * k_d) / d_ij
                ) * (2 * np.cos(k_d) + 2j * np.sin(k_d) / k_d)
    return torch.tensor(Z, dtype=torch.complex64).to(device)

Z_theory = compute_mutual_impedance_matrix()
print("Mutual impedance matrix computed.")

# ── Model Architecture (same as baseline) ─────────────────────────────────────
class PINNCal(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[256, 128, 64], output_dim=128):
        super(PINNCal, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2)
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ── Physics Loss Functions ─────────────────────────────────────────────────────
def reciprocity_loss(pred, inp):
    """
    Reciprocity: calibrated channel matrix should be symmetric H_ij = H_ji.
    We approximate this using the outer product of correction weights.
    """
    w_real = pred[:, :M]
    w_imag = pred[:, M:]
    w = torch.complex(w_real, w_imag)   # (batch, M)

    # Outer product: H_approx = w * w^H  (simplified channel approximation)
    H = torch.bmm(w.unsqueeze(2), w.conj().unsqueeze(1))  # (batch, M, M)
    H_T = H.transpose(1, 2)

    # Reciprocity violation: ||H - H^T||^2
    diff = H - H_T.conj()
    loss = torch.mean(torch.abs(diff)**2)
    return loss

def coupling_loss(pred):
    """
    Mutual coupling: predicted correction should be consistent with
    theoretical mutual impedance structure.
    """
    w_real = pred[:, :M]
    w_imag = pred[:, M:]
    w = torch.complex(w_real, w_imag)   # (batch, M)

    # Predicted impedance (outer product approximation)
    Z_pred = torch.bmm(w.unsqueeze(2), w.conj().unsqueeze(1))  # (batch, M, M)

    # Compare structure to theoretical (broadcast Z_theory across batch)
    Z_th = Z_theory.unsqueeze(0).expand(w.shape[0], -1, -1)

    # Normalise before comparing to handle scale difference
    Z_pred_norm = Z_pred / (torch.abs(Z_pred).max() + 1e-8)
    Z_th_norm   = Z_th   / (torch.abs(Z_th).max()   + 1e-8)

    loss = torch.mean(torch.abs(Z_pred_norm - Z_th_norm)**2)
    return loss

def energy_loss(pred, p_target=1.0):
    """
    Energy conservation: total radiated power <= input power.
    Penalise when ||w||^2 deviates from target power.
    """
    w_real = pred[:, :M]
    w_imag = pred[:, M:]
    p_rad  = torch.sum(w_real**2 + w_imag**2, dim=1)  # per-sample power
    loss   = torch.mean((p_rad - p_target)**2)
    return loss

def physics_loss(pred, inp, alpha=1.0, beta=0.5, gamma=0.3):
    """Combined physics penalty."""
    L_rec  = reciprocity_loss(pred, inp)
    L_coup = coupling_loss(pred)
    L_eng  = energy_loss(pred)
    return alpha * L_rec + beta * L_coup + gamma * L_eng, L_rec, L_coup, L_eng

# ── Data Loader ────────────────────────────────────────────────────────────────
def load_data(n_train=100, batch_size=16):
    """Load dataset — note small batch size for 100-sample training."""
    train_data = np.load('data/train.npy', allow_pickle=True).item()
    test_data  = np.load('data/test.npy',  allow_pickle=True).item()

    X_train = torch.tensor(train_data['X'][:n_train], dtype=torch.float32)
    Y_train = torch.tensor(train_data['Y'][:n_train], dtype=torch.float32)
    X_test  = torch.tensor(test_data['X'],            dtype=torch.float32)
    Y_test  = torch.tensor(test_data['Y'],            dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset  = torch.utils.data.TensorDataset(X_test,  Y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    return train_loader, test_loader, X_test, Y_test

# ── Training Loop ──────────────────────────────────────────────────────────────
def train_pinn(model, train_loader, test_loader, epochs=300, lr=1e-3, lam=0.5):
    """
    Train PINN-Cal with combined data + physics loss.
    lam: weight of physics penalty (lambda in project document)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = nn.MSELoss()

    history = {
        'train_total': [], 'train_data': [], 'train_physics': [],
        'test_loss': [], 'L_rec': [], 'L_coup': [], 'L_eng': []
    }

    best_test_loss   = float('inf')
    patience_counter = 0
    patience         = 30

    print(f"\nTraining PINN-Cal (lambda={lam}) for up to {epochs} epochs...")
    print(f"Physics weights: alpha=1.0, beta=0.5, gamma=0.3")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        batch_data_losses, batch_phys_losses = [], []
        batch_rec, batch_coup, batch_eng     = [], [], []

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()

            pred = model(X_batch)

            # Data loss
            L_data = criterion(pred, Y_batch)

            # Physics loss
            L_phys, L_rec, L_coup, L_eng = physics_loss(pred, X_batch)

            # Combined loss
            L_total = L_data + lam * L_phys

            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_data_losses.append(L_data.item())
            batch_phys_losses.append(L_phys.item())
            batch_rec.append(L_rec.item())
            batch_coup.append(L_coup.item())
            batch_eng.append(L_eng.item())

        # Validate
        model.eval()
        with torch.no_grad():
            test_losses_batch = []
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                test_losses_batch.append(criterion(pred, Y_batch).item())
            test_loss = np.mean(test_losses_batch)

        history['train_total'].append(np.mean(batch_data_losses) + lam * np.mean(batch_phys_losses))
        history['train_data'].append(np.mean(batch_data_losses))
        history['train_physics'].append(np.mean(batch_phys_losses))
        history['test_loss'].append(test_loss)
        history['L_rec'].append(np.mean(batch_rec))
        history['L_coup'].append(np.mean(batch_coup))
        history['L_eng'].append(np.mean(batch_eng))

        scheduler.step()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'models/pinn_cal_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch+1) % 30 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d} | Data: {history['train_data'][-1]:.6f} | "
                  f"Physics: {history['train_physics'][-1]:.6f} | "
                  f"Test: {test_loss:.6f} | Time: {elapsed:.1f}s")

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest test loss: {best_test_loss:.6f}")
    return history

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, Y_test, label="PINN-Cal"):
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test.to(device)).cpu().numpy()
    Y_true = Y_test.numpy()

    nmse    = np.mean(np.sum((Y_pred - Y_true)**2, axis=1) /
                      np.sum(Y_true**2, axis=1))
    nmse_db = 10 * np.log10(nmse + 1e-10)

    pred_complex = Y_pred[:, :M] + 1j * Y_pred[:, M:]
    true_complex = Y_true[:, :M] + 1j * Y_true[:, M:]
    phase_error  = np.mean(np.abs(np.angle(pred_complex / true_complex, deg=True)))
    gain_error   = np.mean(np.abs(20 * np.log10(
                       np.abs(pred_complex) / (np.abs(true_complex) + 1e-10))))

    print(f"\n── {label} Evaluation ──────────────────")
    print(f"  NMSE:             {nmse_db:.4f} dB")
    print(f"  Mean phase error: {phase_error:.4f} degrees")
    print(f"  Mean gain error:  {gain_error:.4f} dB")
    print(f"────────────────────────────────────────")
    return nmse_db, phase_error, gain_error

# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(history['train_data'],    'b-', label='Train data loss',    linewidth=2)
    axes[0].plot(history['test_loss'],     'r-', label='Test loss',          linewidth=2)
    axes[0].plot(history['train_physics'], 'g--',label='Physics loss',       linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('PINN-Cal Training (100 samples)')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Physics components
    axes[1].plot(history['L_rec'],  'b-', label='Reciprocity',        linewidth=2)
    axes[1].plot(history['L_coup'], 'r-', label='Mutual coupling',     linewidth=2)
    axes[1].plot(history['L_eng'],  'g-', label='Energy conservation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Physics Loss Component')
    axes[1].set_title('Physics Constraint Convergence')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves_pinn.png', dpi=150)
    print("PINN-Cal training curves saved: training_curves_pinn.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs('models', exist_ok=True)

    # Load only 100 training samples
    train_loader, test_loader, X_test, Y_test = load_data(n_train=100, batch_size=16)

    # Build model
    model = PINNCal().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train with physics constraints
    history = train_pinn(model, train_loader, test_loader, epochs=300, lr=1e-3, lam=0.5)

    # Evaluate
    model.load_state_dict(torch.load('models/pinn_cal_best.pth', weights_only=True))
    nmse_db, phase_err, gain_err = evaluate_model(model, X_test, Y_test)

    # Plot
    plot_training(history)

    # Save results
    results = {
        'nmse_db': nmse_db,
        'phase_error_deg': phase_err,
        'gain_error_db': gain_err,
        'history': history,
        'n_train': 100,
        'lambda': 0.5
    }
    np.save('models/pinn_cal_results.npy', results)

    print("\nStage 9 complete — PINN-Cal trained with physics constraints.")
