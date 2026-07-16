"""
PINN-Cal: Baseline Neural Network (no physics constraints)
Standard MLP that learns calibration correction from data alone.
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
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Model Architecture ─────────────────────────────────────────────────────────
class BaselineMLP(nn.Module):
    """
    3-layer MLP for antenna calibration.
    Input:  128 values (real + imaginary parts of distorted weights)
    Output: 128 values (real + imaginary parts of correction)
    """
    def __init__(self, input_dim=128, hidden_dims=[256, 128, 64], output_dim=128):
        super(BaselineMLP, self).__init__()
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

# ── Data Loader ────────────────────────────────────────────────────────────────
def load_data(n_train=1000, batch_size=32):
    """Load dataset and create PyTorch DataLoaders."""
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
def train_model(model, train_loader, test_loader, epochs=200, lr=1e-3):
    """Train the baseline model and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    patience_counter = 0
    patience = 20

    print(f"\nTraining for up to {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        batch_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            test_batch_losses = []
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                test_batch_losses.append(loss.item())
            test_loss = np.mean(test_batch_losses)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()

        # ── Early stopping ──
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'models/baseline_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch+1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | "
                  f"Test Loss: {test_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {elapsed:.1f}s")

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest test loss: {best_test_loss:.6f}")
    return train_losses, test_losses

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, Y_test):
    """Compute NMSE and calibration error in dB."""
    model.eval()
    with torch.no_grad():
        X_test_gpu = X_test.to(device)
        Y_pred = model(X_test_gpu).cpu().numpy()
    Y_true = Y_test.numpy()

    # NMSE
    nmse = np.mean(np.sum((Y_pred - Y_true)**2, axis=1) /
                   np.sum(Y_true**2, axis=1))
    nmse_db = 10 * np.log10(nmse + 1e-10)

    # Per-element phase error (degrees)
    M = 64
    pred_complex = Y_pred[:, :M] + 1j * Y_pred[:, M:]
    true_complex = Y_true[:, :M] + 1j * Y_true[:, M:]
    phase_error  = np.angle(pred_complex / true_complex, deg=True)
    mean_phase_error = np.mean(np.abs(phase_error))

    # Gain error (dB)
    gain_error = 20 * np.log10(np.abs(pred_complex) / (np.abs(true_complex) + 1e-10))
    mean_gain_error = np.mean(np.abs(gain_error))

    print(f"\n── Evaluation Results ──────────────────")
    print(f"  NMSE:             {nmse_db:.4f} dB")
    print(f"  Mean phase error: {mean_phase_error:.4f} degrees")
    print(f"  Mean gain error:  {mean_gain_error:.4f} dB")
    print(f"────────────────────────────────────────")
    return nmse_db, mean_phase_error, mean_gain_error

# ── Plot Training Curves ───────────────────────────────────────────────────────
def plot_training(train_losses, test_losses, filename='training_curves_baseline.png'):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, 'b-', label='Train loss', linewidth=2)
    plt.plot(test_losses,  'r-', label='Test loss',  linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Baseline MLP — Training Curves')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Training curves saved: {filename}")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs('models', exist_ok=True)

    # Load data (full 1000 training samples)
    train_loader, test_loader, X_test, Y_test = load_data(n_train=1000, batch_size=32)

    # Build model
    model = BaselineMLP().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    train_losses, test_losses = train_model(model, train_loader, test_loader, epochs=200)

    # Load best weights and evaluate
    model.load_state_dict(torch.load('models/baseline_best.pth', weights_only=True))
    nmse_db, phase_err, gain_err = evaluate_model(model, X_test, Y_test)

    # Plot
    plot_training(train_losses, test_losses)

    # Save results
    results = {
        'nmse_db': nmse_db,
        'phase_error_deg': phase_err,
        'gain_error_db': gain_err,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'n_train': 1000
    }
    np.save('models/baseline_results.npy', results)

    print("\nStage 8 complete — Baseline model trained and evaluated.")
