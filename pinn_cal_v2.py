"""
PINN-Cal v2: Fixed training with warm start + adaptive loss weighting.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time, os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

M          = 64
FREQ       = 28e9
C          = 3e8
WAVELENGTH = C / FREQ
D          = WAVELENGTH / 2
K0         = 2 * np.pi / WAVELENGTH
ETA0       = 377.0

def compute_mutual_impedance_matrix(m=M, d=D):
    Z = np.zeros((m, m), dtype=complex)
    for i in range(m):
        for j in range(m):
            if i == j:
                Z[i, j] = 73.1 + 42.5j
            else:
                d_ij = abs(i - j) * d
                k_d  = K0 * d_ij
                Z[i, j] = (ETA0 / (4 * np.pi)) * (
                    np.exp(-1j * k_d) / d_ij
                ) * (2 * np.cos(k_d) + 2j * np.sin(k_d) / k_d)
    return torch.tensor(Z, dtype=torch.complex64).to(device)

Z_theory = compute_mutual_impedance_matrix()

class PINNCal(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[256, 128, 64], output_dim=128):
        super().__init__()
        layers = []
        prev   = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

def reciprocity_loss(pred):
    w = torch.complex(pred[:, :M], pred[:, M:])
    H = torch.bmm(w.unsqueeze(2), w.conj().unsqueeze(1))
    diff = H - H.transpose(1,2).conj()
    return torch.mean(torch.abs(diff)**2) / (M * M)

def coupling_loss(pred):
    w = torch.complex(pred[:, :M], pred[:, M:])
    Z_pred = torch.bmm(w.unsqueeze(2), w.conj().unsqueeze(1))
    Z_th   = Z_theory.unsqueeze(0).expand(w.shape[0], -1, -1)
    Zp = Z_pred / (torch.abs(Z_pred).max() + 1e-8)
    Zt = Z_th   / (torch.abs(Z_th).max()   + 1e-8)
    return torch.mean(torch.abs(Zp - Zt)**2)

def energy_loss(pred):
    power = torch.sum(pred[:, :M]**2 + pred[:, M:]**2, dim=1)
    return torch.mean((power / M - 1.0)**2)

def get_lambda(epoch, warmup=50, max_lam=0.1):
    if epoch < warmup:
        return 0.0
    elif epoch < warmup + 50:
        return max_lam * (epoch - warmup) / 50
    else:
        return max_lam

def load_data(n_train=100, batch_size=16):
    train_data = np.load('data/train.npy', allow_pickle=True).item()
    test_data  = np.load('data/test.npy',  allow_pickle=True).item()
    X_tr = torch.tensor(train_data['X'][:n_train], dtype=torch.float32)
    Y_tr = torch.tensor(train_data['Y'][:n_train], dtype=torch.float32)
    X_te = torch.tensor(test_data['X'],             dtype=torch.float32)
    Y_te = torch.tensor(test_data['Y'],             dtype=torch.float32)
    tr_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    te_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_te, Y_te), batch_size=batch_size, shuffle=False)
    print(f"Train: {len(X_tr)} samples | Test: {len(X_te)} samples")
    return tr_loader, te_loader, X_te, Y_te

def train_pinn(model, tr_loader, te_loader, epochs=300, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    history   = {'train_data':[], 'train_physics':[], 'test_loss':[], 'lambda':[]}
    best_loss, patience_count, patience = float('inf'), 0, 40

    print(f"\nPhase 1 (epochs 1-50):   data loss only (warm start)")
    print(f"Phase 2 (epochs 51-100): gradually add physics")
    print(f"Phase 3 (epochs 101+):   full PINN training\n")
    t0 = time.time()

    for epoch in range(epochs):
        lam = get_lambda(epoch, warmup=50, max_lam=0.1)
        model.train()
        d_losses, p_losses = [], []

        for Xb, Yb in tr_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            pred   = model(Xb)
            L_data = criterion(pred, Yb)
            if lam > 0:
                L_phys = reciprocity_loss(pred) + \
                         0.5 * coupling_loss(pred) + \
                         0.3 * energy_loss(pred)
            else:
                L_phys = torch.tensor(0.0)
            loss = L_data + lam * L_phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            d_losses.append(L_data.item())
            p_losses.append(L_phys.item() if lam > 0 else 0.0)

        model.eval()
        with torch.no_grad():
            te_loss = np.mean([criterion(model(Xb.to(device)), Yb.to(device)).item()
                               for Xb, Yb in te_loader])

        history['train_data'].append(np.mean(d_losses))
        history['train_physics'].append(np.mean(p_losses))
        history['test_loss'].append(te_loss)
        history['lambda'].append(lam)
        scheduler.step()

        if te_loss < best_loss:
            best_loss = te_loss
            torch.save(model.state_dict(), 'models/pinn_cal_v2_best.pth')
            patience_count = 0
        else:
            patience_count += 1

        if (epoch+1) % 30 == 0:
            phase = "warm-up" if lam == 0 else f"PINN lam={lam:.3f}"
            print(f"  Epoch {epoch+1:3d} [{phase}] | "
                  f"Data: {history['train_data'][-1]:.6f} | "
                  f"Physics: {history['train_physics'][-1]:.6f} | "
                  f"Test: {te_loss:.6f} | {time.time()-t0:.1f}s")

        if patience_count >= patience and epoch > 100:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest test loss: {best_loss:.6f}")
    return history

def evaluate(model, X_te, Y_te):
    model.eval()
    with torch.no_grad():
        Yp = model(X_te.to(device)).cpu().numpy()
    Yt = Y_te.numpy()
    nmse    = np.mean(np.sum((Yp-Yt)**2,axis=1)/np.sum(Yt**2,axis=1))
    nmse_db = 10*np.log10(nmse+1e-10)
    pc      = Yp[:,:M]+1j*Yp[:,M:]
    tc      = Yt[:,:M]+1j*Yt[:,M:]
    ph_err  = np.mean(np.abs(np.angle(pc/tc, deg=True)))
    gn_err  = np.mean(np.abs(20*np.log10(np.abs(pc)/(np.abs(tc)+1e-10))))
    print(f"\n── PINN-Cal v2 Evaluation ──────────────")
    print(f"  NMSE:             {nmse_db:.4f} dB")
    print(f"  Mean phase error: {ph_err:.4f} degrees")
    print(f"  Mean gain error:  {gn_err:.4f} dB")
    print(f"────────────────────────────────────────")
    return nmse_db, ph_err, gn_err

def plot_results(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_data'],    'b-',  label='Train data loss', linewidth=2)
    axes[0].plot(history['test_loss'],     'r-',  label='Test loss',       linewidth=2)
    axes[0].plot(history['train_physics'], 'g--', label='Physics loss',    linewidth=1.5)
    axes[0].axvline(x=50,  color='gray', linestyle=':',  label='Physics ON')
    axes[0].axvline(x=100, color='gray', linestyle='--', label='Full PINN')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('PINN-Cal v2 — Warm Start (100 samples)')
    axes[0].legend(); axes[0].set_yscale('log'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['lambda'], 'purple', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Lambda')
    axes[1].set_title('Adaptive Lambda Schedule')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves_pinn_v2.png', dpi=150)
    print("Saved: training_curves_pinn_v2.png")

if __name__ == '__main__':
    torch.manual_seed(42); np.random.seed(42)
    os.makedirs('models', exist_ok=True)
    tr_loader, te_loader, X_te, Y_te = load_data(n_train=100, batch_size=16)
    model = PINNCal().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    history = train_pinn(model, tr_loader, te_loader, epochs=300, lr=1e-3)
    model.load_state_dict(torch.load('models/pinn_cal_v2_best.pth', weights_only=True))
    nmse_db, ph_err, gn_err = evaluate(model, X_te, Y_te)
    plot_results(history)
    np.save('models/pinn_cal_v2_results.npy', {
        'nmse_db': nmse_db, 'phase_error_deg': ph_err,
        'gain_error_db': gn_err, 'history': history, 'n_train': 100
    })
    print("\nStage 9 (v2) complete.")
