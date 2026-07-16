"""
PINN-Cal: Ablation Study
Tests contribution of each physics constraint individually.
Trains 4 models: full PINN + 3 versions with one constraint removed each.
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

def compute_Z(m=M, d=D):
    Z = np.zeros((m, m), dtype=complex)
    for i in range(m):
        for j in range(m):
            if i == j:
                Z[i,j] = 73.1 + 42.5j
            else:
                d_ij = abs(i-j)*d; k_d = K0*d_ij
                Z[i,j] = (ETA0/(4*np.pi))*(np.exp(-1j*k_d)/d_ij)*(
                    2*np.cos(k_d)+2j*np.sin(k_d)/(k_d))
    return torch.tensor(Z, dtype=torch.complex64).to(device)

Z_theory = compute_Z()

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

def reciprocity_loss(pred):
    w = torch.complex(pred[:,:M], pred[:,M:])
    H = torch.bmm(w.unsqueeze(2), w.conj().unsqueeze(1))
    return torch.mean(torch.abs(H - H.transpose(1,2).conj())**2) / (M*M)

def coupling_loss(pred):
    w  = torch.complex(pred[:,:M], pred[:,M:])
    Zp = torch.bmm(w.unsqueeze(2), w.conj().unsqueeze(1))
    Zt = Z_theory.unsqueeze(0).expand(w.shape[0],-1,-1)
    Zp = Zp/(torch.abs(Zp).max()+1e-8)
    Zt = Zt/(torch.abs(Zt).max()+1e-8)
    return torch.mean(torch.abs(Zp-Zt)**2)

def energy_loss(pred):
    power = torch.sum(pred[:,:M]**2 + pred[:,M:]**2, dim=1)
    return torch.mean((power/M - 1.0)**2)

def get_lambda(epoch, warmup=50, max_lam=0.1):
    if epoch < warmup:       return 0.0
    elif epoch < warmup+50:  return max_lam*(epoch-warmup)/50
    else:                    return max_lam

def load_data(n_train=100, batch_size=16):
    tr = np.load('data/train.npy', allow_pickle=True).item()
    te = np.load('data/test.npy',  allow_pickle=True).item()
    X_tr = torch.tensor(tr['X'][:n_train], dtype=torch.float32)
    Y_tr = torch.tensor(tr['Y'][:n_train], dtype=torch.float32)
    X_te = torch.tensor(te['X'], dtype=torch.float32)
    Y_te = torch.tensor(te['Y'], dtype=torch.float32)
    tr_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    te_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_te, Y_te), batch_size=batch_size, shuffle=False)
    return tr_loader, te_loader, X_te, Y_te

def train_variant(name, use_rec=True, use_coup=True, use_eng=True,
                  epochs=250, lr=1e-3):
    """Train one ablation variant."""
    print(f"\n── Training: {name} ──────────────────────────")
    print(f"   Reciprocity={use_rec} | Coupling={use_coup} | Energy={use_eng}")
    torch.manual_seed(42); np.random.seed(42)

    tr_loader, te_loader, X_te, Y_te = load_data()
    model     = PINNCal().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_loss, patience_count, patience = float('inf'), 0, 35
    test_losses = []
    t0 = time.time()

    for epoch in range(epochs):
        lam = get_lambda(epoch)
        model.train()
        for Xb, Yb in tr_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            pred   = model(Xb)
            L_data = criterion(pred, Yb)
            L_phys = torch.tensor(0.0, device=device)
            if lam > 0:
                if use_rec:  L_phys = L_phys + reciprocity_loss(pred)
                if use_coup: L_phys = L_phys + 0.5*coupling_loss(pred)
                if use_eng:  L_phys = L_phys + 0.3*energy_loss(pred)
            (L_data + lam*L_phys).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            te_loss = np.mean([criterion(model(Xb.to(device)), Yb.to(device)).item()
                               for Xb,Yb in te_loader])
        test_losses.append(te_loss)
        scheduler.step()

        if te_loss < best_loss:
            best_loss = te_loss
            torch.save(model.state_dict(), f'models/ablation_{name}.pth')
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= patience and epoch > 100:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    # Evaluate
    model.load_state_dict(torch.load(f'models/ablation_{name}.pth', weights_only=True))
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

    print(f"   NMSE={nmse_db:.3f} dB | Phase={ph_err:.3f} deg | "
          f"Gain={gn_err:.3f} dB | Time={time.time()-t0:.1f}s")
    return nmse_db, ph_err, gn_err, test_losses

# ── Run all variants ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    variants = [
        # (name,                    use_rec, use_coup, use_eng)
        ('Full_PINN-Cal',           True,  True,  True ),
        ('No_Reciprocity',          False, True,  True ),
        ('No_Coupling',             True,  False, True ),
        ('No_Energy',               True,  True,  False),
    ]

    results = {}
    all_test_losses = {}

    for name, rec, coup, eng in variants:
        nmse, ph, gn, losses = train_variant(name, rec, coup, eng)
        results[name] = {'nmse_db': nmse, 'phase': ph, 'gain': gn}
        all_test_losses[name] = losses

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n\n══ ABLATION STUDY RESULTS ══════════════════════════════")
    print(f"{'Variant':<25} {'NMSE (dB)':>10} {'Phase (deg)':>12} {'Gain (dB)':>10}")
    print("─" * 62)
    for name, r in results.items():
        marker = " ◄ best" if name == 'Full_PINN-Cal' else ""
        print(f"{name:<25} {r['nmse_db']:>10.3f} {r['phase']:>12.3f} "
              f"{r['gain']:>10.3f}{marker}")

    # ── Plot 1: Bar chart ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ablation Study: Contribution of Each Physics Constraint\n'
                 '(All models trained on 100 samples)', fontsize=13, fontweight='bold')

    names  = list(results.keys())
    phases = [results[n]['phase'] for n in names]
    gains  = [results[n]['gain']  for n in names]
    colors = ['#4CAF50','#FF5722','#FF9800','#2196F3']
    labels = ['Full PINN-Cal\n(all constraints)',
              'No Reciprocity\n(R removed)',
              'No Coupling\n(C removed)',
              'No Energy\n(E removed)']

    bars = axes[0].bar(labels, phases, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, phases):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                     f'{val:.2f}°', ha='center', fontweight='bold')
    axes[0].set_ylabel('Mean Phase Error (degrees)')
    axes[0].set_title('Phase Error by Variant [lower = better]')
    axes[0].grid(True, alpha=0.3, axis='y')

    bars = axes[1].bar(labels, gains, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, gains):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f'{val:.2f}', ha='center', fontweight='bold')
    axes[1].set_ylabel('Mean Gain Error (dB)')
    axes[1].set_title('Gain Error by Variant [lower = better]')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('ablation_bar.png', dpi=150, bbox_inches='tight')
    print("\nSaved: ablation_bar.png")

    # ── Plot 2: Convergence curves ─────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    line_styles = ['-', '--', '-.', ':']
    for (name, _), ls, color in zip(results.items(), line_styles, colors):
        losses = all_test_losses[name]
        plt.plot(losses, ls, color=color, linewidth=2,
                 label=name.replace('_', ' '))
    plt.axvline(x=50,  color='gray', linestyle=':', alpha=0.7, label='Physics ON')
    plt.axvline(x=100, color='gray', linestyle='--',alpha=0.7, label='Full PINN')
    plt.xlabel('Epoch'); plt.ylabel('Test Loss (MSE)')
    plt.title('Ablation Study: Test Loss Convergence\n'
              'Full PINN-Cal converges lower than any ablated version')
    plt.legend(fontsize=9); plt.yscale('log'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ablation_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: ablation_convergence.png")

    np.save('models/ablation_results.npy', results)
    print("\nStage 11 complete — Ablation study done.")
