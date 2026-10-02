import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experimental.v4.pinn_v4_model import PINNCalV4, CONFIG, M
from physics_core import DEFAULT_CONFIG

def inspect_saved_results():
    results_path = Path("experimental/v4/results/v4_complete_results.npy")
    if results_path.exists():
        res = np.load(results_path, allow_pickle=True).item()
        print("=== Complete Results Summary ===")
        for k, v in res.items():
            print(f"Model: {k}")
            print(f"  Best epoch         : {v.get('best_epoch')}")
            print(f"  Train loss at best : {v.get('train_loss_at_best')}")
            print(f"  Best val loss      : {v.get('best_val_loss')}")
            print(f"  Gen gap            : {v.get('gen_gap')}")
            ft = v.get("frozen_test", {})
            print(f"  Frozen test NMSE   : {ft.get('nmse_db')} dB")
            print(f"  Frozen phase err   : {ft.get('phase_error_deg')} deg")
            print(f"  Frozen point err   : {ft.get('pointing_error_deg')} deg")
            print(f"  Frozen beam corr   : {ft.get('beam_correlation')}")
            print()

    hist_dir = Path("experimental/v4/histories")
    for hf in sorted(hist_dir.glob("*.npy")):
        h = np.load(hf, allow_pickle=True).item()
        print(f"=== History: {hf.name} ===")
        print(f"  Total epochs: {len(h['epoch'])}")
        print(f"  Epoch 1  : train={h['train_loss'][0]:.6f}, val={h['val_loss'][0]:.6f}, gap={h['gen_gap'][0]:.6f}")
        last = len(h['epoch']) - 1
        print(f"  Epoch {h['epoch'][last]:2d} : train={h['train_loss'][last]:.6f}, val={h['val_loss'][last]:.6f}, gap={h['gen_gap'][last]:.6f}")
        # Print top 5 lowest val loss epochs
        val_losses = h['val_loss']
        best_idx = int(np.argmin(val_losses))
        print(f"  Best Ep {h['epoch'][best_idx]} : train={h['train_loss'][best_idx]:.6f}, val={h['val_loss'][best_idx]:.6f}, gap={h['gen_gap'][best_idx]:.6f}")
        print()

def inspect_datasets():
    train_data = np.load("experimental/v4/datasets/v4_train_2000.npy", allow_pickle=True).item()
    val_data = np.load("experimental/v4/datasets/v4_val_500.npy", allow_pickle=True).item()

    print("=== Dataset Inspection ===")
    print(f"Train X: shape={train_data['X'].shape}, mean={train_data['X'].mean():.4f}, std={train_data['X'].std():.4f}")
    print(f"Train Y: shape={train_data['Y'].shape}, mean={train_data['Y'].mean():.4f}, std={train_data['Y'].std():.4f}")
    print(f"Train SNR: min={train_data['snr_db'].min():.2f}, max={train_data['snr_db'].max():.2f}, mean={train_data['snr_db'].mean():.2f}")

    print(f"Val X: shape={val_data['X'].shape}, mean={val_data['X'].mean():.4f}, std={val_data['X'].std():.4f}")
    print(f"Val Y: shape={val_data['Y'].shape}, mean={val_data['Y'].mean():.4f}, std={val_data['Y'].std():.4f}")
    print(f"Val SNR: min={val_data['snr_db'].min():.2f}, max={val_data['snr_db'].max():.2f}, mean={val_data['snr_db'].mean():.2f}")

    # Check baseline error on train vs val (Physics Prior alone, Delta_C = 0)
    device = torch.device("cpu")
    model = PINNCalV4(bounded=True, alpha=0.5).to(device)
    # With weights zeroed, Delta_C is exactly 0
    nn.init.zeros_(model.residual_denoiser.net[-1].weight)
    nn.init.zeros_(model.residual_denoiser.net[-1].bias)

    criterion = nn.MSELoss()

    # Prior error on Train
    with torch.no_grad():
        c_hat_tr, delta_c_tr, x_prior_tr = model(torch.tensor(train_data["X"], dtype=torch.float32))
        c_hat_tr_real = torch.cat([c_hat_tr.real.to(torch.float32), c_hat_tr.imag.to(torch.float32)], dim=-1)
        prior_loss_tr = criterion(c_hat_tr_real, torch.tensor(train_data["Y"], dtype=torch.float32)).item()

        # Prior error on Val
        c_hat_va, delta_c_va, x_prior_va = model(torch.tensor(val_data["X"], dtype=torch.float32))
        c_hat_va_real = torch.cat([c_hat_va.real.to(torch.float32), c_hat_va.imag.to(torch.float32)], dim=-1)
        prior_loss_va = criterion(c_hat_va_real, torch.tensor(val_data["Y"], dtype=torch.float32)).item()

    print(f"Zero-residual Physics Prior MSE on Train: {prior_loss_tr:.6f}")
    print(f"Zero-residual Physics Prior MSE on Val  : {prior_loss_va:.6f}")

    # Outlier / heavy-tail analysis
    err_tr = np.abs((c_hat_tr.numpy()) - (train_data["Y"][:, :64] + 1j * train_data["Y"][:, 64:]))**2
    err_va = np.abs((c_hat_va.numpy()) - (val_data["Y"][:, :64] + 1j * val_data["Y"][:, 64:]))**2
    sample_err_tr = np.mean(err_tr, axis=1) / 2.0  # divided by 2 to match real/imag MSE definition
    sample_err_va = np.mean(err_va, axis=1) / 2.0

    print("\n--- Error Distribution & Outliers ---")
    print(f"Train: min={np.min(sample_err_tr):.6f}, median={np.median(sample_err_tr):.6f}, mean={np.mean(sample_err_tr):.6f}, 95th={np.percentile(sample_err_tr, 95):.6f}, 99th={np.percentile(sample_err_tr, 99):.6f}, max={np.max(sample_err_tr):.6f}")
    print(f"Val  : min={np.min(sample_err_va):.6f}, median={np.median(sample_err_va):.6f}, mean={np.mean(sample_err_va):.6f}, 95th={np.percentile(sample_err_va, 95):.6f}, 99th={np.percentile(sample_err_va, 99):.6f}, max={np.max(sample_err_va):.6f}")

    worst_tr = np.argsort(sample_err_tr)[-5:]
    print("Worst 5 train sample errors:", np.round(sample_err_tr[worst_tr], 4), "at SNRs:", np.round(train_data["snr_db"][worst_tr], 2))
    worst_va = np.argsort(sample_err_va)[-5:]
    print("Worst 5 val sample errors  :", np.round(sample_err_va[worst_va], 4), "at SNRs:", np.round(val_data["snr_db"][worst_va], 2))

    # Let's inspect loss by SNR bin!
    print("\n--- Train & Val Prior Loss by SNR Bins ---")
    for snr_low, snr_high in [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]:
        tr_mask = (train_data['snr_db'] >= snr_low) & (train_data['snr_db'] < snr_high)
        va_mask = (val_data['snr_db'] >= snr_low) & (val_data['snr_db'] < snr_high)
        l_tr = criterion(c_hat_tr_real[tr_mask], torch.tensor(train_data["Y"][tr_mask], dtype=torch.float32)).item()
        l_va = criterion(c_hat_va_real[va_mask], torch.tensor(val_data["Y"][va_mask], dtype=torch.float32)).item()
        print(f"  SNR [{snr_low:2d}, {snr_high:2d}) dB: Train N={tr_mask.sum():3d}, MSE={l_tr:.6f} | Val N={va_mask.sum():3d}, MSE={l_va:.6f}")

def trace_batches():
    train_data = np.load("experimental/v4/datasets/v4_train_2000.npy", allow_pickle=True).item()
    val_data = np.load("experimental/v4/datasets/v4_val_500.npy", allow_pickle=True).item()

    model = PINNCalV4(bounded=True, alpha=0.5)
    model.eval()
    criterion = nn.MSELoss()

    xb = torch.tensor(train_data["X"][:32], dtype=torch.float32)
    yb = torch.tensor(train_data["Y"][:32], dtype=torch.float32)

    xvb = torch.tensor(val_data["X"][:32], dtype=torch.float32)
    yvb = torch.tensor(val_data["Y"][:32], dtype=torch.float32)

    print("\n=== Single Batch Trace ===")
    print(f"Train batch: xb shape={xb.shape}, yb shape={yb.shape}")
    c_hat, delta_c, x_prior = model(xb)
    print(f"  x_prior: shape={x_prior.shape}, min={x_prior.min():.4f}, max={x_prior.max():.4f}, mean={x_prior.mean():.4f}")
    print(f"  delta_c: shape={delta_c.shape}, min={delta_c.min():.4f}, max={delta_c.max():.4f}, mean={delta_c.mean():.4f}")
    print(f"  c_hat  : shape={c_hat.shape}, dtype={c_hat.dtype}")
    c_hat_real = torch.cat([c_hat.real.to(torch.float32), c_hat.imag.to(torch.float32)], dim=-1)
    train_batch_loss = criterion(c_hat_real, yb).item()
    print(f"  c_hat_real: shape={c_hat_real.shape}, loss against yb: {train_batch_loss:.6f}")

    print(f"\nVal batch: xvb shape={xvb.shape}, yvb shape={yvb.shape}")
    c_hat_v, delta_c_v, x_prior_v = model(xvb)
    c_hat_v_real = torch.cat([c_hat_v.real.to(torch.float32), c_hat_v.imag.to(torch.float32)], dim=-1)
    val_batch_loss = criterion(c_hat_v_real, yvb).item()
    print(f"  c_hat_v_real: shape={c_hat_v_real.shape}, loss against yvb: {val_batch_loss:.6f}")

if __name__ == "__main__":
    inspect_saved_results()
    inspect_datasets()
    trace_batches()
