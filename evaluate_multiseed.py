from pathlib import Path

import numpy as np
import torch

from pinn_cal_final import (
    DEVICE,
    PINNCalV3,
    evaluate_data_loss,
    evaluate_physics_loss,
    load_data,
    train_one_epoch,
)


# =====================================================================
# Configuration
# =====================================================================

SEEDS = (
    42,
    123,
    456,
)

EPOCHS = 200
BATCH_SIZE = 32

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# =====================================================================
# Metrics
# =====================================================================

def nmse_db(predicted, target):
    numerator = np.sum(
        np.abs(predicted - target) ** 2
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


def phase_error_deg(predicted, target):
    phase_difference = np.angle(
        predicted * np.conj(target)
    )

    return float(
        np.degrees(
            np.mean(
                np.abs(phase_difference)
            )
        )
    )


def gain_error_db(predicted, target):
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
                predicted_db - target_db
            )
        )
    )


def relative_error(predicted, target):
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
# Evaluate one trained checkpoint
# =====================================================================

def evaluate_test_checkpoint(
    model,
    test_loader,
):
    """
    Evaluate only after the best validation checkpoint
    has already been selected.
    """

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for X_batch, Y_batch in test_loader:

            X_batch = X_batch.to(DEVICE)

            prediction = model(
                X_batch
            )

            predictions.append(
                prediction
                .detach()
                .cpu()
                .numpy()
            )

            targets.append(
                Y_batch
                .detach()
                .cpu()
                .numpy()
            )

    predicted = np.concatenate(
        predictions,
        axis=0,
    )

    target = np.concatenate(
        targets,
        axis=0,
    )

    predicted_complex = (
        predicted[:, :64]
        + 1j * predicted[:, 64:]
    )

    target_complex = (
        target[:, :64]
        + 1j * target[:, 64:]
    )

    return {
        "nmse":
            nmse_db(
                predicted_complex,
                target_complex,
            ),

        "phase":
            phase_error_deg(
                predicted_complex,
                target_complex,
            ),

        "gain":
            gain_error_db(
                predicted_complex,
                target_complex,
            ),

        "relative":
            relative_error(
                predicted_complex,
                target_complex,
            ),
    }


# =====================================================================
# Train one seed
# =====================================================================

def run_seed(seed):
    print()
    print("=" * 80)
    print(f"MULTI-SEED RUN — SEED {seed}")
    print("=" * 80)

    # -------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------
    # Controlled data split
    # -------------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        _,
        _,
    ) = load_data(
        batch_size=BATCH_SIZE
    )

    # -------------------------------------------------------------
    # Fresh model
    # -------------------------------------------------------------

    model = PINNCalV3().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    best_val_data_loss = float(
        "inf"
    )

    best_epoch = -1

    best_state = None

    # -------------------------------------------------------------
    # Fixed training schedule
    # -------------------------------------------------------------

    for epoch in range(EPOCHS):

        if epoch < 50:

            lambda_physics = 0.0

        elif epoch < 100:

            lambda_physics = (
                0.1
                * (epoch - 50)
                / 50.0
            )

        else:

            lambda_physics = 0.1

        train_one_epoch(
            model,
            train_loader,
            optimizer,
            lambda_physics,
        )

        # ---------------------------------------------------------
        # Validation only
        # ---------------------------------------------------------

        if (
            (epoch + 1) % 20 == 0
            or epoch == 0
        ):

            val_data_loss = (
                evaluate_data_loss(
                    model,
                    val_loader,
                )
            )

            val_physics_loss = (
                evaluate_physics_loss(
                    model,
                    val_loader,
                )
            )

            print(
                f"Epoch {epoch + 1:3d} | "
                f"λ={lambda_physics:.3f} | "
                f"ValData={val_data_loss:.6e} | "
                f"ValPhysics={val_physics_loss:.6e}"
            )

            if (
                val_data_loss
                < best_val_data_loss
            ):

                best_val_data_loss = (
                    val_data_loss
                )

                best_epoch = (
                    epoch + 1
                )

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in model.state_dict().items()
                }

    # -------------------------------------------------------------
    # Restore best validation checkpoint
    # -------------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            f"No validation checkpoint "
            f"was captured for seed {seed}."
        )

    model.load_state_dict(
        best_state
    )

    checkpoint_path = (
        MODEL_DIR
        / f"pinn_cal_multiseed_{seed}.pth"
    )

    torch.save(
        best_state,
        checkpoint_path,
    )

    # -------------------------------------------------------------
    # Final test evaluation
    # -------------------------------------------------------------

    test_metrics = (
        evaluate_test_checkpoint(
            model,
            test_loader,
        )
    )

    print()
    print(
        f"Seed {seed} best epoch : "
        f"{best_epoch}"
    )

    print(
        f"Seed {seed} ValData    : "
        f"{best_val_data_loss:.6e}"
    )

    print(
        f"Seed {seed} Test NMSE  : "
        f"{test_metrics['nmse']:.4f} dB"
    )

    print(
        f"Seed {seed} Test phase : "
        f"{test_metrics['phase']:.4f}°"
    )

    print(
        f"Seed {seed} Test gain  : "
        f"{test_metrics['gain']:.4f} dB"
    )

    print(
        f"Seed {seed} Test rel.  : "
        f"{100.0 * test_metrics['relative']:.4f}%"
    )

    print(
        f"Saved checkpoint      : "
        f"{checkpoint_path}"
    )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_data": best_val_data_loss,
        **test_metrics,
    }


# =====================================================================
# Main
# =====================================================================

def main():

    print("=" * 80)
    print(
        "PINN-Cal — Multi-Seed Validation"
    )
    print("=" * 80)

    print(
        f"Seeds        : {SEEDS}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        "Dataset      : fixed 800 / 200 / 200 split"
    )

    results = []

    for seed in SEEDS:

        results.append(
            run_seed(seed)
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    nmse_values = np.array(
        [r["nmse"] for r in results]
    )

    phase_values = np.array(
        [r["phase"] for r in results]
    )

    gain_values = np.array(
        [r["gain"] for r in results]
    )

    relative_values = np.array(
        [r["relative"] for r in results]
    )

    print()
    print("=" * 80)
    print(
        "MULTI-SEED SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        f"{'Seed':>8}"
        f"{'BestEpoch':>12}"
        f"{'ValData':>14}"
        f"{'NMSE(dB)':>14}"
        f"{'Phase(°)':>14}"
        f"{'Gain(dB)':>14}"
        f"{'Rel.Err(%)':>15}"
    )

    print(
        "-" * 95
    )

    for result in results:

        print(
            f"{result['seed']:8d}"
            f"{result['best_epoch']:12d}"
            f"{result['val_data']:14.6e}"
            f"{result['nmse']:14.4f}"
            f"{result['phase']:14.4f}"
            f"{result['gain']:14.4f}"
            f"{100.0 * result['relative']:15.4f}"
        )

    print()
    print(
        "MEAN ± STD"
    )

    print(
        f"NMSE       : "
        f"{np.mean(nmse_values):.4f} "
        f"± {np.std(nmse_values):.4f} dB"
    )

    print(
        f"Phase      : "
        f"{np.mean(phase_values):.4f} "
        f"± {np.std(phase_values):.4f}°"
    )

    print(
        f"Gain       : "
        f"{np.mean(gain_values):.4f} "
        f"± {np.std(gain_values):.4f} dB"
    )

    print(
        f"Rel. error : "
        f"{100.0 * np.mean(relative_values):.4f} "
        f"± {100.0 * np.std(relative_values):.4f}%"
    )

    print()
    print(
        "Multi-seed validation complete."
    )

    print(
        "Model selection used validation data only."
    )

    print(
        "The fixed test set was evaluated only after "
        "each seed's best validation checkpoint was selected."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
