from pathlib import Path

import numpy as np
import torch

from pinn_cal_final import PINNCalV3, DEVICE


DATA_PATH = Path("data/physics_test_200.npy")
MODEL_PATH = Path("models/pinn_cal_final_best.pth")


def nmse_db(predicted, target):
    numerator = np.sum(
        np.abs(predicted - target) ** 2
    )
    denominator = np.sum(
        np.abs(target) ** 2
    )
    return float(
        10.0 * np.log10(
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
    predicted_db = 20.0 * np.log10(
        np.maximum(
            np.abs(predicted),
            1e-12,
        )
    )

    target_db = 20.0 * np.log10(
        np.maximum(
            np.abs(target),
            1e-12,
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
            np.abs(predicted - target)
            /
            (
                np.abs(target)
                + 1e-12
            )
        )
    )


def main():

    data = np.load(
        DATA_PATH,
        allow_pickle=True,
    ).item()

    X = torch.tensor(
        data["X"],
        dtype=torch.float32,
        device=DEVICE,
    )

    Y = data["Y"]

    target = (
        Y[:, :64]
        + 1j * Y[:, 64:]
    )

    model = PINNCalV3().to(DEVICE)

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        prediction = model(X)

    predicted = (
        prediction[:, :64]
        .detach()
        .cpu()
        .numpy()
        +
        1j
        * prediction[:, 64:]
        .detach()
        .cpu()
        .numpy()
    )

    print("=" * 70)
    print("PINN-Cal — Final PINN Test Evaluation")
    print("=" * 70)

    print(f"Checkpoint         : {MODEL_PATH}")
    print(f"Test samples       : {len(target)}")
    print(f"Input shape        : {X.shape}")
    print(f"Target shape       : {target.shape}")

    print()

    print(
        f"NMSE               : "
        f"{nmse_db(predicted, target):.4f} dB"
    )

    print(
        f"Mean phase error   : "
        f"{phase_error_deg(predicted, target):.4f}°"
    )

    print(
        f"Mean gain error    : "
        f"{gain_error_db(predicted, target):.4f} dB"
    )

    print(
        f"Mean relative err. : "
        f"{100.0 * relative_error(predicted, target):.4f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
