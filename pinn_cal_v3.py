"""
PINN-Cal v3 — Physically Grounded PINN

Input:
    64 complex observed coupled currents
    + sin(theta), cos(theta)

Output:
    64 complex hardware-correction coefficients

Physics loss:
    The predicted correction is applied to the observed hardware
    voltage through the reciprocal impedance network.

    I_obs
      -> V_obs = Z I_obs
      -> V_corr = C_pred * V_obs
      -> I_corr = Z^-1 V_corr

    I_corr is compared directly against the ideal coupled current.

This is intentionally different from the legacy PINN implementation:
there is no w w^H reciprocity construction and no artificial
impedance-matrix matching loss.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

from physics_core import (
    DEFAULT_CONFIG,
    mutual_impedance_matrix,
    steering_vector,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

M = DEFAULT_CONFIG.num_elements
INPUT_DIM = 2 * M + 2
OUTPUT_DIM = 2 * M


class PINNCalV3(nn.Module):
    """MLP that predicts complex per-element hardware corrections."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        output_dim: int = OUTPUT_DIM,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous = input_dim

        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            previous = hidden

        layers.append(nn.Linear(previous, output_dim))

        self.network = nn.Sequential(*layers)

        # Physical constants are buffers so they move with the model.
        Z = mutual_impedance_matrix(DEFAULT_CONFIG)

        self.register_buffer(
            "Z",
            Z.to(torch.complex128),
        )

        self.register_buffer(
            "z_self",
            Z.diagonal().mean().to(torch.complex128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @staticmethod
    def decode_complex(
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Decode [real(64), imag(64)] into a complex tensor."""
        return torch.complex(
            values[:, :M].to(torch.float64),
            values[:, M:].to(torch.float64),
        )

    def observed_current(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Decode the observed coupled current from the input."""
        return torch.complex(
            x[:, :M].to(torch.float64),
            x[:, M:2 * M].to(torch.float64),
        )

    def target_angles(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Recover steering angle from sin(theta), cos(theta)."""
        sin_theta = x[:, 2 * M]
        cos_theta = x[:, 2 * M + 1]

        return torch.rad2deg(
            torch.atan2(sin_theta, cos_theta)
        ).to(torch.float64)

    def ideal_coupled_current(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the ideal coupled current for each requested angle.
        """
        angles = self.target_angles(x)

        v_ideal = steering_vector(
            angles,
            DEFAULT_CONFIG,
        ).to(torch.complex128)

        rhs = v_ideal * self.z_self

        i_ideal = torch.linalg.solve(
            self.Z,
            rhs.T,
        ).T

        return i_ideal

    def corrected_current(
        self,
        x: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the predicted hardware correction through the
        physical impedance network.
        """
        i_observed = self.observed_current(x)
        correction = self.decode_complex(prediction)

        # For row-wise batch representation:
        # V_obs = Z @ I_obs
        # becomes I_obs @ Z.T.
        v_observed = i_observed @ self.Z.T

        v_corrected = correction * v_observed

        i_corrected = torch.linalg.solve(
            self.Z,
            v_corrected.T,
        ).T

        return i_corrected

    def physics_loss(
        self,
        x: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Direct coupled-current physics residual.

        This compares the physical state produced by the predicted
        correction against the ideal coupled array state.
        """
        i_pred = self.corrected_current(
            x,
            prediction,
        )

        i_target = self.ideal_coupled_current(x)

        numerator = torch.sum(
            torch.abs(i_pred - i_target) ** 2,
            dim=1,
        )

        denominator = torch.sum(
            torch.abs(i_target) ** 2,
            dim=1,
        ).clamp_min(1e-12)

        return torch.mean(numerator / denominator)


def load_data(
    train_samples: int = 1000,
    batch_size: int = 32,
):
    """Load the new physically grounded dataset."""
    train_data = np.load(
        "data/physics_train.npy",
        allow_pickle=True,
    ).item()

    test_data = np.load(
        "data/physics_test.npy",
        allow_pickle=True,
    ).item()

    X_train = torch.tensor(
        train_data["X"][:train_samples],
        dtype=torch.float32,
    )

    Y_train = torch.tensor(
        train_data["Y"][:train_samples],
        dtype=torch.float32,
    )

    X_test = torch.tensor(
        test_data["X"],
        dtype=torch.float32,
    )

    Y_test = torch.tensor(
        test_data["Y"],
        dtype=torch.float32,
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            X_train,
            Y_train,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            X_test,
            Y_test,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        test_loader,
        X_test,
        Y_test,
    )


def train_one_epoch(
    model: PINNCalV3,
    loader,
    optimizer,
    lambda_physics: float,
):
    model.train()

    data_losses = []
    physics_losses = []
    total_losses = []

    criterion = nn.MSELoss()

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(X_batch)

        data_loss = criterion(
            prediction,
            Y_batch,
        )

        physics_loss = model.physics_loss(
            X_batch,
            prediction,
        )

        total_loss = (
            data_loss
            + lambda_physics * physics_loss
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        data_losses.append(float(data_loss.detach()))
        physics_losses.append(float(physics_loss.detach()))
        total_losses.append(float(total_loss.detach()))

    return (
        float(np.mean(data_losses)),
        float(np.mean(physics_losses)),
        float(np.mean(total_losses)),
    )


@torch.no_grad()
def evaluate_data_loss(
    model: PINNCalV3,
    loader,
) -> float:
    """Evaluate coefficient-domain MSE."""
    model.eval()

    criterion = nn.MSELoss()
    losses = []

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        prediction = model(X_batch)

        losses.append(
            float(
                criterion(
                    prediction,
                    Y_batch,
                )
            )
        )

    return float(np.mean(losses))


@torch.no_grad()
def evaluate_physics_loss(
    model: PINNCalV3,
    loader,
) -> float:
    """Evaluate the coupled-current physics residual."""
    model.eval()

    losses = []

    for X_batch, _ in loader:
        X_batch = X_batch.to(DEVICE)

        prediction = model(X_batch)

        losses.append(
            float(
                model.physics_loss(
                    X_batch,
                    prediction,
                )
            )
        )

    return float(np.mean(losses))


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs("models", exist_ok=True)

    print(f"Using device: {DEVICE}")

    train_loader, test_loader, _, _ = load_data(
        train_samples=1000,
        batch_size=32,
    )

    model = PINNCalV3().to(DEVICE)

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters: {parameters:,}")
    print(f"Input dimension: {INPUT_DIM}")
    print(f"Output dimension: {OUTPUT_DIM}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    epochs = 200

    start_time = time.time()

    print("\nPhysics-loss verification training")
    print("Phase 1: data loss only")
    print("Phase 2: gradually increase physics loss")

    for epoch in range(epochs):

        if epoch < 50:
            lambda_physics = 0.0
        elif epoch < 100:
            lambda_physics = (
                0.1 * (epoch - 50) / 50.0
            )
        else:
            lambda_physics = 0.1

        data_loss, physics_loss, total_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lambda_physics,
        )

        if (epoch + 1) % 20 == 0:
            test_data_loss = evaluate_data_loss(
                model,
                test_loader,
            )

            test_physics_loss = evaluate_physics_loss(
                model,
                test_loader,
            )

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch + 1:3d} | "
                f"λ={lambda_physics:.3f} | "
                f"Data={data_loss:.6e} | "
                f"Physics={physics_loss:.6e} | "
                f"TestData={test_data_loss:.6e} | "
                f"TestPhysics={test_physics_loss:.6e} | "
                f"{elapsed:.1f}s"
            )

    torch.save(
        model.state_dict(),
        "models/pinn_cal_v3_last.pth",
    )

    print("\nSaved:")
    print("  models/pinn_cal_v3_last.pth")


if __name__ == "__main__":
    main()