"""
PINN-Cal — Final Physics-Informed Neural Calibration Model

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

Training protocol:
    800 samples  -> training
    200 samples  -> validation
    200 samples  -> test

The test set is kept untouched during training and model selection.
The best checkpoint is selected using validation data loss.
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


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------

M = DEFAULT_CONFIG.num_elements

INPUT_DIM = 2 * M + 2
OUTPUT_DIM = 2 * M


# ---------------------------------------------------------------------
# PINN Model
# ---------------------------------------------------------------------

class PINNCalV3(nn.Module):
    """
    MLP that predicts complex per-element hardware corrections.
    """

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
                    nn.Linear(
                        previous,
                        hidden,
                    ),

                    nn.BatchNorm1d(
                        hidden,
                    ),

                    nn.ReLU(),

                    nn.Dropout(
                        0.2,
                    ),
                ]
            )

            previous = hidden

        layers.append(
            nn.Linear(
                previous,
                output_dim,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

        # -------------------------------------------------------------
        # Physical constants
        # -------------------------------------------------------------

        Z = mutual_impedance_matrix(
            DEFAULT_CONFIG
        )

        self.register_buffer(
            "Z",
            Z.to(torch.complex128),
        )

        self.register_buffer(
            "z_self",
            Z.diagonal()
            .mean()
            .to(torch.complex128),
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)

    # -----------------------------------------------------------------
    # Complex decoding
    # -----------------------------------------------------------------

    @staticmethod
    def decode_complex(
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode:
            [real(64), imag(64)]

        into:
            complex(64)
        """

        return torch.complex(
            values[:, :M].to(torch.float64),
            values[:, M:].to(torch.float64),
        )

    # -----------------------------------------------------------------
    # Observed current
    # -----------------------------------------------------------------

    def observed_current(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode observed coupled current from the input.
        """

        return torch.complex(
            x[:, :M].to(torch.float64),
            x[:, M:2 * M].to(torch.float64),
        )

    # -----------------------------------------------------------------
    # Target steering angle
    # -----------------------------------------------------------------

    def target_angles(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover steering angle from:
            sin(theta)
            cos(theta)

        Returned angle is in degrees because
        steering_vector() expects degrees.
        """

        sin_theta = x[:, 2 * M]
        cos_theta = x[:, 2 * M + 1]

        return torch.rad2deg(
            torch.atan2(
                sin_theta,
                cos_theta,
            )
        ).to(torch.float64)

    # -----------------------------------------------------------------
    # Ideal coupled current
    # -----------------------------------------------------------------

    def ideal_coupled_current(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the ideal coupled current for each requested angle.
        """

        angles = self.target_angles(
            x
        )

        v_ideal = steering_vector(
            angles,
            DEFAULT_CONFIG,
        ).to(torch.complex128)

        rhs = (
            v_ideal
            * self.z_self
        )

        i_ideal = torch.linalg.solve(
            self.Z,
            rhs.T,
        ).T

        return i_ideal

    # -----------------------------------------------------------------
    # Corrected physical current
    # -----------------------------------------------------------------

    def corrected_current(
        self,
        x: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply predicted hardware correction through
        the physical impedance network.
        """

        i_observed = self.observed_current(
            x
        )

        correction = self.decode_complex(
            prediction
        )

        # -------------------------------------------------------------
        # Row-wise equivalent of:
        #
        #     V_obs = Z @ I_obs
        #
        # For a batch:
        #
        #     V_obs = I_obs @ Z.T
        # -------------------------------------------------------------

        v_observed = (
            i_observed
            @ self.Z.T
        )

        # -------------------------------------------------------------
        # Apply predicted correction.
        # -------------------------------------------------------------

        v_corrected = (
            correction
            * v_observed
        )

        # -------------------------------------------------------------
        # Convert corrected voltage back to current.
        # -------------------------------------------------------------

        i_corrected = torch.linalg.solve(
            self.Z,
            v_corrected.T,
        ).T

        return i_corrected

    # -----------------------------------------------------------------
    # Physics loss
    # -----------------------------------------------------------------

    def physics_loss(
        self,
        x: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Direct coupled-current physics residual.

        Compares the corrected physical state against
        the ideal coupled array state.
        """

        i_pred = self.corrected_current(
            x,
            prediction,
        )

        i_target = self.ideal_coupled_current(
            x
        )

        numerator = torch.sum(
            torch.abs(
                i_pred - i_target
            ) ** 2,
            dim=1,
        )

        denominator = torch.sum(
            torch.abs(
                i_target
            ) ** 2,
            dim=1,
        ).clamp_min(
            1e-12
        )

        return torch.mean(
            numerator / denominator
        )


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_data(
    batch_size: int = 32,
):
    """
    Load the controlled PINN-Cal dataset split.

    Train:
        800 samples

    Validation:
        200 samples

    Test:
        200 samples

    Test data is loaded but is never used for training
    or checkpoint selection.
    """

    # -------------------------------------------------------------
    # Load controlled splits
    # -------------------------------------------------------------

    train_data = np.load(
        "data/physics_train_800.npy",
        allow_pickle=True,
    ).item()

    val_data = np.load(
        "data/physics_val_200.npy",
        allow_pickle=True,
    ).item()

    test_data = np.load(
        "data/physics_test_200.npy",
        allow_pickle=True,
    ).item()

    # -------------------------------------------------------------
    # Training tensors
    # -------------------------------------------------------------

    X_train = torch.tensor(
        train_data["X"],
        dtype=torch.float32,
    )

    Y_train = torch.tensor(
        train_data["Y"],
        dtype=torch.float32,
    )

    # -------------------------------------------------------------
    # Validation tensors
    # -------------------------------------------------------------

    X_val = torch.tensor(
        val_data["X"],
        dtype=torch.float32,
    )

    Y_val = torch.tensor(
        val_data["Y"],
        dtype=torch.float32,
    )

    # -------------------------------------------------------------
    # Test tensors
    # -------------------------------------------------------------

    X_test = torch.tensor(
        test_data["X"],
        dtype=torch.float32,
    )

    Y_test = torch.tensor(
        test_data["Y"],
        dtype=torch.float32,
    )

    # -------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            X_train,
            Y_train,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            X_val,
            Y_val,
        ),
        batch_size=batch_size,
        shuffle=False,
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
        val_loader,
        test_loader,
        X_test,
        Y_test,
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model: PINNCalV3,
    loader,
    optimizer,
    lambda_physics: float,
):
    """
    Train for one epoch.
    """

    model.train()

    data_losses = []
    physics_losses = []
    total_losses = []

    criterion = nn.MSELoss()

    for X_batch, Y_batch in loader:

        X_batch = X_batch.to(
            DEVICE
        )

        Y_batch = Y_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        prediction = model(
            X_batch
        )

        # ---------------------------------------------------------
        # Supervised data loss
        # ---------------------------------------------------------

        data_loss = criterion(
            prediction,
            Y_batch,
        )

        # ---------------------------------------------------------
        # Physics loss
        # ---------------------------------------------------------

        physics_loss = model.physics_loss(
            X_batch,
            prediction,
        )

        # ---------------------------------------------------------
        # Combined objective
        # ---------------------------------------------------------

        total_loss = (
            data_loss
            + lambda_physics * physics_loss
        )

        total_loss.backward()

        # ---------------------------------------------------------
        # Gradient clipping
        # ---------------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        data_losses.append(
            float(
                data_loss.detach()
            )
        )

        physics_losses.append(
            float(
                physics_loss.detach()
            )
        )

        total_losses.append(
            float(
                total_loss.detach()
            )
        )

    return (
        float(
            np.mean(data_losses)
        ),
        float(
            np.mean(physics_losses)
        ),
        float(
            np.mean(total_losses)
        ),
    )


# ---------------------------------------------------------------------
# Validation data loss
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_data_loss(
    model: PINNCalV3,
    loader,
) -> float:
    """
    Evaluate coefficient-domain MSE.
    """

    model.eval()

    criterion = nn.MSELoss()

    losses = []

    for X_batch, Y_batch in loader:

        X_batch = X_batch.to(
            DEVICE
        )

        Y_batch = Y_batch.to(
            DEVICE
        )

        prediction = model(
            X_batch
        )

        losses.append(
            float(
                criterion(
                    prediction,
                    Y_batch,
                )
            )
        )

    return float(
        np.mean(losses)
    )


# ---------------------------------------------------------------------
# Validation physics loss
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_physics_loss(
    model: PINNCalV3,
    loader,
) -> float:
    """
    Evaluate coupled-current physics residual.
    """

    model.eval()

    losses = []

    for X_batch, _ in loader:

        X_batch = X_batch.to(
            DEVICE
        )

        prediction = model(
            X_batch
        )

        losses.append(
            float(
                model.physics_loss(
                    X_batch,
                    prediction,
                )
            )
        )

    return float(
        np.mean(losses)
    )


# ---------------------------------------------------------------------
# Main training procedure
# ---------------------------------------------------------------------

def main() -> None:

    # -------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------

    torch.manual_seed(42)

    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # -------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    print(
        f"Using device: {DEVICE}"
    )

    # -------------------------------------------------------------
    # Controlled dataset
    # -------------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        _,
        _,
    ) = load_data(
        batch_size=32,
    )

    # -------------------------------------------------------------
    # Explicit split sizes
    # -------------------------------------------------------------

    print(
        "Dataset split:"
    )

    print(
        "  Train      : 800 samples"
    )

    print(
        "  Validation : 200 samples"
    )

    print(
        "  Test       : 200 samples"
    )

    print(
        "  Test usage : FINAL EVALUATION ONLY"
    )

    # Keep the variable intentionally referenced so it is clear
    # that the test loader exists but is not used during training.
    _ = test_loader

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = PINNCalV3().to(
        DEVICE
    )

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: {parameters:,}"
    )

    print(
        f"Input dimension: {INPUT_DIM}"
    )

    print(
        f"Output dimension: {OUTPUT_DIM}"
    )

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    # -------------------------------------------------------------
    # Training configuration
    # -------------------------------------------------------------

    epochs = 200

    # -------------------------------------------------------------
    # Best validation checkpoint
    # -------------------------------------------------------------

    best_val_data_loss = float(
        "inf"
    )

    best_epoch = -1

    best_checkpoint_path = (
        "models/pinn_cal_final_best.pth"
    )

    last_checkpoint_path = (
        "models/pinn_cal_final_last.pth"
    )

    # -------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------

    start_time = time.time()

    # -------------------------------------------------------------
    # Training information
    # -------------------------------------------------------------

    print()
    print(
        "Final PINN training"
    )

    print(
        "Phase 1: data loss only"
    )

    print(
        "Phase 2: gradually increase physics loss"
    )

    print(
        "Model selection: validation data loss"
    )

    print(
        "Test set: untouched during training"
    )

    print()

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------

    for epoch in range(
        epochs
    ):

        # ---------------------------------------------------------
        # Physics-loss schedule
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Training
        # ---------------------------------------------------------

        (
            data_loss,
            physics_loss,
            total_loss,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lambda_physics,
        )

        # ---------------------------------------------------------
        # Validation
        #
        # Validation is used for model selection.
        # Test data is NOT evaluated here.
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

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Epoch {epoch + 1:3d} | "
                f"λ={lambda_physics:.3f} | "
                f"Data={data_loss:.6e} | "
                f"Physics={physics_loss:.6e} | "
                f"ValData={val_data_loss:.6e} | "
                f"ValPhysics={val_physics_loss:.6e} | "
                f"{elapsed:.1f}s"
            )

            # -----------------------------------------------------
            # Save best validation checkpoint
            # -----------------------------------------------------

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

                torch.save(
                    model.state_dict(),
                    best_checkpoint_path,
                )

                print(
                    "  -> New best checkpoint "
                    f"(epoch {best_epoch}, "
                    f"ValData="
                    f"{best_val_data_loss:.6e})"
                )

    # -------------------------------------------------------------
    # Save final state
    # -------------------------------------------------------------

    torch.save(
        model.state_dict(),
        last_checkpoint_path,
    )

    # -------------------------------------------------------------
    # Training summary
    # -------------------------------------------------------------

    print()
    print(
        "Training complete."
    )

    print(
        f"Best epoch        : "
        f"{best_epoch}"
    )

    print(
        f"Best ValData      : "
        f"{best_val_data_loss:.6e}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"  {best_checkpoint_path}"
    )

    print(
        f"  {last_checkpoint_path}"
    )

    print()
    print(
        "The test set was not evaluated during training."
    )

    print(
        "Final test evaluation will use the best checkpoint."
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()