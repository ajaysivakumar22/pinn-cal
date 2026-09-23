"""
PINN-Cal — Coupling Uncertainty Model

Separates the nominal coupling model available to PINN-Cal from the
unknown true coupling matrix used to generate physical observations.

Properties:
    - Reciprocal: Z_true = Z_true.T
    - Complex-valued
    - Controlled parameter mismatch
    - Deterministic when a seed is supplied
    - Validated for numerical stability and positive-real behavior

This is still a simplified network model, not a full-wave EM solver.
"""

from __future__ import annotations

import torch

from physics_core import (
    ArrayConfig,
    DEFAULT_CONFIG,
    mutual_impedance_matrix,
)


def generate_true_coupling_matrix(
    config: ArrayConfig = DEFAULT_CONFIG,
    mismatch_fraction: float = 0.20,
    seed: int | None = None,
    nominal_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate a perturbed true coupling matrix.

    Each independent impedance entry is multiplied by a real-valued
    factor drawn uniformly from:

        [1 - mismatch_fraction, 1 + mismatch_fraction]

    Reciprocity is explicitly preserved:

        Z_true[i,j] = Z_true[j,i]

    Parameters
    ----------
    mismatch_fraction:
        0.20 means ±20% entry-wise magnitude mismatch.

    seed:
        Optional deterministic random seed.

    nominal_matrix:
        Optional nominal matrix. If omitted, it is generated from the
        physics core.
    """
    if mismatch_fraction < 0.0:
        raise ValueError("mismatch_fraction must be non-negative.")

    if mismatch_fraction >= 1.0:
        raise ValueError(
            "mismatch_fraction must be smaller than 1.0."
        )

    if nominal_matrix is None:
        nominal = mutual_impedance_matrix(config)
    else:
        nominal = nominal_matrix.clone()

    nominal = nominal.to(torch.complex128).cpu()

    generator = torch.Generator(device="cpu")

    if seed is not None:
        generator.manual_seed(seed)

    low = 1.0 - mismatch_fraction
    high = 1.0 + mismatch_fraction

    true_matrix = nominal.clone()

    m = config.num_elements

    # Perturb diagonal/self impedances independently.
    diagonal_factors = low + (
        high - low
    ) * torch.rand(
        m,
        generator=generator,
        dtype=torch.float64,
    )

    for i in range(m):
        true_matrix[i, i] = (
            nominal[i, i]
            * diagonal_factors[i]
        )

    # Perturb mutual terms while preserving reciprocity.
    for i in range(m):
        for j in range(i + 1, m):
            factor = low + (
                high - low
            ) * torch.rand(
                (),
                generator=generator,
                dtype=torch.float64,
            )

            true_matrix[i, j] = (
                nominal[i, j] * factor
            )

            true_matrix[j, i] = true_matrix[i, j]

    return true_matrix


def coupling_relative_error(
    nominal: torch.Tensor,
    true: torch.Tensor,
) -> float:
    """Return relative Frobenius-norm difference."""
    nominal = nominal.to(torch.complex128)
    true = true.to(torch.complex128)

    return float(
        torch.linalg.vector_norm(true - nominal)
        / (
            torch.linalg.vector_norm(nominal)
            + 1e-12
        )
    )


def reciprocity_error(
    matrix: torch.Tensor,
) -> float:
    """Return max absolute reciprocity violation."""
    matrix = matrix.to(torch.complex128)

    return float(
        torch.max(
            torch.abs(matrix - matrix.T)
        )
    )


def hermitian_part_eigenvalues(
    matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Return eigenvalues of the Hermitian/real part used for the
    positive-real sanity check.
    """
    matrix = matrix.to(torch.complex128)

    hermitian_part = (
        matrix + matrix.conj().T
    ) / 2.0

    return torch.linalg.eigvalsh(
        hermitian_part
    ).real


def coupling_condition_number(
    matrix: torch.Tensor,
) -> float:
    """Return the numerical condition number."""
    matrix = matrix.to(torch.complex128)

    return float(
        torch.linalg.cond(matrix).real
    )


def validate_coupling_matrix(
    matrix: torch.Tensor,
    max_condition_number: float = 100.0,
) -> dict[str, float | bool]:
    """
    Validate reciprocal and numerical properties of a coupling matrix.
    """
    matrix = matrix.to(torch.complex128)

    eigenvalues = hermitian_part_eigenvalues(
        matrix
    )

    condition_number = coupling_condition_number(
        matrix
    )

    rec_error = reciprocity_error(
        matrix
    )

    finite = bool(
        torch.isfinite(matrix.real).all()
        and torch.isfinite(matrix.imag).all()
    )

    positive_real = bool(
        eigenvalues.min() >= 0.0
    )

    well_conditioned = bool(
        condition_number <= max_condition_number
    )

    return {
        "finite": finite,
        "reciprocal": rec_error <= 1e-12,
        "positive_real": positive_real,
        "well_conditioned": well_conditioned,
        "reciprocity_error": rec_error,
        "min_hermitian_eigenvalue": float(
            eigenvalues.min()
        ),
        "max_hermitian_eigenvalue": float(
            eigenvalues.max()
        ),
        "condition_number": condition_number,
    }