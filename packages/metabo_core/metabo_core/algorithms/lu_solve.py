"""LU decomposition / inverse ported from MS-DIAL MatrixCalculate.cs.

Numerical-Recipes ``ludcmp`` (Crout with implicit pivoting). MSDec builds a
Gram matrix from near-collinear model basis vectors and needs its inverse's
row 0; the design is deliberately singular-tolerant:

  - an all-zero row returns ``None`` (caller falls back to a lower-order
    deconvolution pattern), and
  - a zero diagonal is forced to ``1e-10`` rather than raising.

numpy's ``inv`` raises ``LinAlgError`` on singular input, which is NOT
equivalent, so this routine is ported faithfully (handoff pitfall #6).

Reference: Common/CommonStandard/Mathematics/Matrix/MatrixCalculate.cs
L101-219.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class LUMatrix(NamedTuple):
    """Result of :func:`matrix_decompose` (MS-DIAL ``LUmatrix``)."""

    matrix: np.ndarray      # in-place LU (row-permuted)
    index_vector: np.ndarray  # original row index now living at row i
    reverse: float          # permutation sign (±1), used by determinant


def matrix_decompose(raw_matrix: np.ndarray) -> LUMatrix | None:
    """Crout LU with implicit pivoting (MatrixCalculate.cs L101-164).

    Returns ``None`` iff some row is entirely zero. The input is not
    mutated (MS-DIAL mutates its caller-built matrix; we copy for safety).
    """
    a = np.array(raw_matrix, dtype=np.float64, copy=True)
    element_size = a.shape[0]
    scaling_vector = np.empty(element_size, dtype=np.float64)
    index_vector = np.arange(element_size)
    d = 1.0

    for i in range(element_size):
        big = 0.0
        for j in range(element_size):
            temp = abs(a[i, j])
            if temp > big:
                big = temp
        if big == 0.0:
            return None  # singular matrix in routine ludcmp
        scaling_vector[i] = 1.0 / big

    for j in range(element_size):
        imax = j
        for i in range(j):
            s = a[i, j]
            for k in range(i):
                s -= a[i, k] * a[k, j]
            a[i, j] = s

        big = 0.0
        for i in range(j, element_size):
            s = a[i, j]
            for k in range(j):
                s -= a[i, k] * a[k, j]
            a[i, j] = s

            dum = scaling_vector[i] * abs(s)
            if dum >= big:
                big = dum
                imax = i

        if j != imax:
            for k in range(element_size):
                dum = a[imax, k]
                a[imax, k] = a[j, k]
                a[j, k] = dum
            d = -1.0 * d
            scaling_vector[imax], scaling_vector[j] = (
                scaling_vector[j],
                scaling_vector[imax],
            )
            index_vector[imax], index_vector[j] = (
                index_vector[j],
                index_vector[imax],
            )

        if a[j, j] == 0.0:
            a[j, j] = 10.0 ** -10
        if j != element_size:
            dum = 1.0 / a[j, j]
            for i in range(j + 1, element_size):
                a[i, j] *= dum

    return LUMatrix(matrix=a, index_vector=index_vector, reverse=d)


def helper_solve(lu: LUMatrix, b: np.ndarray) -> np.ndarray:
    """Forward/backward substitution for ``LU x = b`` (L167-190)."""
    a = lu.matrix
    n = a.shape[0]
    x = np.array(b, dtype=np.float64, copy=True)
    for i in range(1, n):
        s = x[i]
        for j in range(i):
            s -= a[i, j] * x[j]
        x[i] = s
    x[n - 1] /= a[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        s = x[i]
        for j in range(i + 1, n):
            s -= a[i, j] * x[j]
        x[i] = s / a[i, i]
    return x


def determinant_a(lu: LUMatrix) -> float:
    """Determinant from the decomposed matrix (L192-198)."""
    det = lu.reverse
    for i in range(lu.matrix.shape[0]):
        det *= lu.matrix[i, i]
    return float(det)


def matrix_inverse(lu: LUMatrix) -> np.ndarray:
    """Inverse via per-column solve, honouring the row permutation (L200-219)."""
    element_size = lu.matrix.shape[0]
    inverse = np.empty((element_size, element_size), dtype=np.float64)
    for j in range(element_size):
        col_vector = np.zeros(element_size, dtype=np.float64)
        for i in range(element_size):
            if j == lu.index_vector[i]:
                col_vector[i] = 1.0
        inverse_vector = helper_solve(lu, col_vector)
        for i in range(element_size):
            inverse[i, j] = inverse_vector[i]
    return inverse
