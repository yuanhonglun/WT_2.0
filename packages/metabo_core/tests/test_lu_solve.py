"""Tests for the ludcmp-style LU solve ported from MS-DIAL MatrixCalculate.cs.

MSDec builds a Gram matrix from (near-collinear) model basis vectors and
needs its inverse's row 0. MS-DIAL uses a Numerical-Recipes ``ludcmp`` with
implicit pivoting that (a) returns null only for an all-zero row and
(b) forces a 1e-10 pivot on a zero diagonal rather than throwing. numpy's
inv raises on singular input, which is NOT equivalent, so the behaviour is
ported faithfully (handoff pitfall #6).

Reference: Common/CommonStandard/Mathematics/Matrix/MatrixCalculate.cs
L101-219.
"""
from __future__ import annotations

import numpy as np

from metabo_core.algorithms.lu_solve import (
    matrix_decompose,
    matrix_inverse,
    determinant_a,
)


def test_inverse_matches_numpy_on_well_conditioned_symmetric():
    a = np.array([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]])
    lu = matrix_decompose(a.copy())
    assert lu is not None
    inv = matrix_inverse(lu)
    np.testing.assert_allclose(inv, np.linalg.inv(a), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(inv @ a, np.eye(3), atol=1e-9)


def test_decompose_does_not_mutate_caller_array():
    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    a_copy = a.copy()
    matrix_decompose(a)
    np.testing.assert_array_equal(a, a_copy)


def test_identity_inverse_is_identity():
    lu = matrix_decompose(np.eye(4))
    assert lu is not None
    np.testing.assert_allclose(matrix_inverse(lu), np.eye(4), atol=1e-12)


def test_determinant_matches_numpy():
    a = np.array([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]])
    lu = matrix_decompose(a.copy())
    assert determinant_a(lu) == np.linalg.det(a) or abs(
        determinant_a(lu) - np.linalg.det(a)
    ) < 1e-6


def test_all_zero_row_returns_none():
    # An all-zero row makes the scaling 1/big undefined → MS-DIAL returns null.
    a = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
    assert matrix_decompose(a) is None


def test_singular_nonzero_rows_forces_pivot_not_crash():
    # Rank-1 matrix with no zero row: ludcmp hits a zero diagonal mid-way and
    # forces a 1e-10 pivot instead of failing. det is tiny but nonzero.
    a = np.array([[1.0, 1.0], [1.0, 1.0]])
    lu = matrix_decompose(a)
    assert lu is not None
    det = determinant_a(lu)
    assert det != 0.0
    assert abs(det) < 1e-6


def test_inverse_solves_target_coefficient_row0():
    # The actual MSDec use: invMatrix[0, :] dotted with z gives the target
    # coefficient. Verify row 0 of the inverse solves A x = e0.
    a = np.array([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]])
    lu = matrix_decompose(a.copy())
    inv = matrix_inverse(lu)
    # inv[0, :] @ a should be e0 = [1, 0, 0].
    np.testing.assert_allclose(inv[0, :] @ a, np.array([1.0, 0.0, 0.0]), atol=1e-9)
