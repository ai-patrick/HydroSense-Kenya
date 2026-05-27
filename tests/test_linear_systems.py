"""
Test suite for linear system solvers.

Verification strategy:
  - Known solution systems
  - Partial pivoting correctness
  - Near-singular matrix handling
  - Cross-verification against numpy.linalg.solve
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.numerical_methods import (
    gaussian_elimination,
    forward_substitution,
    backward_substitution,
    condition_number,
)


class TestGaussianElimination:

    def test_simple_2x2(self):
        """Solve [2,1;5,7]x = [11,13] → x = [4.555..., 1.888...]."""
        A = np.array([[2.0, 1.0], [5.0, 7.0]])
        b = np.array([11.0, 13.0])
        x = gaussian_elimination(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, atol=1e-12)

    def test_3x3_water_allocation(self):
        """Realistic 3-zone water allocation problem."""
        # Zone constraints: total supply, inter-zone transfer, efficiency
        A = np.array([
            [1.0, 1.0, 1.0],     # Total supply = 100
            [0.8, -0.2, 0.0],    # Zone A gets 80% of its share minus transfer
            [0.0, 0.3, 0.7],     # Zone C gets weighted mix
        ])
        b = np.array([100.0, 30.0, 45.0])
        x = gaussian_elimination(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, atol=1e-10)

    def test_partial_pivoting(self):
        """System requiring row swap for numerical stability."""
        A = np.array([
            [1e-15, 1.0],
            [1.0, 1.0],
        ])
        b = np.array([1.0, 2.0])
        x = gaussian_elimination(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, atol=1e-8)

    def test_singular_matrix_raises(self):
        """Singular matrix should raise LinAlgError."""
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        b = np.array([3.0, 6.0])
        with pytest.raises(np.linalg.LinAlgError):
            gaussian_elimination(A, b)

    def test_identity_matrix(self):
        """Identity matrix: x should equal b."""
        n = 4
        A = np.eye(n)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = gaussian_elimination(A, b)
        np.testing.assert_allclose(x, b, atol=1e-14)


class TestSubstitution:

    def test_forward_substitution(self):
        L = np.array([[2.0, 0.0, 0.0],
                       [1.0, 3.0, 0.0],
                       [4.0, 2.0, 1.0]])
        b = np.array([4.0, 7.0, 16.0])
        x = forward_substitution(L, b)
        np.testing.assert_allclose(L @ x, b, atol=1e-12)

    def test_backward_substitution(self):
        U = np.array([[3.0, 2.0, 1.0],
                       [0.0, 4.0, 2.0],
                       [0.0, 0.0, 5.0]])
        b = np.array([14.0, 14.0, 10.0])
        x = backward_substitution(U, b)
        np.testing.assert_allclose(U @ x, b, atol=1e-12)


class TestConditionNumber:

    def test_identity_condition(self):
        """Identity matrix has condition number 1."""
        assert abs(condition_number(np.eye(3)) - 1.0) < 1e-10

    def test_ill_conditioned(self):
        """Hilbert matrix is notoriously ill-conditioned."""
        from scipy.linalg import hilbert
        H = hilbert(5)
        cond = condition_number(H)
        assert cond > 1e4  # Should be very large
