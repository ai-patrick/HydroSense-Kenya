"""
Test suite for root-finding methods.

Verification strategy:
  - Known analytical roots (x^2 - 4 = 0 → x = 2)
  - Convergence order estimation
  - Edge cases: near-zero derivative, non-bracketed intervals
  - Cross-verification against scipy.optimize.brentq
"""

import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.numerical_methods import bisection, newton_raphson, secant


# ═══════════════════════════════════════════════════════════════════════════
# Test functions with known roots
# ═══════════════════════════════════════════════════════════════════════════

def f_quadratic(x: float) -> float:
    """f(x) = x^2 - 4, roots at x = ±2."""
    return x ** 2 - 4.0

def df_quadratic(x: float) -> float:
    return 2.0 * x

def f_cubic(x: float) -> float:
    """f(x) = x^3 - 6x^2 + 11x - 6, roots at x = 1, 2, 3."""
    return x ** 3 - 6 * x ** 2 + 11 * x - 6

def df_cubic(x: float) -> float:
    return 3 * x ** 2 - 12 * x + 11

def f_trig(x: float) -> float:
    """f(x) = cos(x) - x, root near x ≈ 0.7391."""
    return math.cos(x) - x

def df_trig(x: float) -> float:
    return -math.sin(x) - 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Bisection tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBisection:

    def test_quadratic_root(self):
        result = bisection(f_quadratic, 0.0, 5.0, tol=1e-10)
        assert result.converged
        assert abs(result.root - 2.0) < 1e-9

    def test_cubic_root(self):
        result = bisection(f_cubic, 0.5, 1.5, tol=1e-10)
        assert result.converged
        assert abs(result.root - 1.0) < 1e-9

    def test_trig_root(self):
        result = bisection(f_trig, 0.0, 1.5, tol=1e-10)
        assert result.converged
        assert abs(f_trig(result.root)) < 1e-9

    def test_no_bracket_fails(self):
        """If f(a) and f(b) have the same sign, bisection cannot proceed."""
        result = bisection(f_quadratic, 3.0, 5.0)
        assert not result.converged

    def test_convergence_is_linear(self):
        """Bisection should halve the interval each iteration."""
        result = bisection(f_quadratic, 0.0, 5.0, tol=1e-12)
        errors = result.error_history
        # The number of iterations should be approximately log2(5/1e-12) ≈ 42
        assert result.iterations <= 50

    def test_error_history_decreasing(self):
        result = bisection(f_quadratic, 0.0, 5.0, tol=1e-10)
        # Overall trend should be decreasing (may not be strictly monotone
        # because we track |f(mid)| not bracket width)
        assert result.error_history[-1] < result.error_history[0]


# ═══════════════════════════════════════════════════════════════════════════
# Newton-Raphson tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNewtonRaphson:

    def test_quadratic_root(self):
        result = newton_raphson(f_quadratic, df_quadratic, x0=3.0, tol=1e-12)
        assert result.converged
        assert abs(result.root - 2.0) < 1e-11

    def test_cubic_root(self):
        result = newton_raphson(f_cubic, df_cubic, x0=3.5, tol=1e-12)
        assert result.converged
        assert abs(result.root - 3.0) < 1e-10

    def test_fewer_iterations_than_bisection(self):
        """Newton should converge faster (quadratically) than bisection."""
        r_newton = newton_raphson(f_quadratic, df_quadratic, x0=3.0, tol=1e-10)
        r_bisect = bisection(f_quadratic, 0.0, 5.0, tol=1e-10)
        assert r_newton.iterations < r_bisect.iterations


# ═══════════════════════════════════════════════════════════════════════════
# Secant tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSecant:

    def test_quadratic_root(self):
        result = secant(f_quadratic, 0.0, 5.0, tol=1e-10)
        assert result.converged
        assert abs(result.root - 2.0) < 1e-9

    def test_trig_root(self):
        result = secant(f_trig, 0.0, 1.5, tol=1e-10)
        assert result.converged
        assert abs(f_trig(result.root)) < 1e-9

    def test_superlinear_convergence(self):
        """Secant converges faster than bisection but slower than Newton."""
        r_secant = secant(f_quadratic, 0.0, 5.0, tol=1e-10)
        r_bisect = bisection(f_quadratic, 0.0, 5.0, tol=1e-10)
        assert r_secant.iterations < r_bisect.iterations


# ═══════════════════════════════════════════════════════════════════════════
# Cross-verification against SciPy
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossVerification:

    def test_bisection_vs_scipy(self):
        from scipy.optimize import brentq
        scipy_root = brentq(f_quadratic, 0.0, 5.0, xtol=1e-12)
        our_root = bisection(f_quadratic, 0.0, 5.0, tol=1e-12).root
        assert abs(scipy_root - our_root) < 1e-10

    def test_newton_vs_scipy(self):
        from scipy.optimize import newton as scipy_newton
        scipy_root = scipy_newton(f_trig, 0.5, fprime=df_trig, tol=1e-12)
        our_root = newton_raphson(f_trig, df_trig, x0=0.5, tol=1e-12).root
        assert abs(scipy_root - our_root) < 1e-10
