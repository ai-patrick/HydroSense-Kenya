"""
Test suite for numerical integration methods.

Verification strategy:
  - Known integrals: ∫sin(x)dx from 0 to π = 2
  - Polynomial exactness: trapezoidal exact for degree ≤ 1, Simpson for ≤ 3
  - Error order verification
  - Cross-verification against scipy.integrate.quad
"""

import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.numerical_methods import (
    trapezoidal_rule, trapezoidal_rule_func,
    simpson_rule, simpson_rule_func,
)


class TestTrapezoidalRule:

    def test_linear_function_exact(self):
        """Trapezoidal rule is exact for linear functions."""
        x = np.linspace(0, 10, 101)
        y = 3.0 * x + 2.0  # integral = 3*50 + 2*10 = 170
        result = trapezoidal_rule(y, dx=0.1)
        assert abs(result.value - 170.0) < 1e-10

    def test_sin_integral(self):
        """∫₀^π sin(x) dx = 2."""
        n = 1000
        x = np.linspace(0, math.pi, n + 1)
        y = np.sin(x)
        result = trapezoidal_rule(y, dx=math.pi / n)
        assert abs(result.value - 2.0) < 1e-5

    def test_callable_interface(self):
        result = trapezoidal_rule_func(math.sin, 0, math.pi, n=1000)
        assert abs(result.value - 2.0) < 1e-5

    def test_error_order_h_squared(self):
        """Error should decrease as O(h^2) when n is doubled."""
        exact = 2.0
        e1 = abs(trapezoidal_rule_func(math.sin, 0, math.pi, n=100).value - exact)
        e2 = abs(trapezoidal_rule_func(math.sin, 0, math.pi, n=200).value - exact)
        ratio = e1 / e2
        # For O(h^2), doubling n should reduce error by factor ~4
        assert 3.0 < ratio < 5.0


class TestSimpsonRule:

    def test_cubic_function_exact(self):
        """Simpson's rule is exact for polynomials up to degree 3."""
        x = np.linspace(0, 2, 101)
        y = x ** 3  # integral = 2^4/4 = 4
        result = simpson_rule(y, dx=0.02)
        assert abs(result.value - 4.0) < 1e-8

    def test_sin_integral(self):
        """∫₀^π sin(x) dx = 2."""
        n = 100
        x = np.linspace(0, math.pi, n + 1)
        y = np.sin(x)
        result = simpson_rule(y, dx=math.pi / n)
        assert abs(result.value - 2.0) < 5e-8

    def test_more_accurate_than_trapezoidal(self):
        """Simpson should be more accurate than trapezoidal for same n."""
        n = 50
        x = np.linspace(0, math.pi, n + 1)
        y = np.sin(x)
        dx = math.pi / n
        trap_err = abs(trapezoidal_rule(y, dx).value - 2.0)
        simp_err = abs(simpson_rule(y, dx).value - 2.0)
        assert simp_err < trap_err

    def test_callable_interface(self):
        result = simpson_rule_func(math.sin, 0, math.pi, n=100)
        assert abs(result.value - 2.0) < 5e-8


class TestCrossVerification:

    def test_vs_scipy_quad(self):
        from scipy.integrate import quad
        scipy_val, _ = quad(math.sin, 0, math.pi)
        our_val = simpson_rule_func(math.sin, 0, math.pi, n=200).value
        assert abs(scipy_val - our_val) < 1e-8
