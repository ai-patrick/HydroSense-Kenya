"""
Numerical Methods Engine for HydroSense-Kenya

Hand-rolled implementations of root-finding, numerical differentiation,
numerical integration, and linear system solvers. SciPy is used ONLY
for cross-verification in the test suite — never in production paths.

Every solver returns structured results including iteration history,
enabling convergence analysis and method comparison in notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RootResult:
    """Structured output for root-finding methods."""
    root: float
    converged: bool
    iterations: int
    error_history: list[float] = field(default_factory=list)
    method: str = ""
    message: str = ""


@dataclass
class IntegrationResult:
    """Structured output for numerical integration."""
    value: float
    n_subintervals: int
    method: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# ROOT FINDING
# ═══════════════════════════════════════════════════════════════════════════

def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> RootResult:
    """Find root of f in [a, b] via bisection.

    Convergence: Linear. After k iterations the bracket width is
    (b - a) / 2^k, so iterations needed = ceil(log2((b-a)/tol)).

    Guaranteed convergence if f(a)*f(b) < 0 and f is continuous (IVT).
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return RootResult(root=np.nan, converged=False, iterations=0,
                          method="bisection",
                          message="f(a) and f(b) have the same sign — no bracket")
    if abs(fa) < tol:
        return RootResult(root=a, converged=True, iterations=0,
                          error_history=[abs(fa)], method="bisection")
    if abs(fb) < tol:
        return RootResult(root=b, converged=True, iterations=0,
                          error_history=[abs(fb)], method="bisection")

    errors: list[float] = []
    for i in range(1, max_iter + 1):
        mid = a + (b - a) / 2.0  # Avoids overflow vs (a+b)/2
        fm = f(mid)
        errors.append(abs(fm))

        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return RootResult(root=mid, converged=True, iterations=i,
                              error_history=errors, method="bisection")
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm

    return RootResult(root=a + (b - a) / 2.0, converged=False,
                      iterations=max_iter, error_history=errors,
                      method="bisection",
                      message=f"Did not converge in {max_iter} iterations")


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> RootResult:
    """Find root of f using Newton-Raphson iteration.

    Convergence: Quadratic near simple roots (error_{k+1} ~ C * error_k^2).
    Fails when f'(x) ≈ 0 (division instability) or when the iterate
    escapes the basin of attraction.
    """
    x = x0
    errors: list[float] = []

    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        errors.append(abs(fx))

        if abs(dfx) < np.finfo(float).eps:
            return RootResult(root=x, converged=False, iterations=i,
                              error_history=errors, method="newton_raphson",
                              message="Derivative near zero — iteration unstable")

        x_new = x - fx / dfx

        if abs(x_new - x) < tol:
            errors.append(abs(f(x_new)))
            return RootResult(root=x_new, converged=True, iterations=i,
                              error_history=errors, method="newton_raphson")
        x = x_new

    return RootResult(root=x, converged=False, iterations=max_iter,
                      error_history=errors, method="newton_raphson",
                      message=f"Did not converge in {max_iter} iterations")


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> RootResult:
    """Find root of f using secant method.

    Convergence: Superlinear (order ≈ 1.618, the golden ratio).
    No derivative required — approximates f' from two prior evaluations.
    Fails when consecutive iterates coincide (zero denominator).
    """
    f0, f1 = f(x0), f(x1)
    errors: list[float] = [abs(f0), abs(f1)]

    for i in range(1, max_iter + 1):
        denom = f1 - f0
        if abs(denom) < np.finfo(float).eps:
            return RootResult(root=x1, converged=False, iterations=i,
                              error_history=errors, method="secant",
                              message="Consecutive f-values coincide")

        x2 = x1 - f1 * (x1 - x0) / denom
        f2 = f(x2)
        errors.append(abs(f2))

        if abs(x2 - x1) < tol:
            return RootResult(root=x2, converged=True, iterations=i,
                              error_history=errors, method="secant")

        x0, f0 = x1, f1
        x1, f1 = x2, f2

    return RootResult(root=x1, converged=False, iterations=max_iter,
                      error_history=errors, method="secant",
                      message=f"Did not converge in {max_iter} iterations")


# ═══════════════════════════════════════════════════════════════════════════
# NUMERICAL DIFFERENTIATION
# ═══════════════════════════════════════════════════════════════════════════

def forward_difference(
    f: Callable[[float], float], x: float, h: float = 1e-5,
) -> float:
    """First-order forward difference approximation: O(h) truncation error."""
    return (f(x + h) - f(x)) / h


def backward_difference(
    f: Callable[[float], float], x: float, h: float = 1e-5,
) -> float:
    """First-order backward difference approximation: O(h) truncation error."""
    return (f(x) - f(x - h)) / h


def central_difference(
    f: Callable[[float], float], x: float, h: float = 1e-5,
) -> float:
    """Second-order central difference approximation: O(h^2) truncation error.

    The leading error term cancels due to symmetry of the stencil,
    yielding one order of magnitude better accuracy than forward/backward
    for the same step size h.
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def differentiate_series(
    values: np.ndarray, dt: float = 1.0, method: str = "central",
) -> np.ndarray:
    """Compute numerical derivative of a discrete time series.

    Uses central differences for interior points, and one-sided
    differences at boundaries (unavoidable for finite data).
    """
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 data points for differentiation")

    deriv = np.zeros(n)
    deriv[0] = (values[1] - values[0]) / dt
    deriv[-1] = (values[-1] - values[-2]) / dt

    if method == "central" and n > 2:
        deriv[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    elif method == "forward":
        deriv[:-1] = (values[1:] - values[:-1]) / dt
    elif method == "backward":
        deriv[1:] = (values[1:] - values[:-1]) / dt
    else:
        deriv[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)

    return deriv


# ═══════════════════════════════════════════════════════════════════════════
# NUMERICAL INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def trapezoidal_rule(
    f_values: np.ndarray, dx: float = 1.0,
) -> IntegrationResult:
    """Composite trapezoidal rule for equally-spaced data.

    Error: O(h^2) — exact for linear functions.
    For n+1 data points: integral ≈ h * [f_0/2 + f_1 + ... + f_{n-1} + f_n/2].
    """
    n = len(f_values) - 1
    if n < 1:
        raise ValueError("Need at least 2 data points for integration")
    val = dx * (f_values[0] / 2.0 + np.sum(f_values[1:-1]) + f_values[-1] / 2.0)
    return IntegrationResult(value=float(val), n_subintervals=n,
                             method="trapezoidal")


def trapezoidal_rule_func(
    f: Callable[[float], float], a: float, b: float, n: int = 100,
) -> IntegrationResult:
    """Composite trapezoidal rule for a callable function."""
    x = np.linspace(a, b, n + 1)
    fx = np.array([f(xi) for xi in x])
    return trapezoidal_rule(fx, dx=(b - a) / n)


def simpson_rule(
    f_values: np.ndarray, dx: float = 1.0,
) -> IntegrationResult:
    """Composite Simpson's 1/3 rule for equally-spaced data.

    Error: O(h^4) — exact for polynomials up to degree 3.
    Requires an even number of subintervals (odd number of data points).
    If n is odd (even number of points), we apply Simpson's 3/8 rule
    to the last 3 subintervals to handle the remainder.
    """
    n = len(f_values) - 1
    if n < 2:
        raise ValueError("Need at least 3 data points for Simpson's rule")

    if n % 2 == 0:
        # Standard Simpson's 1/3
        val = f_values[0] + f_values[-1]
        val += 4.0 * np.sum(f_values[1:-1:2])
        val += 2.0 * np.sum(f_values[2:-1:2])
        val *= dx / 3.0
    else:
        # Apply 1/3 rule to first n-3 intervals, 3/8 rule to last 3
        val_main = f_values[0] + f_values[n - 3]
        val_main += 4.0 * np.sum(f_values[1:n - 3:2])
        val_main += 2.0 * np.sum(f_values[2:n - 3:2])
        val_main *= dx / 3.0

        # Simpson's 3/8 for the last 4 points
        val_tail = (3.0 * dx / 8.0) * (
            f_values[n - 3] + 3.0 * f_values[n - 2]
            + 3.0 * f_values[n - 1] + f_values[n]
        )
        val = val_main + val_tail

    return IntegrationResult(value=float(val), n_subintervals=n,
                             method="simpson")


def simpson_rule_func(
    f: Callable[[float], float], a: float, b: float, n: int = 100,
) -> IntegrationResult:
    """Composite Simpson's rule for a callable function."""
    if n % 2 != 0:
        n += 1
    x = np.linspace(a, b, n + 1)
    fx = np.array([f(xi) for xi in x])
    return simpson_rule(fx, dx=(b - a) / n)


# ═══════════════════════════════════════════════════════════════════════════
# LINEAR SYSTEMS — Gaussian Elimination with Partial Pivoting
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_elimination(
    A: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Partial pivoting selects the largest absolute value in each column
    as the pivot, reducing round-off amplification. Without pivoting,
    small diagonal elements produce large multipliers that amplify
    floating-point errors catastrophically.

    Complexity: O(n^3) for the elimination phase, O(n^2) for back-substitution.
    """
    n = len(b)
    A = A.astype(np.float64).copy()
    b = b.astype(np.float64).copy()

    # Forward elimination with partial pivoting
    for k in range(n - 1):
        # Find pivot
        max_row = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[max_row, k]) < np.finfo(float).eps * np.max(np.abs(A)):
            raise np.linalg.LinAlgError(
                f"Matrix is singular or near-singular at column {k}")

        # Swap rows
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        # Eliminate below pivot
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k + 1:] -= factor * A[k, k + 1:]
            A[i, k] = 0.0
            b[i] -= factor * b[k]

    # Back substitution
    x = backward_substitution(A, b)
    return x


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve lower-triangular system Lx = b. Complexity: O(n^2)."""
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if abs(L[i, i]) < np.finfo(float).eps:
            raise np.linalg.LinAlgError(f"Zero diagonal at row {i}")
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x


def backward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve upper-triangular system Ux = b. Complexity: O(n^2)."""
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < np.finfo(float).eps:
            raise np.linalg.LinAlgError(f"Zero diagonal at row {i}")
        x[i] = (b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def condition_number(A: np.ndarray) -> float:
    """Estimate condition number using infinity norm: cond(A) = ||A|| * ||A^-1||.

    A large condition number warns that the solution is sensitive to
    perturbations in the input data — critical for the 3-zone water
    allocation problem where measurement errors propagate.
    """
    try:
        A_inv = np.linalg.inv(A.astype(np.float64))
        norm_A = np.max(np.sum(np.abs(A), axis=1))
        norm_A_inv = np.max(np.sum(np.abs(A_inv), axis=1))
        return float(norm_A * norm_A_inv)
    except np.linalg.LinAlgError:
        return np.inf
