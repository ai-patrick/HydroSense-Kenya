"""
Optimization Engine for HydroSense-Kenya

Constrained irrigation scheduling via penalised gradient descent.

Objective: Minimize total water use sum(I_t) subject to S_t >= S_min for all t.

The constraint is enforced via a quadratic penalty:
    L(I) = sum(I_t) + lambda * sum(max(0, S_min - S_t(I))^2)

Gradient descent with Armijo backtracking line search adjusts the
irrigation schedule iteratively. The penalty weight lambda controls
the water-conservation vs crop-safety tradeoff — sweeping lambda
produces a Pareto frontier for decision makers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from simulation import (
    compute_et,
    simulate_water_balance,
    SimulationResult,
)


@dataclass
class OptimizationResult:
    """Output container for irrigation optimization."""
    irrigation_schedule: np.ndarray
    total_water_used: float
    objective_history: list[float] = field(default_factory=list)
    constraint_violation: float = 0.0
    converged: bool = False
    iterations: int = 0
    final_moisture: Optional[np.ndarray] = None


def compute_objective(
    irrigation: np.ndarray,
    s0: float,
    rainfall: np.ndarray,
    et: np.ndarray,
    field_capacity: float,
    drainage_coeff: float,
    min_moisture: float,
    penalty_weight: float,
) -> tuple[float, np.ndarray]:
    """Evaluate penalised objective and return (cost, soil_moisture_trajectory).

    Cost = sum(irrigation) + lambda * sum(max(0, S_min - S_t)^2)
    """
    result = simulate_water_balance(
        n_days=len(irrigation), s0=s0,
        rainfall=rainfall, irrigation=irrigation, et=et,
        field_capacity=field_capacity, drainage_coeff=drainage_coeff,
        method="rk4",
    )
    moisture = result.soil_moisture

    # Penalty for moisture below threshold
    violation = np.maximum(0.0, min_moisture - moisture[1:])
    penalty = penalty_weight * np.sum(violation ** 2)

    cost = np.sum(irrigation) + penalty
    return float(cost), moisture


def compute_gradient_fd(
    irrigation: np.ndarray,
    s0: float,
    rainfall: np.ndarray,
    et: np.ndarray,
    field_capacity: float,
    drainage_coeff: float,
    min_moisture: float,
    penalty_weight: float,
    epsilon: float = 0.01,
) -> np.ndarray:
    """Estimate gradient of objective via central finite differences.

    Each component requires 2 function evaluations, so the total cost
    is O(2 * n_days) simulations per gradient call. For n_days = 30
    this is manageable; for larger problems, adjoint methods would be
    needed.
    """
    n = len(irrigation)
    grad = np.zeros(n)
    base_cost, _ = compute_objective(
        irrigation, s0, rainfall, et,
        field_capacity, drainage_coeff, min_moisture, penalty_weight,
    )

    for i in range(n):
        irr_plus = irrigation.copy()
        irr_minus = irrigation.copy()
        irr_plus[i] += epsilon
        irr_minus[i] = max(0.0, irr_minus[i] - epsilon)

        cost_plus, _ = compute_objective(
            irr_plus, s0, rainfall, et,
            field_capacity, drainage_coeff, min_moisture, penalty_weight,
        )
        cost_minus, _ = compute_objective(
            irr_minus, s0, rainfall, et,
            field_capacity, drainage_coeff, min_moisture, penalty_weight,
        )
        grad[i] = (cost_plus - cost_minus) / (2.0 * epsilon)

    return grad


def armijo_backtrack(
    irrigation: np.ndarray,
    grad: np.ndarray,
    current_cost: float,
    s0: float,
    rainfall: np.ndarray,
    et: np.ndarray,
    field_capacity: float,
    drainage_coeff: float,
    min_moisture: float,
    penalty_weight: float,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    max_backtrack: int = 20,
) -> float:
    """Armijo backtracking line search for step size selection.

    Ensures sufficient decrease: f(x - alpha*grad) <= f(x) - c*alpha*||grad||^2.
    """
    alpha = alpha_init
    grad_norm_sq = np.dot(grad, grad)

    for _ in range(max_backtrack):
        irr_trial = np.maximum(0.0, irrigation - alpha * grad)
        trial_cost, _ = compute_objective(
            irr_trial, s0, rainfall, et,
            field_capacity, drainage_coeff, min_moisture, penalty_weight,
        )
        if trial_cost <= current_cost - c * alpha * grad_norm_sq:
            return alpha
        alpha *= beta

    return alpha


def gradient_descent_irrigation(
    n_days: int,
    s0: float,
    rainfall: np.ndarray,
    et: np.ndarray,
    field_capacity: float,
    drainage_coeff: float,
    min_moisture: float,
    penalty_weight: float = 100.0,
    max_iter: int = 200,
    tol: float = 1e-4,
    initial_irrigation: Optional[np.ndarray] = None,
) -> OptimizationResult:
    """Optimize irrigation schedule via penalised gradient descent.

    Parameters
    ----------
    n_days : int
        Planning horizon.
    s0 : float
        Initial soil moisture (%).
    rainfall, et : np.ndarray
        Daily forcing arrays.
    field_capacity, drainage_coeff : float
        Soil parameters.
    min_moisture : float
        Minimum acceptable moisture (crop stress threshold).
    penalty_weight : float
        Lambda for constraint violation penalty.
    max_iter : int
        Maximum gradient descent iterations.
    tol : float
        Convergence tolerance on relative cost change.
    initial_irrigation : np.ndarray, optional
        Starting irrigation schedule. Defaults to uniform 2.0 mm/day.
    """
    if initial_irrigation is None:
        irrigation = np.full(n_days, 2.0)
    else:
        irrigation = initial_irrigation.copy()

    cost_history: list[float] = []
    cost, moisture = compute_objective(
        irrigation, s0, rainfall, et,
        field_capacity, drainage_coeff, min_moisture, penalty_weight,
    )
    cost_history.append(cost)

    converged = False
    for it in range(1, max_iter + 1):
        grad = compute_gradient_fd(
            irrigation, s0, rainfall, et,
            field_capacity, drainage_coeff, min_moisture, penalty_weight,
        )

        if np.linalg.norm(grad) < tol:
            converged = True
            break

        alpha = armijo_backtrack(
            irrigation, grad, cost, s0, rainfall, et,
            field_capacity, drainage_coeff, min_moisture, penalty_weight,
        )

        irrigation = np.maximum(0.0, irrigation - alpha * grad)
        cost, moisture = compute_objective(
            irrigation, s0, rainfall, et,
            field_capacity, drainage_coeff, min_moisture, penalty_weight,
        )
        cost_history.append(cost)

        if len(cost_history) > 1:
            rel_change = abs(cost_history[-1] - cost_history[-2])
            if cost_history[-2] > 0:
                rel_change /= cost_history[-2]
            if rel_change < tol:
                converged = True
                break

    violation = np.sum(np.maximum(0.0, min_moisture - moisture[1:]))

    return OptimizationResult(
        irrigation_schedule=irrigation,
        total_water_used=float(np.sum(irrigation)),
        objective_history=cost_history,
        constraint_violation=float(violation),
        converged=converged,
        iterations=it if converged else max_iter,
        final_moisture=moisture,
    )


def pareto_tradeoff_analysis(
    n_days: int,
    s0: float,
    rainfall: np.ndarray,
    et: np.ndarray,
    field_capacity: float,
    drainage_coeff: float,
    min_moisture: float,
    lambda_values: Optional[np.ndarray] = None,
    max_iter: int = 150,
) -> list[dict]:
    """Sweep penalty weight to trace the Pareto frontier.

    Each lambda value produces a different tradeoff point:
        - Low lambda: minimal water use, higher crop-stress risk
        - High lambda: near-zero stress, more water consumed

    Returns list of dicts with lambda, total_water, max_violation, converged.
    """
    if lambda_values is None:
        lambda_values = np.array([1, 5, 10, 25, 50, 100, 250, 500, 1000])

    results = []
    for lam in lambda_values:
        opt = gradient_descent_irrigation(
            n_days=n_days, s0=s0, rainfall=rainfall, et=et,
            field_capacity=field_capacity, drainage_coeff=drainage_coeff,
            min_moisture=min_moisture, penalty_weight=float(lam),
            max_iter=max_iter,
        )
        results.append({
            "lambda": float(lam),
            "total_water_mm": opt.total_water_used,
            "constraint_violation": opt.constraint_violation,
            "converged": opt.converged,
            "iterations": opt.iterations,
        })

    return results
