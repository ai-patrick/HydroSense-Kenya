"""
Simulation Engine for HydroSense-Kenya

Implements the water balance ODE system with Euler and Runge-Kutta
integrators, Monte Carlo rainfall uncertainty modelling, and ensemble
risk quantification.

The core ODE:
    dS/dt = R(t) + I(t) - ET(t) - D(t)

where drainage D(t) = alpha * max(0, S(t) - S_fc) models gravitational
drainage as proportional to excess moisture above field capacity. This
is a standard bucket-model assumption in operational hydrology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Core physics
# ═══════════════════════════════════════════════════════════════════════════

def compute_et(
    temperature: np.ndarray,
    wind_speed: np.ndarray,
    solar_index: np.ndarray,
    humidity: np.ndarray,
) -> np.ndarray:
    """Compute daily evapotranspiration using simplified empirical model.

    ET = max(0, 0.12*T + 0.35*W + 2.4*Solar - 0.025*H)

    This is a linearised surrogate for Penman-Monteith, calibrated for
    teaching purposes. The coefficients encode physically reasonable
    sensitivities: ET increases with temperature and radiation, decreases
    with humidity (vapour-pressure deficit shrinks), and increases with
    wind (boundary-layer conductance rises).

    All inputs must be array-like with matching shapes. Uses np.maximum
    for element-wise clamping — numerically safe, no branching overhead.
    """
    et_raw = (
        0.12 * np.asarray(temperature, dtype=np.float64)
        + 0.35 * np.asarray(wind_speed, dtype=np.float64)
        + 2.4 * np.asarray(solar_index, dtype=np.float64)
        - 0.025 * np.asarray(humidity, dtype=np.float64)
    )
    return np.maximum(0.0, et_raw)


def compute_et_scalar(
    temperature: float, wind_speed: float,
    solar_index: float, humidity: float,
) -> float:
    """Scalar version of ET computation for ODE right-hand side."""
    return max(0.0, 0.12 * temperature + 0.35 * wind_speed
               + 2.4 * solar_index - 0.025 * humidity)


def compute_drainage(
    soil_moisture: float | np.ndarray,
    field_capacity: float,
    drainage_coeff: float,
) -> float | np.ndarray:
    """Gravitational drainage model: D = alpha * max(0, S - S_fc).

    Water drains only when moisture exceeds field capacity. The drainage
    coefficient alpha (day^-1) controls how quickly excess water leaves
    the root zone — higher for sandy soils, lower for clay.
    """
    if isinstance(soil_moisture, np.ndarray):
        return drainage_coeff * np.maximum(0.0, soil_moisture - field_capacity)
    return drainage_coeff * max(0.0, soil_moisture - field_capacity)


# ═══════════════════════════════════════════════════════════════════════════
# Water balance right-hand side
# ═══════════════════════════════════════════════════════════════════════════

def water_balance_rhs(
    s: float,
    rainfall: float,
    irrigation: float,
    et: float,
    field_capacity: float,
    drainage_coeff: float,
) -> float:
    """Evaluate dS/dt = R + I - ET - D(S) at current state.

    This is the function passed to the ODE integrators. Separating it
    from the integrator allows the same physics to be used with Euler,
    RK4, or any future method without code duplication.
    """
    drainage = compute_drainage(s, field_capacity, drainage_coeff)
    return rainfall + irrigation - et - drainage


# ═══════════════════════════════════════════════════════════════════════════
# ODE integrators
# ═══════════════════════════════════════════════════════════════════════════

def euler_step(
    s: float, dt: float,
    rainfall: float, irrigation: float, et: float,
    field_capacity: float, drainage_coeff: float,
) -> float:
    """Single forward Euler step: S_{n+1} = S_n + dt * f(S_n).

    First-order accurate: local truncation error O(dt^2), global O(dt).
    Stability constraint: dt < 1/drainage_coeff for the drainage term,
    otherwise the scheme amplifies perturbations exponentially.
    """
    dsdt = water_balance_rhs(s, rainfall, irrigation, et,
                             field_capacity, drainage_coeff)
    s_new = s + dt * dsdt
    return max(0.0, s_new)  # Soil moisture cannot be negative


def rk4_step(
    s: float, dt: float,
    rainfall: float, irrigation: float, et: float,
    field_capacity: float, drainage_coeff: float,
) -> float:
    """Single classical Runge-Kutta (RK4) step.

    Fourth-order accurate: local truncation error O(dt^5), global O(dt^4).
    For the same step size as Euler, RK4 is dramatically more accurate,
    at the cost of 4 function evaluations per step vs 1 for Euler.

    Since forcing terms (R, I, ET) are piecewise-constant over each day,
    the intermediate stage evaluations use the same forcing — only the
    state S varies through the stages.
    """
    k1 = water_balance_rhs(s, rainfall, irrigation, et,
                           field_capacity, drainage_coeff)
    k2 = water_balance_rhs(s + 0.5 * dt * k1, rainfall, irrigation, et,
                           field_capacity, drainage_coeff)
    k3 = water_balance_rhs(s + 0.5 * dt * k2, rainfall, irrigation, et,
                           field_capacity, drainage_coeff)
    k4 = water_balance_rhs(s + dt * k3, rainfall, irrigation, et,
                           field_capacity, drainage_coeff)
    s_new = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return max(0.0, s_new)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation driver
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """Container for water balance simulation output."""
    time_days: np.ndarray
    soil_moisture: np.ndarray
    et_series: np.ndarray
    drainage_series: np.ndarray
    rainfall_series: np.ndarray
    irrigation_series: np.ndarray
    method: str
    dt: float


def simulate_water_balance(
    n_days: int,
    s0: float,
    rainfall: np.ndarray,
    irrigation: np.ndarray,
    et: np.ndarray,
    field_capacity: float,
    drainage_coeff: float,
    method: str = "rk4",
    dt: float = 1.0,
) -> SimulationResult:
    """Simulate soil moisture trajectory over n_days.

    Parameters
    ----------
    n_days : int
        Number of days to simulate.
    s0 : float
        Initial soil moisture (% volumetric).
    rainfall, irrigation, et : np.ndarray
        Daily forcing arrays of length >= n_days.
    field_capacity : float
        Field capacity threshold (%).
    drainage_coeff : float
        Drainage coefficient (day^-1).
    method : str
        'euler' or 'rk4'.
    dt : float
        Time step in days (1.0 for daily resolution).

    Returns
    -------
    SimulationResult with full trajectory and diagnostics.
    """
    steps_per_day = max(1, int(1.0 / dt))
    total_steps = n_days * steps_per_day
    sub_dt = 1.0 / steps_per_day

    soil = np.zeros(total_steps + 1)
    et_out = np.zeros(total_steps)
    drain_out = np.zeros(total_steps)

    soil[0] = s0
    step_fn = rk4_step if method == "rk4" else euler_step

    for i in range(total_steps):
        day_idx = min(i // steps_per_day, n_days - 1)
        r = rainfall[day_idx] / steps_per_day
        irr = irrigation[day_idx] / steps_per_day
        e = et[day_idx] / steps_per_day

        et_out[i] = e
        drain_out[i] = compute_drainage(soil[i], field_capacity, drainage_coeff)

        soil[i + 1] = step_fn(soil[i], sub_dt, r, irr, e,
                              field_capacity, drainage_coeff)

    # Aggregate back to daily resolution
    time_days = np.arange(n_days + 1, dtype=np.float64)
    soil_daily = soil[::steps_per_day][:n_days + 1]
    et_daily = np.array([
        et_out[d * steps_per_day:(d + 1) * steps_per_day].sum()
        for d in range(n_days)
    ])
    drain_daily = np.array([
        drain_out[d * steps_per_day:(d + 1) * steps_per_day].sum()
        for d in range(n_days)
    ])

    return SimulationResult(
        time_days=time_days,
        soil_moisture=soil_daily,
        et_series=et_daily,
        drainage_series=drain_daily,
        rainfall_series=rainfall[:n_days].copy(),
        irrigation_series=irrigation[:n_days].copy(),
        method=method,
        dt=dt,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo engine
# ═══════════════════════════════════════════════════════════════════════════

def fit_rainfall_gamma(observed: np.ndarray) -> tuple[float, float]:
    """Fit Gamma distribution parameters to observed rainfall via method of moments.

    Rainfall is non-negative and right-skewed, making the Gamma distribution
    a physically motivated choice. Method of moments avoids iterative MLE
    and is transparent:
        shape k = mean^2 / var
        scale theta = var / mean

    Rainy days only (rainfall > 0) are used to estimate the wet-day
    distribution separately from the occurrence probability.
    """
    wet = observed[observed > 0]
    if len(wet) < 2:
        return 1.0, 1.0  # Degenerate fallback
    mean_r = np.mean(wet)
    var_r = np.var(wet, ddof=1)
    if var_r < np.finfo(float).eps:
        return 1.0, float(mean_r)
    shape = mean_r ** 2 / var_r
    scale = var_r / mean_r
    return float(shape), float(scale)


def monte_carlo_rainfall(
    observed: np.ndarray,
    n_scenarios: int = 1000,
    n_days: int = 30,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Generate stochastic rainfall scenarios using fitted Gamma distribution.

    Models two processes:
      1. Rainfall occurrence (Bernoulli) — probability = fraction of wet days
      2. Rainfall intensity | occurrence (Gamma) — fitted to wet-day amounts

    Returns array of shape (n_scenarios, n_days).
    """
    rng = np.random.default_rng(seed)

    # Fit from observed data
    wet_fraction = np.mean(observed > 0)
    shape, scale = fit_rainfall_gamma(observed)

    # Generate scenarios
    occurrence = rng.random((n_scenarios, n_days)) < wet_fraction
    intensity = rng.gamma(shape, scale, size=(n_scenarios, n_days))

    return occurrence * intensity


@dataclass
class RiskMetrics:
    """Decision-support metrics from Monte Carlo ensemble."""
    p_shortage: float          # P(min moisture < threshold)
    p_over_irrigation: float   # P(max moisture > field capacity)
    expected_demand: float     # E[total irrigation needed]
    worst_case_demand: float   # 95th percentile total irrigation
    mean_min_moisture: float   # E[min moisture across trajectory]
    shortage_days_mean: float  # E[number of days below threshold]


def monte_carlo_simulation(
    n_scenarios: int,
    n_days: int,
    s0: float,
    rainfall_scenarios: np.ndarray,
    irrigation: np.ndarray,
    temperature: np.ndarray,
    wind_speed: np.ndarray,
    solar_index: np.ndarray,
    humidity: np.ndarray,
    field_capacity: float,
    drainage_coeff: float,
    method: str = "rk4",
) -> np.ndarray:
    """Run ensemble of water balance simulations.

    Returns soil moisture trajectories: array of shape (n_scenarios, n_days+1).
    """
    et = compute_et(temperature, wind_speed, solar_index, humidity)
    trajectories = np.zeros((n_scenarios, n_days + 1))

    for s in range(n_scenarios):
        result = simulate_water_balance(
            n_days=n_days, s0=s0,
            rainfall=rainfall_scenarios[s],
            irrigation=irrigation,
            et=et,
            field_capacity=field_capacity,
            drainage_coeff=drainage_coeff,
            method=method,
        )
        trajectories[s] = result.soil_moisture

    return trajectories


def compute_risk_metrics(
    trajectories: np.ndarray,
    min_moisture: float,
    field_capacity: float,
) -> RiskMetrics:
    """Compute decision-support risk metrics from ensemble trajectories.

    Parameters
    ----------
    trajectories : np.ndarray
        Shape (n_scenarios, n_days+1) soil moisture ensemble.
    min_moisture : float
        Crop stress threshold (%).
    field_capacity : float
        Over-irrigation threshold (%).
    """
    n_scenarios = trajectories.shape[0]

    min_per_scenario = trajectories.min(axis=1)
    max_per_scenario = trajectories.max(axis=1)

    p_shortage = float(np.mean(min_per_scenario < min_moisture))
    p_over = float(np.mean(max_per_scenario > field_capacity))

    # Irrigation demand: sum of deficit below min_moisture
    deficit = np.maximum(0.0, min_moisture - trajectories[:, 1:])
    total_demand = deficit.sum(axis=1)

    shortage_days = np.sum(trajectories[:, 1:] < min_moisture, axis=1)

    return RiskMetrics(
        p_shortage=p_shortage,
        p_over_irrigation=p_over,
        expected_demand=float(np.mean(total_demand)),
        worst_case_demand=float(np.percentile(total_demand, 95)),
        mean_min_moisture=float(np.mean(min_per_scenario)),
        shortage_days_mean=float(np.mean(shortage_days)),
    )
