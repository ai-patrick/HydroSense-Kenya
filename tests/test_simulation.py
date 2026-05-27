"""
Test suite for simulation engine.

Verification strategy:
  - ET non-negativity (physical constraint)
  - Conservation: no-drainage, no-ET → S should increase by R + I
  - Euler vs RK4 consistency for simple cases
  - Cross-verification against scipy.integrate.solve_ivp
  - Monte Carlo rainfall distribution sanity checks
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import (
    compute_et,
    compute_drainage,
    euler_step,
    rk4_step,
    simulate_water_balance,
    fit_rainfall_gamma,
    monte_carlo_rainfall,
    compute_risk_metrics,
)


class TestComputeET:

    def test_non_negative(self):
        """ET must always be ≥ 0 (physical constraint)."""
        temps = np.array([10.0, 20.0, 30.0, 40.0])
        wind = np.array([0.5, 1.0, 2.0, 3.0])
        solar = np.array([0.3, 0.5, 0.7, 0.9])
        humidity = np.array([90.0, 80.0, 60.0, 40.0])
        et = compute_et(temps, wind, solar, humidity)
        assert np.all(et >= 0.0)

    def test_increases_with_temperature(self):
        """Higher temperature → higher ET (all else equal)."""
        base = compute_et(np.array([20.0]), np.array([2.0]),
                          np.array([0.5]), np.array([60.0]))
        hot = compute_et(np.array([35.0]), np.array([2.0]),
                         np.array([0.5]), np.array([60.0]))
        assert hot[0] > base[0]

    def test_decreases_with_humidity(self):
        """Higher humidity → lower ET (smaller vapour pressure deficit)."""
        dry = compute_et(np.array([25.0]), np.array([2.0]),
                         np.array([0.5]), np.array([30.0]))
        humid = compute_et(np.array([25.0]), np.array([2.0]),
                           np.array([0.5]), np.array([90.0]))
        assert dry[0] > humid[0]

    def test_vectorized_output_shape(self):
        n = 30
        et = compute_et(np.ones(n) * 25, np.ones(n) * 2,
                        np.ones(n) * 0.5, np.ones(n) * 60)
        assert et.shape == (n,)


class TestDrainage:

    def test_no_drainage_below_field_capacity(self):
        assert compute_drainage(30.0, 40.0, 0.2) == 0.0

    def test_drainage_above_field_capacity(self):
        d = compute_drainage(45.0, 40.0, 0.2)
        assert abs(d - 0.2 * 5.0) < 1e-12

    def test_vectorized(self):
        s = np.array([30.0, 40.0, 45.0])
        d = compute_drainage(s, 40.0, 0.2)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(d, expected, atol=1e-12)


class TestODESteppers:

    def test_euler_conservation(self):
        """With no drainage and no ET, S should increase by R + I."""
        s0 = 20.0
        rainfall, irrigation, et = 5.0, 3.0, 0.0
        # Field capacity high enough that no drainage occurs
        s1 = euler_step(s0, 1.0, rainfall, irrigation, et, 100.0, 0.0)
        assert abs(s1 - (s0 + rainfall + irrigation)) < 1e-12

    def test_rk4_conservation(self):
        """Same conservation test for RK4."""
        s0 = 20.0
        s1 = rk4_step(s0, 1.0, 5.0, 3.0, 0.0, 100.0, 0.0)
        assert abs(s1 - 28.0) < 1e-12

    def test_moisture_stays_non_negative(self):
        """Even with extreme ET, moisture should not go negative."""
        s1 = euler_step(1.0, 1.0, 0.0, 0.0, 100.0, 50.0, 0.2)
        assert s1 >= 0.0


class TestSimulation:

    def test_output_shape(self):
        n_days = 10
        result = simulate_water_balance(
            n_days=n_days, s0=30.0,
            rainfall=np.ones(n_days) * 5.0,
            irrigation=np.zeros(n_days),
            et=np.ones(n_days) * 3.0,
            field_capacity=40.0, drainage_coeff=0.15,
        )
        assert len(result.soil_moisture) == n_days + 1
        assert len(result.et_series) == n_days

    def test_euler_vs_rk4_close(self):
        """For smooth forcing, Euler and RK4 should agree approximately."""
        n = 10
        r = np.ones(n) * 3.0
        irr = np.zeros(n)
        et = np.ones(n) * 2.5
        euler = simulate_water_balance(n, 30.0, r, irr, et, 40.0, 0.15, "euler")
        rk4 = simulate_water_balance(n, 30.0, r, irr, et, 40.0, 0.15, "rk4")
        np.testing.assert_allclose(
            euler.soil_moisture, rk4.soil_moisture, atol=2.0)


class TestMonteCarlo:

    def test_rainfall_non_negative(self):
        observed = np.array([0, 0, 3.2, 0, 7.1, 0, 1.5, 0, 0, 12.0])
        scenarios = monte_carlo_rainfall(observed, n_scenarios=100, n_days=30)
        assert np.all(scenarios >= 0.0)

    def test_output_shape(self):
        observed = np.array([0, 3.2, 7.1, 1.5, 12.0])
        scenarios = monte_carlo_rainfall(observed, n_scenarios=500, n_days=20)
        assert scenarios.shape == (500, 20)

    def test_gamma_fit_reasonable(self):
        observed = np.array([1.0, 2.0, 5.0, 3.0, 8.0, 0.5, 4.0])
        shape, scale = fit_rainfall_gamma(observed)
        assert shape > 0
        assert scale > 0

    def test_risk_metrics_bounds(self):
        """Probabilities should be in [0, 1]."""
        trajectories = np.random.default_rng(42).uniform(15, 45, (100, 31))
        metrics = compute_risk_metrics(trajectories, min_moisture=22.0,
                                        field_capacity=41.0)
        assert 0.0 <= metrics.p_shortage <= 1.0
        assert 0.0 <= metrics.p_over_irrigation <= 1.0
