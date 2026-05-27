"""
Scientific Visualization Engine for HydroSense-Kenya

Publication-quality figure generation for water balance analysis,
uncertainty quantification, and decision support.

Design principles:
  - Consistent colour palette across all figures (colourblind-safe)
  - LaTeX-style labels with physical units on every axis
  - Interpretable legends and annotations
  - Separation of computation from rendering
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Global style configuration
# ═══════════════════════════════════════════════════════════════════════════

# Colourblind-safe palette (Wong, 2011 — Nature Methods)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#999999",
    "dark": "#1a1a2e",
    "accent": "#16213e",
}

ZONE_COLORS = {
    "Zone_A": "#0072B2",
    "Zone_B": "#E69F00",
    "Zone_C": "#009E73",
}


def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cccccc",
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "sans-serif",
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def _add_minor_ticks(ax: plt.Axes) -> None:
    """Add subtle minor ticks for publication quality."""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


# ═══════════════════════════════════════════════════════════════════════════
# Weather overview
# ═══════════════════════════════════════════════════════════════════════════

def plot_weather_overview(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-panel weather time series: rainfall, temperature, humidity, wind, solar.

    Reveals seasonal structure and highlights the data quality issues
    (missing values, outliers) that motivate the cleaning pipeline.
    """
    setup_publication_style()
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    dates = pd.to_datetime(df["date"])

    configs = [
        ("rainfall_mm", "Rainfall", "mm/day", COLORS["blue"], "bar"),
        ("temperature_c", "Temperature", "°C", COLORS["red"], "line"),
        ("humidity_pct", "Relative Humidity", "%", COLORS["cyan"], "line"),
        ("wind_speed_mps", "Wind Speed", "m/s", COLORS["green"], "line"),
        ("solar_index", "Solar Index", "fraction", COLORS["orange"], "line"),
    ]

    for ax, (col, title, unit, color, style) in zip(axes, configs):
        if style == "bar":
            ax.bar(dates, df[col], color=color, alpha=0.7, width=0.8,
                   edgecolor="white", linewidth=0.5)
        else:
            ax.plot(dates, df[col], color=color, marker="o", markersize=3,
                    linewidth=1.5)
            ax.fill_between(dates, df[col], alpha=0.1, color=color)

        ax.set_ylabel(f"{title} [{unit}]")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        _add_minor_ticks(ax)

        # Mark missing values
        if df[col].isna().any():
            missing_dates = dates[df[col].isna()]
            for md in missing_dates:
                ax.axvline(md, color=COLORS["red"], alpha=0.5,
                           linestyle=":", linewidth=1.5)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("Date (March 2026)")
    fig.suptitle("HydroSense-Kenya — Weather Station Overview",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Soil moisture by zone
# ═══════════════════════════════════════════════════════════════════════════

def plot_soil_moisture_zones(
    df: pd.DataFrame,
    params_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlaid soil moisture trajectories with crop stress thresholds."""
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    for zone, color in ZONE_COLORS.items():
        zone_data = df[df["zone_id"] == zone]
        dates = pd.to_datetime(zone_data["timestamp"])
        ax.plot(dates, zone_data["soil_moisture_pct"],
                color=color, marker="o", markersize=4, label=zone, linewidth=2)

    # Add threshold bands if parameters available
    if params_df is not None:
        for _, row in params_df.iterrows():
            zone = row["zone_id"]
            color = ZONE_COLORS.get(zone, COLORS["grey"])
            ax.axhline(row["min_moisture_pct"], color=color,
                       linestyle="--", alpha=0.4, linewidth=1.2)
            ax.axhline(row["field_capacity_pct"], color=color,
                       linestyle=":", alpha=0.4, linewidth=1.2)

        ax.fill_between(ax.get_xlim(), 0, params_df["min_moisture_pct"].min(),
                        alpha=0.05, color=COLORS["red"], label="Stress zone")

    ax.set_xlabel("Date")
    ax.set_ylabel("Soil Moisture [% vol.]")
    ax.set_title("Soil Moisture Dynamics by Farm Zone")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    _add_minor_ticks(ax)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Convergence comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence_comparison(
    results: dict[str, list[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Log-scale error vs iteration for root-finding method comparison.

    Reveals the convergence order: linear (bisection) vs quadratic
    (Newton-Raphson) vs superlinear (secant).
    """
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    method_styles = {
        "bisection": (COLORS["blue"], "o", "-"),
        "newton_raphson": (COLORS["red"], "s", "-"),
        "secant": (COLORS["green"], "^", "-"),
    }

    for method_name, errors in results.items():
        color, marker, ls = method_styles.get(
            method_name, (COLORS["grey"], "x", "--"))
        ax.semilogy(range(len(errors)), errors,
                    color=color, marker=marker, linestyle=ls,
                    label=method_name.replace("_", " ").title(),
                    markersize=5, linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("|f(x)| (log scale)")
    ax.set_title("Root-Finding Convergence Comparison")
    ax.legend()
    _add_minor_ticks(ax)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Integration accuracy
# ═══════════════════════════════════════════════════════════════════════════

def plot_integration_comparison(
    methods: list[str],
    values: list[float],
    errors: list[float],
    exact_value: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing integration methods with error annotations."""
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]][:len(methods)]

    # Values
    bars = ax1.bar(methods, values, color=colors, alpha=0.8,
                   edgecolor="white", linewidth=1.5)
    ax1.axhline(exact_value, color=COLORS["red"], linestyle="--",
                linewidth=2, label=f"Exact = {exact_value:.6f}")
    ax1.set_ylabel("Integral Value")
    ax1.set_title("Integration Results")
    ax1.legend()

    # Errors
    ax2.bar(methods, errors, color=colors, alpha=0.8,
            edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("Absolute Error")
    ax2.set_title("Integration Error")
    ax2.set_yscale("log")

    for ax in (ax1, ax2):
        _add_minor_ticks(ax)

    fig.suptitle("Numerical Integration: Trapezoidal vs Simpson",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Simulation ensemble (Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════

def plot_simulation_ensemble(
    trajectories: np.ndarray,
    time_days: np.ndarray,
    min_moisture: float,
    field_capacity: float,
    title: str = "Monte Carlo Soil Moisture Ensemble",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Spaghetti plot with percentile confidence bands.

    Shows individual scenario traces (faded) with 5th/25th/50th/75th/95th
    percentile envelopes overlaid. Threshold lines show crop-stress and
    over-irrigation danger zones.
    """
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    n_scenarios = trajectories.shape[0]
    n_show = min(50, n_scenarios)

    # Spaghetti traces
    for i in range(n_show):
        ax.plot(time_days, trajectories[i], color=COLORS["blue"],
                alpha=0.05, linewidth=0.5)

    # Percentile bands
    p5 = np.percentile(trajectories, 5, axis=0)
    p25 = np.percentile(trajectories, 25, axis=0)
    p50 = np.percentile(trajectories, 50, axis=0)
    p75 = np.percentile(trajectories, 75, axis=0)
    p95 = np.percentile(trajectories, 95, axis=0)

    ax.fill_between(time_days, p5, p95, alpha=0.15, color=COLORS["blue"],
                    label="5th–95th percentile")
    ax.fill_between(time_days, p25, p75, alpha=0.3, color=COLORS["blue"],
                    label="25th–75th percentile")
    ax.plot(time_days, p50, color=COLORS["blue"], linewidth=2.5,
            label="Median trajectory")

    # Thresholds
    ax.axhline(min_moisture, color=COLORS["red"], linestyle="--",
               linewidth=2, label=f"Stress threshold ({min_moisture}%)")
    ax.axhline(field_capacity, color=COLORS["orange"], linestyle=":",
               linewidth=2, label=f"Field capacity ({field_capacity}%)")

    ax.set_xlabel("Day")
    ax.set_ylabel("Soil Moisture [% vol.]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    _add_minor_ticks(ax)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Risk histogram
# ═══════════════════════════════════════════════════════════════════════════

def plot_risk_histogram(
    trajectories: np.ndarray,
    min_moisture: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of minimum soil moisture across scenarios."""
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    min_values = trajectories.min(axis=1)

    ax.hist(min_values, bins=40, color=COLORS["blue"], alpha=0.7,
            edgecolor="white", linewidth=0.8, density=True)
    ax.axvline(min_moisture, color=COLORS["red"], linestyle="--",
               linewidth=2.5, label=f"Stress threshold = {min_moisture}%")

    shortage_frac = np.mean(min_values < min_moisture)
    ax.axvline(np.mean(min_values), color=COLORS["orange"], linestyle="-",
               linewidth=2, label=f"Mean min = {np.mean(min_values):.1f}%")

    ax.fill_betweenx([0, ax.get_ylim()[1] * 0.8], 0, min_moisture,
                     alpha=0.1, color=COLORS["red"])

    ax.set_xlabel("Minimum Soil Moisture [% vol.]")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Water Shortage Risk — P(shortage) = {shortage_frac:.1%}")
    ax.legend()
    _add_minor_ticks(ax)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Optimization curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_optimization_convergence(
    cost_history: list[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot objective function convergence during optimization."""
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(cost_history)), cost_history,
            color=COLORS["blue"], marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value (water use + penalty)")
    ax.set_title("Irrigation Optimization Convergence")
    _add_minor_ticks(ax)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_pareto_frontier(
    pareto_data: list[dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Pareto frontier: water use vs constraint violation."""
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    water = [d["total_water_mm"] for d in pareto_data]
    violation = [d["constraint_violation"] for d in pareto_data]
    lambdas = [d["lambda"] for d in pareto_data]

    scatter = ax.scatter(water, violation, c=np.log10(lambdas),
                         cmap="viridis", s=100, edgecolors="white",
                         linewidth=1.5, zorder=5)
    ax.plot(water, violation, color=COLORS["grey"], linestyle="--",
            alpha=0.5, linewidth=1)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("log₁₀(λ)")

    for w, v, lam in zip(water, violation, lambdas):
        ax.annotate(f"λ={lam:.0f}", (w, v), textcoords="offset points",
                    xytext=(8, 5), fontsize=8, alpha=0.7)

    ax.set_xlabel("Total Irrigation Water [mm]")
    ax.set_ylabel("Constraint Violation (moisture deficit)")
    ax.set_title("Pareto Frontier: Water Conservation vs Crop Safety")
    _add_minor_ticks(ax)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_irrigation_schedule(
    irrigation: np.ndarray,
    moisture: np.ndarray,
    min_moisture: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Combined bar + line chart: irrigation schedule with resulting moisture."""
    setup_publication_style()
    fig, ax1 = plt.subplots(figsize=(14, 6))

    days = np.arange(len(irrigation))

    # Irrigation bars
    ax1.bar(days, irrigation, color=COLORS["cyan"], alpha=0.7,
            edgecolor="white", linewidth=0.8, label="Irrigation applied")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Irrigation [mm/day]", color=COLORS["cyan"])
    ax1.tick_params(axis="y", labelcolor=COLORS["cyan"])

    # Moisture line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(moisture)), moisture, color=COLORS["blue"],
             linewidth=2.5, marker="o", markersize=3, label="Soil moisture")
    ax2.axhline(min_moisture, color=COLORS["red"], linestyle="--",
                linewidth=2, label=f"Min threshold = {min_moisture}%")
    ax2.set_ylabel("Soil Moisture [% vol.]", color=COLORS["blue"])
    ax2.tick_params(axis="y", labelcolor=COLORS["blue"])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Optimized Irrigation Schedule with Soil Moisture Response")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Error propagation
# ═══════════════════════════════════════════════════════════════════════════

def plot_error_propagation(
    noise_levels: np.ndarray,
    et_std: np.ndarray,
    irrigation_std: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Show how measurement uncertainty propagates to recommendations."""
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(noise_levels, et_std, color=COLORS["orange"],
             marker="s", linewidth=2, label="ET uncertainty (σ)")
    ax1.set_xlabel("Temperature Noise σ [°C]")
    ax1.set_ylabel("ET Standard Deviation [mm/day]")
    ax1.set_title("Error Propagation: Temperature → ET")
    ax1.legend()
    _add_minor_ticks(ax1)

    ax2.plot(noise_levels, irrigation_std, color=COLORS["red"],
             marker="^", linewidth=2, label="Irrigation uncertainty (σ)")
    ax2.set_xlabel("Temperature Noise σ [°C]")
    ax2.set_ylabel("Irrigation Rec. Std Dev [mm/day]")
    ax2.set_title("Error Propagation: Temperature → Irrigation")
    ax2.legend()
    _add_minor_ticks(ax2)

    fig.suptitle("Measurement Uncertainty Propagation Analysis",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_euler_vs_rk4(
    time_days: np.ndarray,
    euler_moisture: np.ndarray,
    rk4_moisture: np.ndarray,
    min_moisture: float,
    field_capacity: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare Euler and RK4 soil moisture trajectories."""
    setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(time_days, euler_moisture, color=COLORS["orange"],
             linewidth=2, marker="o", markersize=3, label="Euler method")
    ax1.plot(time_days, rk4_moisture, color=COLORS["blue"],
             linewidth=2, marker="s", markersize=3, label="RK4 method")
    ax1.axhline(min_moisture, color=COLORS["red"], linestyle="--",
                alpha=0.6, label=f"Min threshold = {min_moisture}%")
    ax1.axhline(field_capacity, color=COLORS["green"], linestyle=":",
                alpha=0.6, label=f"Field capacity = {field_capacity}%")
    ax1.set_ylabel("Soil Moisture [% vol.]")
    ax1.set_title("Euler vs RK4 Soil Moisture Simulation")
    ax1.legend()
    _add_minor_ticks(ax1)

    # Error subplot
    error = np.abs(euler_moisture - rk4_moisture)
    ax2.fill_between(time_days, 0, error, color=COLORS["red"], alpha=0.3)
    ax2.plot(time_days, error, color=COLORS["red"], linewidth=1.5)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("|Euler − RK4| [%]")
    ax2.set_title("Absolute Difference (Euler vs RK4)", fontsize=11)
    _add_minor_ticks(ax2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig
