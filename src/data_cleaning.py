"""
Data Cleaning Engine for HydroSense-Kenya

Provides scientifically defensible data validation, outlier detection,
imputation, and quality assessment for agricultural sensor data.

Design rationale:
    Agricultural sensor networks in semi-arid Kenya produce data with
    characteristic failure modes: capacitive soil sensors drift under
    salinity changes, tipping-bucket rain gauges jam during heavy events,
    and thermistors saturate at extreme temperatures. Each cleaning
    decision must be traceable and justified.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "rainfall_mm": (0.0, 150.0),
    "temperature_c": (5.0, 42.0),
    "humidity_pct": (10.0, 100.0),
    "wind_speed_mps": (0.0, 25.0),
    "solar_index": (0.0, 1.0),
    "soil_moisture_pct": (0.0, 60.0),
    "tank_level_liters": (0.0, 6000.0),
    "pump_flow_lpm": (0.0, 50.0),
    "pump_power_watts": (0.0, 750.0),
}


@dataclass
class CleaningRecord:
    """Audit trail entry for a single cleaning action."""
    column: str
    row_index: int | str
    original_value: float | str | None
    new_value: float | str | None
    reason: str
    method: str


@dataclass
class CleaningReport:
    """Aggregated cleaning report with full provenance."""
    records: list[CleaningRecord] = field(default_factory=list)
    n_missing_detected: int = 0
    n_outliers_detected: int = 0
    n_sensor_faults: int = 0
    n_values_imputed: int = 0
    n_values_clipped: int = 0

    def summary(self) -> str:
        return (
            f"Cleaning Report\n"
            f"  Missing values detected : {self.n_missing_detected}\n"
            f"  Outliers detected       : {self.n_outliers_detected}\n"
            f"  Sensor faults flagged   : {self.n_sensor_faults}\n"
            f"  Values imputed          : {self.n_values_imputed}\n"
            f"  Values clipped          : {self.n_values_clipped}\n"
            f"  Total audit records     : {len(self.records)}"
        )


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Flag outliers using the interquartile range fence method.

    Chosen over z-score for rainfall data because rainfall distributions
    are strongly right-skewed (gamma-like), making Gaussian assumptions
    unreliable. The IQR method is distribution-agnostic.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr < np.finfo(float).eps:
        return pd.Series(False, index=series.index)
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    return (series < lower_fence) | (series > upper_fence)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag outliers using modified z-score (MAD-based).

    Uses median absolute deviation instead of standard deviation
    to resist masking by the outliers themselves.
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad < np.finfo(float).eps:
        mean, std = series.mean(), series.std()
        if std < np.finfo(float).eps:
            return pd.Series(False, index=series.index)
        z = np.abs((series - mean) / std)
    else:
        modified_z = 0.6745 * (series - median) / mad
        z = np.abs(modified_z)
    return z > threshold


def validate_physical_ranges(
    df: pd.DataFrame,
    bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> tuple[pd.DataFrame, list[CleaningRecord]]:
    """Clip values outside physically plausible ranges with audit trail."""
    if bounds is None:
        bounds = PHYSICAL_BOUNDS
    df_clean = df.copy()
    records: list[CleaningRecord] = []
    for col, (lo, hi) in bounds.items():
        if col not in df_clean.columns:
            continue
        series = df_clean[col]
        for idx in series[series < lo].index:
            records.append(CleaningRecord(col, idx, float(series[idx]), lo,
                           f"Below physical minimum ({lo})", "clip_to_bound"))
        for idx in series[series > hi].index:
            records.append(CleaningRecord(col, idx, float(series[idx]), hi,
                           f"Above physical maximum ({hi})", "clip_to_bound"))
        df_clean[col] = series.clip(lower=lo, upper=hi)
    return df_clean, records


def impute_missing_linear(
    series: pd.Series, max_gap: int = 2,
) -> tuple[pd.Series, list[CleaningRecord]]:
    """Fill missing values via linear interpolation for short gaps.

    Justified for slowly-varying meteorological fields where adjacent-day
    correlation is high. Gaps longer than max_gap remain NaN.
    """
    records: list[CleaningRecord] = []
    mask_before = series.isna()
    filled = series.interpolate(method="linear", limit=max_gap)
    newly_filled = mask_before & ~filled.isna()
    for idx in series[newly_filled].index:
        records.append(CleaningRecord(
            series.name or "unknown", idx, None, float(filled[idx]),
            "Short-gap linear interpolation", "interpolate_linear"))
    return filled, records


def impute_missing_median(series: pd.Series) -> tuple[pd.Series, list[CleaningRecord]]:
    """Fill remaining NaN values with column median as conservative fallback."""
    records: list[CleaningRecord] = []
    median_val = series.median()
    if np.isnan(median_val):
        warnings.warn(f"Column '{series.name}' is entirely NaN.")
        return series.copy(), records
    mask = series.isna()
    filled = series.fillna(median_val)
    for idx in series[mask].index:
        records.append(CleaningRecord(
            series.name or "unknown", idx, None, float(median_val),
            "Median imputation (fallback)", "fillna_median"))
    return filled, records


def flag_sensor_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[CleaningRecord]]:
    """Cross-validate soil sensor readings for internal consistency.

    Detects tank level spikes (>2x rolling median), zero pump flow with
    CHECK status, and abrupt soil moisture drops (>15 pct-points/day).
    """
    df_f = df.copy()
    df_f["anomaly_flag"] = False
    records: list[CleaningRecord] = []
    if "tank_level_liters" in df_f.columns:
        for zone in df_f.get("zone_id", pd.Series(dtype=str)).unique():
            zm = df_f["zone_id"] == zone
            zt = df_f.loc[zm, "tank_level_liters"]
            rm = zt.rolling(window=5, min_periods=1, center=True).median()
            for idx in zt[zt > 2.0 * rm].index:
                df_f.loc[idx, "anomaly_flag"] = True
                records.append(CleaningRecord("tank_level_liters", idx,
                    float(zt[idx]), None,
                    f"Tank spike: {zt[idx]:.0f} L > 2x rolling median",
                    "rolling_median_spike"))
    if "pump_flow_lpm" in df_f.columns and "sensor_status" in df_f.columns:
        chk = (df_f["pump_flow_lpm"] == 0.0) & (df_f["sensor_status"] == "CHECK")
        for idx in df_f[chk].index:
            df_f.loc[idx, "anomaly_flag"] = True
            records.append(CleaningRecord("pump_flow_lpm", idx, 0.0, None,
                "Zero pump flow with CHECK status", "status_cross_check"))
    if "soil_moisture_pct" in df_f.columns:
        for zone in df_f.get("zone_id", pd.Series(dtype=str)).unique():
            zm = df_f["zone_id"] == zone
            sm = df_f.loc[zm, "soil_moisture_pct"]
            dc = sm.diff()
            for idx in sm[dc < -15.0].index:
                df_f.loc[idx, "anomaly_flag"] = True
                records.append(CleaningRecord("soil_moisture_pct", idx,
                    float(sm[idx]), None,
                    f"Abrupt drop: {dc[idx]:.1f} pct-points/day",
                    "gradient_threshold"))
    return df_f, records


def compute_data_quality_score(
    df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
    numeric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Quantify dataset reliability as fraction of original vs imputed values."""
    if numeric_cols is None:
        numeric_cols = [c for c in df_original.select_dtypes(include=[np.number]).columns
                        if c in df_cleaned.columns]
    rows = []
    for col in numeric_cols:
        n = len(df_original)
        mb = int(df_original[col].isna().sum())
        ma = int(df_cleaned[col].isna().sum())
        ni = mb - ma
        rows.append({"column": col, "n_total": n, "n_missing_before": mb,
                      "completeness_before_pct": 100.0 * (1.0 - mb / n),
                      "n_missing_after": ma,
                      "completeness_after_pct": 100.0 * (1.0 - ma / n),
                      "n_imputed": ni,
                      "imputation_rate_pct": 100.0 * ni / n if n > 0 else 0.0})
    return pd.DataFrame(rows)


def clean_weather_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """End-to-end cleaning pipeline for weather_daily.csv.

    Order: parse dates -> clip physical ranges -> detect outliers -> impute.
    Clipping before imputation ensures interpolation endpoints are plausible.
    """
    report = CleaningReport()
    dw = df.copy()
    if "date" in dw.columns:
        dw["date"] = pd.to_datetime(dw["date"], errors="coerce")
        dw = dw.sort_values("date").reset_index(drop=True)
    nc = dw.select_dtypes(include=[np.number]).columns.tolist()
    report.n_missing_detected = int(dw[nc].isna().sum().sum())
    wb = {k: v for k, v in PHYSICAL_BOUNDS.items() if k in dw.columns}
    dw, cr = validate_physical_ranges(dw, wb)
    report.records.extend(cr)
    report.n_values_clipped = len(cr)
    for col in ["rainfall_mm", "temperature_c", "humidity_pct", "wind_speed_mps"]:
        if col in dw.columns:
            report.n_outliers_detected += int(detect_outliers_iqr(dw[col].dropna()).sum())
    for col in nc:
        dw[col], lr = impute_missing_linear(dw[col])
        report.records.extend(lr)
        dw[col], mr = impute_missing_median(dw[col])
        report.records.extend(mr)
    report.n_values_imputed = sum(1 for r in report.records
                                  if r.method in ("interpolate_linear", "fillna_median"))
    return dw, report


def clean_soil_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """End-to-end cleaning pipeline for soil_sensor_data.csv."""
    report = CleaningReport()
    dw = df.copy()
    if "timestamp" in dw.columns:
        dw["timestamp"] = pd.to_datetime(dw["timestamp"], errors="coerce")
        dw = dw.sort_values(["timestamp", "zone_id"]).reset_index(drop=True)
    nc = dw.select_dtypes(include=[np.number]).columns.tolist()
    report.n_missing_detected = int(dw[nc].isna().sum().sum())
    sb = {k: v for k, v in PHYSICAL_BOUNDS.items() if k in dw.columns}
    dw, cr = validate_physical_ranges(dw, sb)
    report.records.extend(cr)
    report.n_values_clipped = len(cr)
    dw, ar = flag_sensor_anomalies(dw)
    report.records.extend(ar)
    report.n_sensor_faults = len(ar)
    for col in nc:
        dw[col], lr = impute_missing_linear(dw[col])
        report.records.extend(lr)
        dw[col], mr = impute_missing_median(dw[col])
        report.records.extend(mr)
    report.n_values_imputed = sum(1 for r in report.records
                                  if r.method in ("interpolate_linear", "fillna_median"))
    return dw, report
