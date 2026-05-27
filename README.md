# HydroSense-Kenya

**A Scientific Computing System for Smart Irrigation, Water Balance Simulation, and Climate-Aware Decision Support**

*ICS 2207 — Scientific Computing Capstone Project, February–May 2026*

---

## Scientific Objective

Given daily weather and soil-sensor data from a Kenyan demonstration farm, HydroSense-Kenya answers:

> *How can we model water availability, estimate water deficit, simulate future soil moisture under rainfall uncertainty, and recommend an efficient irrigation plan that minimises water use without exposing crops to moisture stress?*

The system implements the discrete water balance equation:

```
S(t+1) = S(t) + R(t) + I(t) − ET(t) − D(t)
```

with a simplified evapotranspiration model:

```
ET = max(0, 0.12·T + 0.35·W + 2.4·Solar − 0.025·H)
```

## Repository Structure

```
HydroSense-Kenya/
├── data/
│   ├── raw/                      # Original sensor datasets
│   │   ├── weather_daily.csv
│   │   ├── soil_sensor_data.csv
│   │   └── crop_zone_parameters.csv
│   └── processed/                # Cleaned datasets
│       └── cleaned_irrigation_dataset.csv
├── notebooks/                    # Six-level analysis progression
│   ├── Level_1_Problem_Framing.ipynb
│   ├── Level_2_Vectorization_and_Error.ipynb
│   ├── Level_3_Numerical_Methods.ipynb
│   ├── Level_4_Data_Analysis_and_Visualization.ipynb
│   ├── Level_5_Simulation_and_Optimization.ipynb
│   └── Level_6_Final_Integration.ipynb
├── src/                          # Computation engine
│   ├── __init__.py
│   ├── data_cleaning.py          # Validation, imputation, outlier detection
│   ├── numerical_methods.py      # Root-finding, integration, linear systems
│   ├── simulation.py             # Water balance ODE, Monte Carlo engine
│   ├── optimization.py           # Irrigation scheduling optimiser
│   └── visualization.py          # Publication-quality figures
├── tests/                        # pytest-compatible test suite
│   ├── test_root_finding.py      # 14 tests
│   ├── test_integration.py       # 9 tests
│   ├── test_linear_systems.py    # 9 tests
│   └── test_simulation.py        # 16 tests
├── reports/
│   ├── final_scientific_report.pdf
│   └── presentation_slides.pdf
├── AI_USE_LOG.md
├── README.md
└── requirements.txt
```

## Installation

```bash
# Clone or extract the repository
cd HydroSense-Kenya

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Reproduction

```bash
# Run the full test suite
pytest tests/ -v

# Execute notebooks in order (Level 1 → Level 6)
jupyter notebook notebooks/
```

All computations use fixed random seeds (`np.random.default_rng(42)`) for deterministic output.

## Numerical Methods Implemented

| Category | Methods | Implemented From Scratch |
|---|---|---|
| Root finding | Bisection, Newton-Raphson, Secant | ✓ |
| Differentiation | Forward, backward, central finite differences | ✓ |
| Integration | Composite trapezoidal, composite Simpson's 1/3 | ✓ |
| Linear systems | Gaussian elimination with partial pivoting | ✓ |
| ODEs | Forward Euler, classical RK4 | ✓ |
| Optimisation | Penalised gradient descent with Armijo line search | ✓ |

SciPy is used **only** for cross-verification in the test suite.

## Dependencies

- Python ≥ 3.10
- NumPy ≥ 2.0
- Pandas ≥ 2.0
- Matplotlib ≥ 3.8
- SciPy ≥ 1.12 (verification only)
- pytest ≥ 8.0

## Datasets

Three synthetic datasets modelling a Kenyan demonstration farm (30 days, March 2026):

| File | Records | Purpose |
|---|---|---|
| `weather_daily.csv` | 30 days | Rainfall, temperature, humidity, wind, solar index |
| `soil_sensor_data.csv` | 90 records | Soil moisture, tank level, pump flow across 3 zones |
| `crop_zone_parameters.csv` | 3 zones | Crop thresholds and drainage coefficients |

Datasets include intentional anomalies (missing values, sensor faults, outliers) for data-cleaning exercises.

## License

Academic use — ICS 2207 Scientific Computing, 2026.
