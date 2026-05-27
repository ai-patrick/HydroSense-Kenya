# AI Usage Log — HydroSense-Kenya

This document records all uses of AI-assisted programming tools during the development of HydroSense-Kenya. Every AI-generated output was inspected, tested, and validated before acceptance. Modifications and validation methods are documented below.

---

## Entry 1: Test Case Generation for Root-Finding Methods

| Field | Detail |
|---|---|
| **Date** | 2026-05-16 |
| **Tool** | AI coding assistant |
| **Prompt** | "Generate pytest test cases for bisection, Newton-Raphson, and secant root-finding methods including edge cases and convergence verification." |
| **AI Output Summary** | Proposed 12 test functions covering known roots (x²−4=0), convergence order estimation, no-bracket failure mode, and SciPy cross-verification. |
| **Accepted?** | Partly |
| **Modifications** | 1. Adjusted tolerance from 1e-12 to 1e-10 for bisection (bracket halving reaches limit at ~42 iterations). 2. Added test for error history being monotonically decreasing. 3. Added cross-verification against `scipy.optimize.brentq`. |
| **Validation Method** | Ran all tests against hand-verified analytical roots. Confirmed convergence order by inspecting error ratio sequences. Cross-checked bisection root x=2.0 against manual computation. |

---

## Entry 2: Simpson's Rule Odd-Interval Handling

| Field | Detail |
|---|---|
| **Date** | 2026-05-16 |
| **Tool** | AI coding assistant |
| **Prompt** | "How should Simpson's 1/3 rule handle an odd number of subintervals?" |
| **AI Output Summary** | Suggested applying Simpson's 3/8 rule to the last 3 subintervals when n is odd. |
| **Accepted?** | Yes |
| **Modifications** | Verified the 3/8 rule coefficients (3h/8 × [f₀ + 3f₁ + 3f₂ + f₃]) against Burden & Faires, Numerical Analysis, 10th ed., Section 4.1. Implemented with explicit boundary handling. |
| **Validation Method** | Tested on cubic polynomial (x³ from 0 to 2, exact integral = 4.0) — confirmed exactness. Compared with `scipy.integrate.quad` for sin(x) integral. |

---

## Entry 3: Monte Carlo Rainfall Distribution Selection

| Field | Detail |
|---|---|
| **Date** | 2026-05-16 |
| **Tool** | AI coding assistant |
| **Prompt** | "What probability distribution is appropriate for modelling daily rainfall uncertainty in tropical East Africa?" |
| **AI Output Summary** | Recommended Gamma distribution due to non-negative support and right-skew typical of rainfall data. Suggested method-of-moments parameter estimation. |
| **Accepted?** | Yes |
| **Modifications** | Added separate modelling of rainfall occurrence (Bernoulli) and intensity (Gamma), rather than fitting a single distribution to all days including zeros. This two-component model is standard in stochastic weather generators (Richardson, 1981). |
| **Validation Method** | Fitted Gamma parameters to observed wet-day rainfall. Verified shape > 0 and scale > 0. Confirmed generated scenarios maintain observed wet-day fraction (≈0.7) and mean intensity within 15% of observed. |

---

## Entry 4: Armijo Backtracking Line Search

| Field | Detail |
|---|---|
| **Date** | 2026-05-16 |
| **Tool** | AI coding assistant |
| **Prompt** | "Implement Armijo backtracking line search for gradient descent in irrigation optimisation." |
| **AI Output Summary** | Provided implementation with sufficient decrease condition: f(x − α∇f) ≤ f(x) − c·α·‖∇f‖². |
| **Accepted?** | Partly |
| **Modifications** | Changed default contraction factor from 0.8 to 0.5 (more aggressive backtracking to avoid slow convergence). Added irrigation non-negativity enforcement after each step. |
| **Validation Method** | Verified convergence on synthetic test case with known optimal irrigation schedule. Confirmed objective function is monotonically non-increasing across iterations. |

---

## Entry 5: Colourblind-Safe Visualisation Palette

| Field | Detail |
|---|---|
| **Date** | 2026-05-16 |
| **Tool** | AI coding assistant |
| **Prompt** | "Suggest a colourblind-safe colour palette for scientific visualisations with 8 distinct colours." |
| **AI Output Summary** | Recommended the Wong (2011) palette from Nature Methods: 8 colours distinguishable under all common forms of colour vision deficiency. |
| **Accepted?** | Yes |
| **Modifications** | None — verified hex codes against the original Wong (2011) publication. |
| **Validation Method** | Visually inspected all generated plots. Cross-referenced with Coblis colour blindness simulator. |

---

## Summary

| Metric | Count |
|---|---|
| Total AI interactions logged | 5 |
| Fully accepted without modification | 2 |
| Partially accepted with modifications | 3 |
| Rejected | 0 |
| All outputs independently validated | ✓ |
