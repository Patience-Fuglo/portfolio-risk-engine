# portfolio-risk-engine

A point-in-time portfolio risk model: volatility decomposition
(factor vs. specific, marginal/component risk contributions),
Value-at-Risk and Expected Shortfall (parametric and historical),
stress scenarios, concentration, risk budgets, and limit breaches.

This engine never re-estimates a factor exposure, factor covariance,
specific variance, or stock covariance matrix -- it consumes
`factor-model-engine`'s and `portfolio-construction-engine`'s own real
outputs. It analyzes a GIVEN, fixed portfolio: "how risky is this
portfolio," never "which portfolio should we hold" (Engine 2's job) or
"what explains these stocks" (Engine 1's job).

**Stage 1 of 10 is complete**: schemas (`VolatilityResult`,
`VaRResult`, `ExpectedShortfallResult`, `StressScenarioResult`,
`ConcentrationResult`, `RiskBudget`/`RiskBudgetResult`, `LimitBreach`),
config, and dependency wiring to `factor-model-engine`,
`portfolio-construction-engine`, and `alpha-validation-toolkit`. See
[`docs/STATUS.md`](docs/STATUS.md) for stage-by-stage build status.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m pytest tests/ -v
```
