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

**Stages 1–7 of 10 are complete**: schemas, config, dependency wiring,
real volatility decomposition, historical simulation, real VaR and
Expected Shortfall, real stress scenarios, and real concentration
(Herfindahl effective-N, top-N and sector concentration, using gross
weight for long/short books). See [`docs/STATUS.md`](docs/STATUS.md)
for stage-by-stage build status.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m pytest tests/ -v
```
