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

**Stages 1–3 of 10 are complete**: schemas, config, dependency wiring,
a real portfolio volatility decomposition (factor vs. specific,
marginal/component risk contributions), and a real historical
simulation engine that holds today's weights fixed and replays
historical stock returns -- the shared input the coming VaR/ES/stress
stages build on. See [`docs/STATUS.md`](docs/STATUS.md) for
stage-by-stage build status.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m pytest tests/ -v
```
