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

**All 10 stages are complete.** Schemas, config, dependency wiring to
both upstream engines; real volatility decomposition (factor vs.
specific, marginal/component contributions); historical simulation;
real parametric/historical VaR and Expected Shortfall; real stress
scenarios (hypothetical shocks and historical worst-period replay);
real concentration, risk budgets, and one consolidated limit-breach
check; one reproducible JSON/Markdown report; and a comprehensive
point-in-time/reproducibility hardening pass plus a real CLI-chain
subprocess test. See [`docs/STATUS.md`](docs/STATUS.md) for the full
stage-by-stage build history, the final architecture/formula summary,
test evidence, and an honest list of limitations -- including a real
numerical-stability finding (floating-point noise near a true-zero
quantity, computed via a plain matrix multiply with no solver
involved) caught and fixed at the source during Stage 10's hardening
pass.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m pytest tests/ -v
python scripts/run_risk_analysis.py --n-periods-months 36
```

## Repository layout

```
src/portfolio_risk/
  data/schemas.py                      # VolatilityResult, VaRResult, ExpectedShortfallResult,
                                         # StressScenarioResult, ConcentrationResult,
                                         # RiskBudget/RiskBudgetResult, LimitBreach
  volatility/volatility.py                # compute_volatility() -- factor/specific decomposition
                                           # + marginal/component contributions
  historical/historical_simulation.py       # simulate_historical_portfolio_returns()
  var/value_at_risk.py                        # compute_parametric_var()/compute_historical_var()
  expected_shortfall/expected_shortfall.py      # compute_parametric_es()/compute_historical_es()
  stress/stress_scenarios.py                      # apply_factor_shock()/historical_worst_period_scenario()
  concentration/concentration.py                    # compute_concentration()
  budgets/budgets.py                                  # check_risk_budgets()
  limits/limits.py                                      # check_limits()
  pipeline/pipeline.py                                    # run_full_risk_analysis()
  reporting/report.py                                       # report_to_dict()/render_markdown_report()

tests/
  unit/           # one file per module above
  integration/     # cross-repo pipeline tests, the comprehensive point-in-time check,
                    # real CLI-chain subprocess test

scripts/
  run_risk_analysis.py   # end-to-end: synthetic data -> full risk analysis -> report
```
