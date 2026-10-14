# Build status

Tracks stage-by-stage completion, mirroring `factor-model-engine`'s
and `portfolio-construction-engine`'s own `docs/STATUS.md` convention.

## Stage 1 -- Foundation: COMPLETE

Schemas (`data/schemas.py`): `VolatilityResult`, `VaRResult`,
`ExpectedShortfallResult`, `StressScenarioResult`,
`ConcentrationResult`, `RiskBudget`/`RiskBudgetResult`, `LimitBreach`.
`EngineConfig`/`load_config` (`config.py`). Dependency wiring to
`factor-model-engine`, `portfolio-construction-engine`, and
`alpha-validation-toolkit` -- confirmed resolving cleanly through a
fresh venv install (a 3-deep real dependency chain). 7 tests passing.
Pushed.

## Next up: Stage 2 -- Volatility & Component Risk

Marginal/component risk contributions (reusing `portfolio_construction.
optimization.risk_contributions`), factor-vs-specific decomposition
(reusing `portfolio_construction.risk.compute_portfolio_risk`).
