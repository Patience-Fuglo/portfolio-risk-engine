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

## Stage 2 -- Volatility & Component Risk: COMPLETE

Added `volatility/volatility.py` (`compute_volatility`): the real
factor/specific decomposition reuses `factor-model-engine`'s own
`compute_portfolio_risk` (Stage 6 of that repo); real per-security
marginal (`(Sigma w)_i / portfolio_volatility`) and component
(`w_i * marginal_i`, summing exactly to total volatility) contributions
are computed from the real stock covariance matrix directly, reusing
`portfolio-construction-engine`'s `risk_contributions` (Stage 4 of
that repo) for the underlying `w_i * (Sigma w)_i` math. One real
caught mixup during this build: `compute_portfolio_risk`/`RiskConfig`
live in `factor_model`, not `portfolio_construction` -- fixed before
the import error reached a committed file.

5 new unit tests (including a direct check that component
contributions sum exactly to portfolio volatility, and that the
factor+specific decomposition agrees with the stock-covariance-based
total for a fully-specified fixture) plus a genuine 3-repo integration
test on the real synthetic dataset (factor-model-engine's pipeline ->
portfolio-construction-engine's risk model -> this engine's
volatility decomposition). 13 tests passing (was 7). Pushed.

## Next up: Stage 3 -- Historical Simulation

Hold today's weights fixed, replay real historical stock returns
(reusing factor-model-engine's point-in-time realized-return helper)
-- the shared input for VaR/ES/stress in later stages.
