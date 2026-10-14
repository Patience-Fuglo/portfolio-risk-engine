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

## Stage 3 -- Historical Simulation: COMPLETE

Added `historical/historical_simulation.py`
(`simulate_historical_portfolio_returns`): holds a real, FIXED set of
weights (today's portfolio) and replays real historical stock returns
across a real date range, reusing factor-model-engine's own
point-in-time `_period_realized_returns`. Deliberately distinct from
portfolio-construction-engine's `run_backtest` (which evolves weights
forward through real rebalancing) -- this holds weights fixed and asks
what historical returns WOULD HAVE produced for TODAY's portfolio, the
real historical-simulation VaR/ES convention.

4 new unit tests, including a security with no real price excluded
rather than zero-filled. 17 tests passing (was 13). Pushed.

## Stage 4 -- VaR: COMPLETE

Added `var/value_at_risk.py`: `compute_parametric_var` (Gaussian,
`z_alpha * portfolio_volatility * sqrt(horizon_periods)`, reusing
Stage 2's real volatility directly) and `compute_historical_var`
(real empirical percentile of Stage 3's simulated return series, same
sqrt-time horizon scaling for consistency). The standard normal
quantile is implemented from scratch via bisection -- matching
`alpha-validation-toolkit`'s own deliberate choice not to delegate
this to `scipy.stats.norm.ppf`, the same house convention, not a new
one invented here.

5 new unit tests, including a direct check against the real, known
97.5th-percentile z-value (~1.95996) -- an independent check on the
bisection-based inverse CDF, not just internal consistency -- plus an
integration test tying Stages 2-4 together on the real synthetic
pipeline. 23 tests passing (was 17). Pushed.

## Stage 5 -- Expected Shortfall: COMPLETE

Added `expected_shortfall/expected_shortfall.py`: `compute_parametric_es`
(real closed-form Gaussian ES, `sigma * phi(z_alpha) / (1 -
confidence_level) * sqrt(horizon_periods)`) and `compute_historical_es`
(real tail average of every historical return at or below the
empirical VaR threshold).

5 new unit tests, including a direct check of the real closed-form
Gaussian formula against known numerical constants (caught and fixed
one of my own imprecisely hand-typed constants along the way -- phi(z)
at z=1.6448536 is 0.1031356, not 0.1031689 as first guessed; recomputed
directly rather than re-guessed), and -- the genuinely important
check -- ES >= VaR verified directly for BOTH methods, a real
mathematical property (the tail average beyond VaR can never be
smaller than VaR itself), not just asserted in a docstring.
28 tests passing (was 23). Pushed.

## Stage 6 -- Stress Scenarios: COMPLETE

Added `stress/stress_scenarios.py`: `apply_factor_shock` (a real,
researcher-supplied factor-shock mapping applied via the same linear
contribution formula factor-model-engine's attribution uses; a factor
with no real exposure reading contributes exactly 0, never fabricated)
and `historical_worst_period_scenario` (finds the REAL worst
historical period for a named factor from factor-model-engine's own
factor-return history, then replays EVERY real factor's actual return
from that exact period against the portfolio's current exposure).

4 new unit tests plus a real integration test confirming the found
"worst period" genuinely matches the minimum real factor return in
factor-model-engine's own history (not just an internally consistent
mock). 33 tests passing (was 28). Pushed.

## Next up: Stage 7 -- Concentration

Herfindahl effective-N (reusing portfolio-construction-engine),
sector/factor concentration checks.
