# Build status

Tracks stage-by-stage completion, mirroring `factor-model-engine`'s
and `portfolio-construction-engine`'s own `docs/STATUS.md` convention.

## Final summary (all 10 stages complete)

**Architecture.** `data/` (schemas) -> `volatility/` (factor/specific
decomposition + marginal/component contributions, reusing both
upstream engines) -> `historical/` (hold-weights-fixed replay) ->
`var/` + `expected_shortfall/` (parametric + historical, from-scratch
normal quantile matching `alpha-validation-toolkit`'s own convention)
-> `stress/` (hypothetical shocks + real historical worst-period
replay) -> `concentration/` + `budgets/` + `limits/` (consolidated,
severity-tagged breach list) -> `pipeline/` + `reporting/` (one
orchestration function, one JSON/Markdown report). Depends on BOTH
`factor-model-engine` and `portfolio-construction-engine` as pinned
git dependencies (a 3-deep real dependency chain); never re-estimates
a factor exposure, factor covariance, specific variance, or stock
covariance matrix.

**Important formulas:** marginal contribution `(Sigma w)_i /
portfolio_volatility`; component contribution `w_i * marginal_i`
(sums exactly to total volatility); parametric VaR `z_alpha * sigma *
sqrt(horizon)`; parametric ES `sigma * phi(z_alpha) / (1 -
confidence) * sqrt(horizon)`; Herfindahl effective-N `1 /
sum(w_i^2)`.

**Test counts.** 59 tests passing: unit tests per module, integration
tests on the real synthetic dataset and on both upstream engines' real
pipeline output, and three Stage 10 hardening tests.

**Point-in-time integrity and reproducibility, including a real
numerical-stability finding caught along the way.** The comprehensive
Stage 10 test (extending the dataset 12 real months into the future;
re-running the identical analysis twice) initially failed
intermittently -- NOT from a look-ahead bug, and NOT from an iterative
solver (this engine never runs one, unlike portfolio-construction-
engine's SLSQP). The real cause: an equal-weight benchmark over the
FULL universe has a TRUE factor variance of exactly 0 (confirmed
elsewhere in this build), but the real `w^T Sigma w` quadratic form,
computed via a BLAS matrix multiply, occasionally lands on a tiny
nonzero float (~1e-30) instead of exact 0.0 -- the same root cause
(floating-point non-associativity in multi-threaded BLAS) already
documented for portfolio-construction-engine's SLSQP solver, now
confirmed to also affect a PLAIN matrix multiply with no solver
involved. Dividing by that noise (for marginal/component contributions
and risk-budget shares) would report numerically meaningless values
that happen to look like real numbers.

**Fixed at the source, not papered over in the test:** `volatility.
compute_volatility` now floors any variance-scale quantity below a
real noise epsilon (`1e-12`, far below any economically meaningful
variance) to exactly `0.0` before it is used in any further division
or reported in a result -- confirmed via 5 independent repeated runs
after the fix, all exactly reproducible.

**End-to-end demonstration.** `scripts/run_risk_analysis.py` runs the
complete Stage 2-9 pipeline and writes one JSON + Markdown risk
report; `tests/integration/test_cli_workflow.py` runs it as a real
subprocess.

**Limitations, identified rather than hidden:**
- Stress scenarios shock factor returns only (`specific_pnl_assumed_zero
  =True`) -- a real, documented simplification, not a full repricing.
- Historical VaR/ES use a single-period empirical percentile scaled by
  `sqrt(horizon)` for multi-period horizons, rather than true
  overlapping-window aggregation.
- This engine inherits every upstream limitation already documented in
  `factor-model-engine` (the Value-factor long-horizon finding, plain
  sample covariance with no shrinkage) and `portfolio-construction-
  engine` (the SLSQP non-determinism, now shown to have a sibling
  effect even without a solver).
- All data is synthetic; this engine has no real-vendor data adapter
  of its own.

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

## Stage 7 -- Concentration: COMPLETE

Added `concentration/concentration.py` (`compute_concentration`): the
same real Herfindahl effective-N formula `portfolio-construction-
engine` uses internally (re-derived locally -- a one-line formula, no
real logic to share beyond it), plus real top-N weight and sector
concentration, both using GROSS weight so a large short position
registers as real concentration risk, never offset by a long
elsewhere.

5 new unit tests, including a direct check that a dollar-neutral but
highly concentrated long/short book is correctly flagged (gross, not
net, weight). 38 tests passing (was 33). Pushed.

## Stage 8 -- Risk Budgets: COMPLETE

Added `budgets/budgets.py` (`check_risk_budgets`): reuses Stage 2's
own real decomposition directly
(`VolatilityResult.factor_risk_contribution`/`component_contribution`)
-- never a separately re-derived risk figure. A budget's name
resolves against EITHER real dictionary (factor name -> share of
total variance; security id -> share of total volatility); a name
matching neither raises, rather than silently skipping a
misconfigured budget.

6 new unit tests, including both factor- and position-level budgets,
breach and no-breach cases, and the real error on an unknown name.
44 tests passing (was 38). Pushed.

## Stage 9 -- Limit Breaches: COMPLETE

Added `limits/limits.py` (`check_limits`): one consolidated,
severity-tagged (`warning`/`critical`) list of real limit breaches
across VaR (two-tier threshold), concentration (effective-N and
sector), and risk budgets. Every real number reused directly from the
earlier stages -- this stage applies NEW threshold comparisons to
already-computed figures, never re-derives the underlying risk itself.

6 new unit tests plus an integration test tying Stages 2-9 together
on the real synthetic pipeline (volatility -> VaR -> concentration ->
budgets -> one consolidated breach list). 51 tests passing (was 44).
Pushed.

## Stage 10 -- Reporting + Hardening: COMPLETE

Added `pipeline/pipeline.py` (`run_full_risk_analysis`, orchestrates
Stages 2-9) and `reporting/report.py` (`report_to_dict`/
`render_markdown_report`, both derived from the same real
`RiskAnalysisResult`). Added `tests/integration/test_full_system.py`
(the comprehensive PIT/reproducibility check -- see the Final Summary
above for the real numerical-stability finding it caught and the fix
applied) and `tests/integration/test_cli_workflow.py` (real subprocess
CLI test). Added `scripts/run_risk_analysis.py`, this repo's first CLI
entry point.

59 tests passing (was 51). Full leak-check clean across the whole
repo. Pushed.

**All 10 stages of the portfolio-risk-engine build are now
complete.**
