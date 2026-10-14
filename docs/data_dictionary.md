# Data dictionary

Every schema field this engine actually defines, plus the real formula
each build stage computes -- matching `factor-model-engine`'s and
`portfolio-construction-engine`'s own `docs/data_dictionary.md`
convention. Fields reused directly from those two repos (e.g.
`PortfolioHolding`, `FactorExposureRow`, `FactorReturnRow`) are not
re-documented here; see their own dictionaries. This engine never
re-estimates a factor exposure, factor covariance, specific variance,
or stock covariance matrix -- it consumes those as real, already-given
upstream inputs and answers only "how risky is this portfolio."

## `VolatilityResult` (`data/schemas.py`)

- `as_of`, `portfolio_id` -- identifies the real snapshot this
  decomposition belongs to.
- `portfolio_variance`, `portfolio_volatility` -- the real `w^T Sigma w`
  and its square root, floored at exactly `0.0` below
  `_VARIANCE_EPSILON` (see `research_assumptions.md`).
- `factor_variance`, `specific_variance` -- the real factor/specific
  split, reused directly from `factor-model-engine`'s own
  `compute_portfolio_risk` (never re-derived here).
- `factor_risk_contribution` -- `dict[factor_name, float]`, each
  factor's real share of `portfolio_variance`, also reused directly
  from `compute_portfolio_risk`.
- `marginal_contribution` -- `dict[security_id, float]`, the real
  `(Sigma @ w)_i / portfolio_volatility`, computed here from the actual
  stock-level covariance (not a factor-only approximation).
- `component_contribution` -- `dict[security_id, float]`, the real
  `w_i * marginal_contribution_i`, which sums exactly to
  `portfolio_volatility` -- a genuine decomposition, reusing
  `portfolio_construction.optimization.risk_contributions` directly.
- `warnings` -- includes `factor-model-engine`'s own real warnings
  plus this engine's own (e.g. zero/negligible volatility).

## `VaRResult` / `ExpectedShortfallResult` (`data/schemas.py`)

Both share the same real shape: `as_of`, `portfolio_id`,
`confidence_level`, `horizon_periods`, `method` (`"parametric"` or
`"historical"`), the value field (`var_value`/`es_value`, always
reported as a positive loss magnitude), and `n_observations` (real
sample size, historical method only; `None` for parametric).
`__post_init__` on both validates `0 < confidence_level < 1`,
`horizon_periods >= 1`, and `method` is one of the two real supported
values.

## `StressScenarioResult` (`data/schemas.py`)

- `shocked_factor_returns` -- the real shock actually applied
  (`dict[factor_name, float]`): either a researcher-supplied
  hypothetical mapping, or a real historical period's actual observed
  factor returns for a historical replay.
- `portfolio_pnl` -- the real resulting return, computed from the
  portfolio's own real exposure vector via the linear contribution
  formula `exposure * factor_return`, summed across every shocked
  factor.
- `specific_pnl_assumed_zero: bool = True` -- documented: stress
  scenarios shock FACTOR returns only; idiosyncratic (specific) P&L
  under the shock is not modeled.

## `ConcentrationResult` (`data/schemas.py`)

- `effective_n_positions` -- the real Herfindahl-based effective
  position count, `1 / sum(w_i^2)` over GROSS weights (`abs(weight)`),
  consistent treatment for long-only or long/short books.
- `top_n_weight` -- `dict[n, float]`, e.g. `{5: 0.42}` meaning the top
  5 gross positions hold 42% of total gross weight.
- `sector_concentration` -- `dict[sector, float]`, real gross weight
  per sector.
- `warnings` -- fires when `effective_n_positions` is below
  `ConcentrationConfig.min_effective_n_positions`, or any sector
  exceeds `max_sector_concentration`.

## `RiskBudget` / `RiskBudgetResult` (`data/schemas.py`)

`RiskBudget(name, target_share)` -- a real, researcher-set target:
this factor or security should account for at most `target_share` of
total real portfolio risk. `__post_init__` requires `0 <
target_share <= 1`.

`RiskBudgetResult(name, target_share, actual_share, breached)` -- the
real outcome of `budgets.check_risk_budgets` (Stage 8 below).

## `LimitBreach` (`data/schemas.py`)

`limit_name`, `limit_value`, `actual_value`, `severity` (`"warning"` or
`"critical"`, validated in `__post_init__`), `detail` -- one real,
uniform record for every breach `limits.check_limits` (Stage 9 below)
produces, across VaR, concentration, and risk budgets alike.

## Stage 2 -- volatility (`volatility/volatility.py`)

`compute_volatility(...)`:

1. Calls `factor_model.risk.compute_portfolio_risk` directly for the
   real `factor_variance`/`specific_variance` split -- never
   re-derived.
2. Computes real per-security marginal/component contributions
   directly from the actual stock-level `stock_covariance` (not a
   factor-only approximation), via
   `portfolio_construction.optimization.risk_contributions`:
   `marginal_i = (Sigma @ w)_i / volatility`,
   `component_i = w_i * marginal_raw_i / volatility`.
3. Applies the real `_VARIANCE_EPSILON = 1e-12` floor to every
   BLAS-computed quantity (`portfolio_variance`, `factor_variance`,
   `specific_variance`, each `factor_risk_contribution` value) --
   see `research_assumptions.md`.

## Stage 3 -- historical simulation (`historical/historical_simulation.py`)

`simulate_historical_portfolio_returns(weights, market_observations,
period_dates)` -- holds `weights` (today's portfolio) **fixed** and
replays real historical stock returns across each consecutive pair of
sorted `period_dates`, via `factor_model.regression._period_realized_returns`
(reused directly). A security with no real price at either end of a
given period is excluded from that period's weighted sum, never
treated as a silent 0% return. Deliberately distinct from
`portfolio_construction.backtest.run_backtest`, which evolves weights
forward through real rebalancing decisions -- this holds one snapshot
fixed and asks what history would have produced for it.

## Stage 4 -- Value-at-Risk (`var/value_at_risk.py`)

- **Parametric**: `var_value = z_alpha * portfolio_volatility *
  sqrt(horizon_periods)`, `z_alpha` from `_norm_ppf` (the standard
  normal inverse CDF, implemented from scratch via bisection on
  `_norm_cdf` -- `math.erf`-based, not `scipy.stats.norm.ppf`, matching
  `alpha-validation-toolkit`'s own house convention of implementing
  statistics from scratch). Assumes zero real expected return over the
  horizon -- documented as a standard short-horizon simplification.
- **Historical**: `var_value = max(-percentile(returns,
  1-confidence_level), 0) * sqrt(horizon_periods)`, the real empirical
  quantile of Stage 3's simulated return series. Still computed (never
  withheld) below `min_historical_observations`; `n_observations` is
  always reported so the caller can check reliability explicitly.

## Stage 5 -- Expected Shortfall (`expected_shortfall/expected_shortfall.py`)

- **Parametric**: the real closed-form `es_value = volatility *
  phi(z_alpha) / (1 - confidence_level) * sqrt(horizon_periods)`,
  `phi` the standard normal density at the real VaR quantile.
- **Historical**: the real average of every historical return at or
  below the empirical VaR threshold -- the real tail average.
- A genuine mathematical property verified in this module's own tests:
  ES >= VaR for both methods, since ES averages the tail beyond VaR.

## Stage 6 -- stress scenarios (`stress/stress_scenarios.py`)

- `apply_factor_shock(...)` -- `portfolio_pnl = sum(exposure.get(factor,
  0.0) * shock for factor, shock in shocked_factor_returns.items())`. A
  factor the portfolio has no real exposure reading for contributes
  exactly `0.0` -- documented, never fabricated.
- `historical_worst_period_scenario(...)` -- finds the real historical
  period where the named factor's own realized return was most
  negative, then applies every real factor's actual return from that
  exact period (not just the named one) to the portfolio's current
  exposure vector via `apply_factor_shock`.

## Stage 7 -- concentration (`concentration/concentration.py`)

`compute_concentration(...)`: `_effective_n_positions(weights) = 1 /
sum(w_i^2)` over gross weights (the same real definition
`portfolio_construction.risk.portfolio_risk` uses internally,
re-derived locally as a one-line formula rather than importing a
private helper). `top_n_weight[n] = sum(top n gross weights)`.
`sector_concentration[sector] = sum(gross weight in that sector)`.
Warnings fire per `ConcentrationConfig`'s two thresholds.

## Stage 8 -- risk budgets (`budgets/budgets.py`)

`check_risk_budgets(budgets, volatility)` -- each `budget.name` is
looked up against **both** real dictionaries on `VolatilityResult`: a
factor name resolves against `factor_risk_contribution` (its real
share of `portfolio_variance`); a security id resolves against
`component_contribution` (its real share of `portfolio_volatility`).
A name matching neither raises -- a real error, not a silent skip.
`actual_share = contribution / total`; `breached = actual_share >
target_share`.

## Stage 9 -- limits (`limits/limits.py`)

`check_limits(...)` applies severity-tiered threshold comparisons to
already-computed real figures from Stages 4, 7, and 8 -- it never
re-derives any underlying risk computation. Three real breach sources:
VaR (`"critical"` above `max_var_fraction_critical`, `"warning"` above
`max_var_fraction_warning`), concentration (effective-N and
per-sector, both `"warning"`), and any breached risk budget
(`"warning"`).

## Stage 10 -- pipeline & reporting (`pipeline/pipeline.py`,
## `reporting/report.py`)

`run_full_risk_analysis(...)` runs Stages 2-9 once, in dependency
order, for one real portfolio, requiring `as_of` to be the last
(sorted) element of `period_dates` -- the real estimation window for
factor returns/covariance/specific risk ends exactly where the
analysis is dated, never later. Depends on both `factor-model-engine`
(universe, exposures, factor returns/covariance, specific risk) and
`portfolio-construction-engine` (`build_exposure_matrix`/
`build_stock_covariance_matrix`, the real stock-level covariance) for
everything upstream of this engine's own nine stages.

`report_to_dict(result)` / `render_markdown_report(result)` both
derive from the same real `RiskAnalysisResult` -- never recomputed
separately, so they cannot silently disagree.
