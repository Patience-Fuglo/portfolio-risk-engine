# Research assumptions

Every field below is a real modeling judgment call, not a safe
engineering default, and needs the researcher's explicit sign-off
before being relied on for any real analysis. Each lives in
`configs/default.yaml`, tagged `# RESEARCH ASSUMPTION` at the point of
definition, and in `src/portfolio_risk/config.py`'s own inline
comments -- this document consolidates both into one reference,
matching `factor-model-engine`'s own convention.

## VaR / Expected Shortfall (`var:` in `configs/default.yaml`)

**`confidence_level: 0.95`**
The standard real risk-management convention for both VaR and ES
(`var/value_at_risk.py`, `expected_shortfall/expected_shortfall.py`).
A specific real mandate's own risk-reporting requirement should
override this -- some desks report at 0.99, not 0.95.

**`horizon_periods: 1`**
The VaR/ES horizon, in real periods, matching whatever periodicity the
underlying returns/covariance actually use (monthly, for this engine's
synthetic demonstrations). Both methods scale by `sqrt(horizon_periods)`
-- a real, documented simplification: true multi-period historical VaR
technically wants overlapping-window aggregation, not a scaled
single-period percentile (see `var/value_at_risk.py`'s own docstring).

**`min_historical_observations: 24`**
Below this many real historical observations, historical VaR/ES is
still computed and returned (never withheld), but should be treated by
the caller as unreliable -- `VaRResult.n_observations`/
`ExpectedShortfallResult.n_observations` are always reported precisely
so this can be checked explicitly rather than silently trusted.

## Concentration (`concentration:`)

**`min_effective_n_positions: 5.0`**
Below this many real Herfindahl-based effective positions
(`concentration/concentration.py`), a portfolio is flagged
concentrated. A genuinely diversified book should clear this
comfortably; a concentrated conviction book may legitimately sit below
it -- the flag is a real signal for review, not an automatic rejection.

**`max_sector_concentration: 0.40`**
No single real sector's gross weight may exceed this share of total
portfolio weight before a concentration warning (and, via
`limits/limits.py`, a limit breach) fires. A round, moderate default --
a specific real mandate's own sector-limit policy should override this.

## Limits (`limits:`)

**`max_var_fraction_critical: 0.15`**
Real VaR, as a fraction of portfolio value, above which
`limits/limits.py` flags a breach `"critical"` rather than `"warning"`.

**`max_var_fraction_warning: 0.08`**
The `"warning"` threshold, strictly below the critical one
(`config.py`'s own `__post_init__` enforces this ordering). Both are
real risk-tolerance judgment calls, not derived from any statistical
property of the portfolio itself.

## Not a research assumption (real engineering defaults)

`run.output_dir`, `run.random_seed` are safe engineering defaults, not
modeling choices, and are not listed here.

## A real numerical-stability constant, not a research assumption

`volatility.py`'s own `_VARIANCE_EPSILON = 1e-12` is a directly
investigated floor for floating-point noise in the `w^T Sigma w`
quadratic form (a true-zero-variance portfolio can land on a tiny
nonzero float due to BLAS non-associativity), not a modeling judgment
call -- it is not configurable and is documented in the module itself,
not here.

## Reused real inputs, not re-estimated here

This engine makes no independent research assumptions about factor
construction, factor covariance, specific variance, or stock
covariance -- those real judgment calls live in
`factor-model-engine`'s and `portfolio-construction-engine`'s own
`docs/research_assumptions.md` files and are consumed here as given
(see `docs/data_dictionary.md` for exactly which real objects cross
that boundary).
