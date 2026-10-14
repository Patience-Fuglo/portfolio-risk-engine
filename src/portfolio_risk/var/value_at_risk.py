"""Stage 4 -- real Value-at-Risk, two ways:

- PARAMETRIC (Gaussian): ``z_alpha * portfolio_volatility *
  sqrt(horizon_periods)``, reusing Stage 2's real
  ``VolatilityResult.portfolio_volatility`` directly. Assumes zero
  real expected return over the horizon -- a standard, documented
  simplification for short-horizon risk management (the mean return
  is typically negligible next to volatility over a short horizon,
  though NOT universally true; a researcher with a real nonzero
  expected-return estimate should subtract it before interpreting this
  number as a worst-case P&L).
- HISTORICAL: the real empirical percentile of Stage 3's simulated
  historical return series, with the SAME sqrt-time horizon scaling
  applied for consistency with the parametric method when
  ``horizon_periods > 1`` -- itself a documented simplification (a
  real multi-period historical VaR technically wants overlapping-
  window aggregation, not a scaled single-period percentile).

The standard normal quantile (``z_alpha``) is implemented from
scratch via bisection, matching ``alpha-validation-toolkit``'s own
deliberate choice not to delegate this to ``scipy.stats.norm.ppf`` --
the same house convention, not a new one invented here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import VaRConfig
from ..data.schemas import VaRResult, VolatilityResult


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF via bisection -- monotonic, so
    exact to float precision without a rational approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compute_parametric_var(volatility: VolatilityResult, var_cfg: VaRConfig) -> VaRResult:
    z_alpha = _norm_ppf(var_cfg.confidence_level)
    var_value = z_alpha * volatility.portfolio_volatility * math.sqrt(var_cfg.horizon_periods)
    return VaRResult(
        as_of=volatility.as_of,
        portfolio_id=volatility.portfolio_id,
        confidence_level=var_cfg.confidence_level,
        horizon_periods=var_cfg.horizon_periods,
        method="parametric",
        var_value=max(var_value, 0.0),
    )


def compute_historical_var(
    historical_returns: pd.Series, portfolio_id: str, as_of: pd.Timestamp, var_cfg: VaRConfig
) -> VaRResult:
    """Real empirical VaR: ``-percentile(returns, 1 - confidence_level)``,
    scaled by ``sqrt(horizon_periods)`` for consistency with the
    parametric method. Below ``var_cfg.min_historical_observations``
    real observations, the result is still returned (never withheld),
    but the caller should treat it as unreliable -- see
    ``VaRResult.n_observations``, always reported, so this can be
    checked explicitly rather than silently trusted.
    """
    clean = historical_returns.dropna()
    n = len(clean)
    if n < 2:
        var_value = float("nan")
    else:
        percentile = (1.0 - var_cfg.confidence_level) * 100.0
        empirical_quantile = float(np.percentile(clean.to_numpy(), percentile))
        var_value = max(-empirical_quantile, 0.0) * math.sqrt(var_cfg.horizon_periods)

    return VaRResult(
        as_of=as_of,
        portfolio_id=portfolio_id,
        confidence_level=var_cfg.confidence_level,
        horizon_periods=var_cfg.horizon_periods,
        method="historical",
        var_value=var_value,
        n_observations=n,
    )
