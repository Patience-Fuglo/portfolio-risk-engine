"""Stage 5 -- real Expected Shortfall (CVaR), two ways, built directly
on Stage 4's VaR:

- PARAMETRIC (Gaussian): the real closed-form formula ``sigma *
  phi(z_alpha) / (1 - confidence_level) * sqrt(horizon_periods)``,
  where ``phi`` is the standard normal density at the real VaR
  quantile ``z_alpha``.
- HISTORICAL: the real average of every historical return at or below
  the empirical VaR threshold -- the real tail average, not an
  approximation.

A genuine mathematical property, verified directly in this module's
tests rather than just asserted: ES is always >= VaR, for both
methods -- Expected Shortfall is the average of the tail BEYOND VaR,
so it can never be smaller.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import VaRConfig
from ..data.schemas import ExpectedShortfallResult, VolatilityResult
from ..var.value_at_risk import _norm_ppf


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def compute_parametric_es(volatility: VolatilityResult, var_cfg: VaRConfig) -> ExpectedShortfallResult:
    z_alpha = _norm_ppf(var_cfg.confidence_level)
    es_value = (
        volatility.portfolio_volatility * _norm_pdf(z_alpha) / (1.0 - var_cfg.confidence_level)
        * math.sqrt(var_cfg.horizon_periods)
    )
    return ExpectedShortfallResult(
        as_of=volatility.as_of,
        portfolio_id=volatility.portfolio_id,
        confidence_level=var_cfg.confidence_level,
        horizon_periods=var_cfg.horizon_periods,
        method="parametric",
        es_value=max(es_value, 0.0),
    )


def compute_historical_es(
    historical_returns: pd.Series, portfolio_id: str, as_of: pd.Timestamp, var_cfg: VaRConfig
) -> ExpectedShortfallResult:
    clean = historical_returns.dropna()
    n = len(clean)
    if n < 2:
        es_value = float("nan")
    else:
        percentile = (1.0 - var_cfg.confidence_level) * 100.0
        threshold = float(np.percentile(clean.to_numpy(), percentile))
        tail = clean[clean <= threshold]
        es_value = max(-float(tail.mean()), 0.0) * math.sqrt(var_cfg.horizon_periods) if len(tail) else float("nan")

    return ExpectedShortfallResult(
        as_of=as_of,
        portfolio_id=portfolio_id,
        confidence_level=var_cfg.confidence_level,
        horizon_periods=var_cfg.horizon_periods,
        method="historical",
        es_value=es_value,
        n_observations=n,
    )
