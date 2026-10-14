"""Stage 3 -- the standard "what would today's portfolio have earned
historically" methodology: hold a real, FIXED set of weights (today's
portfolio) and replay real historical stock returns across a real
date range. This is the shared real input VaR/ES (Stages 4-5) and
stress scenarios (Stage 6) build on.

Deliberately NOT the same thing as a backtest (portfolio-construction-
engine's own `backtest.run_backtest`): a backtest evolves weights
forward through real rebalancing decisions over time; this holds
TODAY's weights fixed and asks what historical stock-level returns
would have produced for THAT portfolio -- the real historical-
simulation VaR convention. Reuses factor-model-engine's own
point-in-time ``_period_realized_returns`` -- never re-derived.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from factor_model.data.schemas import MarketObservation
from factor_model.regression import _period_realized_returns


def simulate_historical_portfolio_returns(
    weights: dict[str, float],
    market_observations: Sequence[MarketObservation],
    period_dates: Sequence[pd.Timestamp],
) -> pd.Series:
    """A real historical return series, indexed by each period's END
    date, for a portfolio with FIXED ``weights`` held across every
    consecutive pair of (sorted) ``period_dates``. A security with no
    real price at either end of a given period is excluded from that
    period's weighted sum -- never silently treated as a 0% return,
    consistent with ``_period_realized_returns``'s own real
    point-in-time handling.
    """
    sorted_dates = sorted(period_dates)
    security_ids = list(weights.keys())
    returns: dict[pd.Timestamp, float] = {}

    for t_prev, t_curr in zip(sorted_dates[:-1], sorted_dates[1:]):
        stock_returns = _period_realized_returns(market_observations, security_ids, t_prev, t_curr)
        portfolio_return = sum(weights[sid] * stock_returns[sid] for sid in security_ids if sid in stock_returns.index)
        returns[t_curr] = portfolio_return

    return pd.Series(returns).sort_index()
