"""Stage 6 -- real stress scenarios, two ways:

- A HYPOTHETICAL factor shock: an explicit, researcher-supplied
  ``{factor_name: shocked_return}`` mapping, applied to the
  portfolio's own real exposure vector via the same LINEAR
  contribution formula factor-model-engine's attribution uses
  (``exposure * factor_return``) -- documented, same as there, as a
  linear approximation, not an exact repricing.
- A HISTORICAL replay: finds the REAL worst historical period for a
  named factor (from factor-model-engine's own real
  ``FactorReturnRow`` history) and applies THAT period's real,
  observed factor returns (every factor, not just the named one) to
  the portfolio's CURRENT exposure vector -- "what would today's
  portfolio have lost if it held these exposures during the real
  worst historical period for this factor."
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from factor_model.data.schemas import ExposureSubject, FactorExposureRow, FactorReturnRow

from ..data.schemas import StressScenarioResult


def apply_factor_shock(
    portfolio_exposure_rows: Sequence[FactorExposureRow],
    shocked_factor_returns: dict[str, float],
    portfolio_id: str,
    scenario_name: str,
    as_of: pd.Timestamp,
) -> StressScenarioResult:
    """A factor the portfolio has no real exposure reading for
    contributes exactly 0 to the shocked P&L -- a real, documented
    "no exposure, no P&L impact" assumption, never a fabricated number.
    """
    exposure_by_factor = {
        row.factor_name: row.exposure_value
        for row in portfolio_exposure_rows
        if row.subject_id == portfolio_id and row.subject_type == ExposureSubject.PORTFOLIO
    }
    portfolio_pnl = sum(
        exposure_by_factor.get(factor_name, 0.0) * shock for factor_name, shock in shocked_factor_returns.items()
    )
    return StressScenarioResult(
        as_of=as_of,
        portfolio_id=portfolio_id,
        scenario_name=scenario_name,
        shocked_factor_returns=dict(shocked_factor_returns),
        portfolio_pnl=portfolio_pnl,
    )


def historical_worst_period_scenario(
    portfolio_exposure_rows: Sequence[FactorExposureRow],
    factor_return_rows: Sequence[FactorReturnRow],
    factor_name: str,
    portfolio_id: str,
    as_of: pd.Timestamp,
) -> StressScenarioResult:
    """Finds the real historical period where ``factor_name``'s own
    realized return was most negative, then applies EVERY real
    factor's actual return from that exact period (not just
    ``factor_name``'s) to the portfolio's current exposure vector --
    a real historical event replayed against today's portfolio.
    """
    own_factor_rows = [row for row in factor_return_rows if row.factor_name == factor_name]
    if not own_factor_rows:
        raise ValueError(f"no real factor-return rows for {factor_name!r}")
    worst = min(own_factor_rows, key=lambda row: row.factor_return)

    period_returns = {
        row.factor_name: row.factor_return
        for row in factor_return_rows
        if row.period_start == worst.period_start and row.period_end == worst.period_end
    }
    scenario_name = f"historical_worst_{factor_name}_{worst.period_start.date()}_{worst.period_end.date()}"
    return apply_factor_shock(portfolio_exposure_rows, period_returns, portfolio_id, scenario_name, as_of)
