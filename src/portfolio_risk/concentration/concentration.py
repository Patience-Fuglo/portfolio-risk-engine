"""Stage 7 -- real portfolio concentration: the same Herfindahl-based
effective-N measure ``portfolio-construction-engine`` uses internally
for its own risk-limit checks (a one-line formula, re-derived locally
rather than reaching into that repo's private helper -- there is no
real logic to share beyond the formula itself), plus real top-N
weight concentration and real sector concentration.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from factor_model.data.schemas import PortfolioHolding

from ..config import ConcentrationConfig
from ..data.schemas import ConcentrationResult


def _effective_n_positions(weights: Sequence[float]) -> float:
    """The number of EQUALLY-weighted positions that would produce the
    same real concentration as the actual weight vector -- the same
    definition ``portfolio_construction.risk.portfolio_risk`` uses
    internally.
    """
    sum_sq = sum(w**2 for w in weights)
    return 1.0 / sum_sq if sum_sq > 0 else float("nan")


def compute_concentration(
    holdings: Sequence[PortfolioHolding],
    sector_by_security: dict[str, str],
    portfolio_id: str,
    as_of: pd.Timestamp,
    cfg: ConcentrationConfig,
    top_n_values: Sequence[int] = (1, 5, 10),
) -> ConcentrationResult:
    """Real concentration measures use GROSS weight (``abs(weight)``),
    consistent treatment whether the portfolio is long-only or
    long/short -- a large short position is real concentration risk
    too, not offset by a large long elsewhere.
    """
    own_holdings = [h for h in holdings if h.portfolio_id == portfolio_id]
    gross_weights = sorted((abs(h.weight) for h in own_holdings), reverse=True)
    effective_n = _effective_n_positions([h.weight for h in own_holdings])

    top_n_weight = {n: sum(gross_weights[:n]) for n in top_n_values if n <= len(gross_weights)}

    sector_totals: dict[str, float] = {}
    for h in own_holdings:
        sector = sector_by_security.get(h.security_id, "unknown")
        sector_totals[sector] = sector_totals.get(sector, 0.0) + abs(h.weight)

    warnings: list[str] = []
    if pd.notna(effective_n) and effective_n < cfg.min_effective_n_positions:
        warnings.append(f"effective n positions ({effective_n:.1f}) is below the configured minimum of {cfg.min_effective_n_positions}")
    for sector, total in sorted(sector_totals.items()):
        if total > cfg.max_sector_concentration:
            warnings.append(f"{sector}: gross weight {total:.1%} exceeds the configured {cfg.max_sector_concentration:.0%} sector concentration limit")

    return ConcentrationResult(
        as_of=as_of,
        portfolio_id=portfolio_id,
        effective_n_positions=effective_n,
        top_n_weight=top_n_weight,
        sector_concentration=sector_totals,
        warnings=tuple(warnings),
    )
