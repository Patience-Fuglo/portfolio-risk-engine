"""Stage 9 -- one consolidated, severity-tagged list of real limit
breaches across VaR, concentration, and risk budgets. Reuses every
real number from the earlier stages directly (``VaRResult.var_value``,
``ConcentrationResult.effective_n_positions``/``sector_concentration``,
``RiskBudgetResult.breached``) -- this stage applies NEW,
severity-tiered threshold comparisons to those already-computed real
figures; it never re-derives the underlying risk computations
themselves.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..config import ConcentrationConfig, LimitsConfig
from ..data.schemas import ConcentrationResult, LimitBreach, RiskBudgetResult, VaRResult


def check_limits(
    var_result: VaRResult,
    concentration_result: ConcentrationResult,
    budget_results: Sequence[RiskBudgetResult],
    limits_cfg: LimitsConfig,
    concentration_cfg: ConcentrationConfig,
) -> list[LimitBreach]:
    breaches: list[LimitBreach] = []

    if pd.notna(var_result.var_value):
        if var_result.var_value > limits_cfg.max_var_fraction_critical:
            breaches.append(
                LimitBreach("var", limits_cfg.max_var_fraction_critical, var_result.var_value, "critical",
                            f"{var_result.method} VaR ({var_result.var_value:.2%}) exceeds the critical threshold")
            )
        elif var_result.var_value > limits_cfg.max_var_fraction_warning:
            breaches.append(
                LimitBreach("var", limits_cfg.max_var_fraction_warning, var_result.var_value, "warning",
                            f"{var_result.method} VaR ({var_result.var_value:.2%}) exceeds the warning threshold")
            )

    if pd.notna(concentration_result.effective_n_positions) and concentration_result.effective_n_positions < concentration_cfg.min_effective_n_positions:
        breaches.append(
            LimitBreach("effective_n_positions", concentration_cfg.min_effective_n_positions,
                        concentration_result.effective_n_positions, "warning",
                        f"effective n positions ({concentration_result.effective_n_positions:.1f}) below the configured minimum")
        )

    for sector, total in sorted(concentration_result.sector_concentration.items()):
        if total > concentration_cfg.max_sector_concentration:
            breaches.append(
                LimitBreach(f"sector_concentration:{sector}", concentration_cfg.max_sector_concentration, total, "warning",
                            f"{sector} gross weight ({total:.1%}) exceeds the configured sector concentration limit")
            )

    for budget_result in budget_results:
        if budget_result.breached:
            breaches.append(
                LimitBreach(f"risk_budget:{budget_result.name}", budget_result.target_share, budget_result.actual_share, "warning",
                            f"{budget_result.name}: actual share ({budget_result.actual_share:.1%}) exceeds target ({budget_result.target_share:.1%})")
            )

    return breaches
