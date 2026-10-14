"""Stage 8 -- real risk budgets: does any named factor or security
account for more than its configured share of real portfolio risk?
Reuses Stage 2's own real decomposition directly (``VolatilityResult.
factor_risk_contribution``/``component_contribution``) -- never a
separately re-derived risk figure.

A budget's ``name`` is looked up against BOTH real dictionaries: a
factor name resolves against ``factor_risk_contribution`` (its real
share of total portfolio VARIANCE); a security id resolves against
``component_contribution`` (its real share of total portfolio
VOLATILITY). Each is internally consistent in its own real unit; a
name matching neither is a real error, not silently skipped.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..data.schemas import RiskBudget, RiskBudgetResult, VolatilityResult


def check_risk_budgets(budgets: Sequence[RiskBudget], volatility: VolatilityResult) -> list[RiskBudgetResult]:
    results: list[RiskBudgetResult] = []
    for budget in budgets:
        if budget.name in volatility.factor_risk_contribution:
            contribution = volatility.factor_risk_contribution[budget.name]
            total = volatility.portfolio_variance
        elif budget.name in volatility.component_contribution:
            contribution = volatility.component_contribution[budget.name]
            total = volatility.portfolio_volatility
        else:
            raise ValueError(
                f"{budget.name!r} is neither a real factor nor a real security in this VolatilityResult"
            )

        actual_share = contribution / total if total > 0 else float("nan")
        breached = pd.notna(actual_share) and actual_share > budget.target_share
        results.append(
            RiskBudgetResult(name=budget.name, target_share=budget.target_share, actual_share=actual_share, breached=breached)
        )
    return results
