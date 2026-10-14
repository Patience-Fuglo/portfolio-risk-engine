import pandas as pd
import pytest

from portfolio_risk.budgets import check_risk_budgets
from portfolio_risk.data import RiskBudget, VolatilityResult


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _volatility() -> VolatilityResult:
    return VolatilityResult(
        as_of=_ts("2020-06-30"), portfolio_id="P", portfolio_variance=0.01, portfolio_volatility=0.1,
        factor_variance=0.006, specific_variance=0.004,
        factor_risk_contribution={"size": 0.004, "momentum": 0.002},
        component_contribution={"A": 0.07, "B": 0.03},
    )


def test_factor_budget_computes_the_real_share_of_portfolio_variance():
    budgets = [RiskBudget(name="size", target_share=0.5)]
    results = check_risk_budgets(budgets, _volatility())
    assert results[0].actual_share == pytest.approx(0.004 / 0.01)
    assert not results[0].breached


def test_factor_budget_flags_a_real_breach():
    budgets = [RiskBudget(name="size", target_share=0.3)]  # actual is 0.4
    results = check_risk_budgets(budgets, _volatility())
    assert results[0].breached


def test_position_budget_computes_the_real_share_of_portfolio_volatility():
    budgets = [RiskBudget(name="A", target_share=0.8)]
    results = check_risk_budgets(budgets, _volatility())
    assert results[0].actual_share == pytest.approx(0.07 / 0.1)
    assert not results[0].breached


def test_position_budget_flags_a_real_breach():
    budgets = [RiskBudget(name="A", target_share=0.5)]  # actual is 0.7
    results = check_risk_budgets(budgets, _volatility())
    assert results[0].breached


def test_unknown_budget_name_raises_rather_than_silently_skipping():
    budgets = [RiskBudget(name="nonexistent", target_share=0.5)]
    with pytest.raises(ValueError):
        check_risk_budgets(budgets, _volatility())


def test_multiple_budgets_checked_independently():
    budgets = [RiskBudget(name="size", target_share=0.3), RiskBudget(name="momentum", target_share=0.5)]
    results = check_risk_budgets(budgets, _volatility())
    assert results[0].breached   # size: 0.4 > 0.3
    assert not results[1].breached  # momentum: 0.2 < 0.5
