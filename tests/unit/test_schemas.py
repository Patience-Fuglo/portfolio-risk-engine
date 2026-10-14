import pandas as pd
import pytest

from portfolio_risk.data import ExpectedShortfallResult, LimitBreach, RiskBudget, VaRResult


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_var_result_rejects_invalid_confidence_level():
    with pytest.raises(ValueError):
        VaRResult(as_of=_ts("2020-06-30"), portfolio_id="P", confidence_level=1.5, horizon_periods=1, method="parametric", var_value=0.05)


def test_var_result_rejects_unknown_method():
    with pytest.raises(ValueError):
        VaRResult(as_of=_ts("2020-06-30"), portfolio_id="P", confidence_level=0.95, horizon_periods=1, method="monte_carlo", var_value=0.05)


def test_var_result_rejects_non_positive_horizon():
    with pytest.raises(ValueError):
        VaRResult(as_of=_ts("2020-06-30"), portfolio_id="P", confidence_level=0.95, horizon_periods=0, method="parametric", var_value=0.05)


def test_expected_shortfall_result_validates_the_same_way_as_var():
    with pytest.raises(ValueError):
        ExpectedShortfallResult(as_of=_ts("2020-06-30"), portfolio_id="P", confidence_level=0.95, horizon_periods=1, method="bad", es_value=0.05)


def test_risk_budget_rejects_an_out_of_range_target_share():
    with pytest.raises(ValueError):
        RiskBudget(name="momentum", target_share=1.5)


def test_limit_breach_rejects_an_unknown_severity():
    with pytest.raises(ValueError):
        LimitBreach(limit_name="var", limit_value=0.1, actual_value=0.2, severity="urgent", detail="too high")


def test_limit_breach_accepts_a_real_severity():
    breach = LimitBreach(limit_name="var", limit_value=0.1, actual_value=0.2, severity="critical", detail="too high")
    assert breach.severity == "critical"
