import pandas as pd
import pytest

from portfolio_risk.config import ConcentrationConfig, LimitsConfig
from portfolio_risk.data import ConcentrationResult, RiskBudgetResult, VaRResult
from portfolio_risk.limits import check_limits


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _var(value: float) -> VaRResult:
    return VaRResult(as_of=_ts("2020-06-30"), portfolio_id="P", confidence_level=0.95, horizon_periods=1, method="parametric", var_value=value)


def _concentration(effective_n: float = 10.0, sector_concentration=None) -> ConcentrationResult:
    return ConcentrationResult(as_of=_ts("2020-06-30"), portfolio_id="P", effective_n_positions=effective_n, sector_concentration=sector_concentration or {})


def test_var_breach_escalates_to_critical_above_the_critical_threshold():
    cfg = LimitsConfig(max_var_fraction_warning=0.05, max_var_fraction_critical=0.10)
    breaches = check_limits(_var(0.15), _concentration(), [], cfg, ConcentrationConfig())
    var_breach = next(b for b in breaches if b.limit_name == "var")
    assert var_breach.severity == "critical"


def test_var_breach_is_warning_between_thresholds():
    cfg = LimitsConfig(max_var_fraction_warning=0.05, max_var_fraction_critical=0.10)
    breaches = check_limits(_var(0.07), _concentration(), [], cfg, ConcentrationConfig())
    var_breach = next(b for b in breaches if b.limit_name == "var")
    assert var_breach.severity == "warning"


def test_no_var_breach_below_the_warning_threshold():
    cfg = LimitsConfig(max_var_fraction_warning=0.05, max_var_fraction_critical=0.10)
    breaches = check_limits(_var(0.02), _concentration(), [], cfg, ConcentrationConfig())
    assert not any(b.limit_name == "var" for b in breaches)


def test_concentration_breach_is_promoted_to_a_real_limit_breach():
    concentration = _concentration(effective_n=2.0)
    breaches = check_limits(_var(0.0), concentration, [], LimitsConfig(), ConcentrationConfig(min_effective_n_positions=5.0))
    assert any(b.limit_name == "effective_n_positions" and b.actual_value == pytest.approx(2.0) for b in breaches)


def test_sector_concentration_breach_is_promoted_with_real_numeric_values():
    concentration = _concentration(sector_concentration={"tech": 0.8})
    breaches = check_limits(_var(0.0), concentration, [], LimitsConfig(), ConcentrationConfig(max_sector_concentration=0.4))
    sector_breach = next(b for b in breaches if "tech" in b.limit_name)
    assert sector_breach.actual_value == pytest.approx(0.8)
    assert sector_breach.limit_value == pytest.approx(0.4)


def test_budget_breach_is_promoted_only_when_breached():
    budgets = [
        RiskBudgetResult(name="size", target_share=0.3, actual_share=0.5, breached=True),
        RiskBudgetResult(name="momentum", target_share=0.5, actual_share=0.2, breached=False),
    ]
    breaches = check_limits(_var(0.0), _concentration(), budgets, LimitsConfig(), ConcentrationConfig())
    budget_breaches = [b for b in breaches if b.limit_name.startswith("risk_budget:")]
    assert len(budget_breaches) == 1
    assert budget_breaches[0].limit_name == "risk_budget:size"
