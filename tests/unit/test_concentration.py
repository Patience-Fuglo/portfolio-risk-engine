import pandas as pd
import pytest
from factor_model.data.schemas import PortfolioHolding

from portfolio_risk.concentration import compute_concentration
from portfolio_risk.config import ConcentrationConfig


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _holding(security_id: str, weight: float) -> PortfolioHolding:
    return PortfolioHolding(portfolio_id="P", security_id=security_id, as_of=_ts("2020-06-30"), weight=weight, known_as_of=_ts("2020-06-30"))


def test_effective_n_positions_matches_the_real_herfindahl_formula():
    holdings = [_holding("A", 0.5), _holding("B", 0.3), _holding("C", 0.2)]
    result = compute_concentration(holdings, {}, "P", _ts("2020-06-30"), ConcentrationConfig())
    expected = 1.0 / (0.5**2 + 0.3**2 + 0.2**2)
    assert result.effective_n_positions == pytest.approx(expected)


def test_equal_weight_portfolio_has_effective_n_equal_to_real_count():
    holdings = [_holding(f"S{i}", 0.1) for i in range(10)]
    result = compute_concentration(holdings, {}, "P", _ts("2020-06-30"), ConcentrationConfig())
    assert result.effective_n_positions == pytest.approx(10.0)


def test_top_n_weight_matches_a_hand_computed_sum():
    holdings = [_holding("A", 0.4), _holding("B", 0.3), _holding("C", 0.2), _holding("D", 0.1)]
    result = compute_concentration(holdings, {}, "P", _ts("2020-06-30"), ConcentrationConfig(), top_n_values=(1, 2))
    assert result.top_n_weight[1] == pytest.approx(0.4)
    assert result.top_n_weight[2] == pytest.approx(0.7)


def test_sector_concentration_flags_a_real_breach():
    holdings = [_holding("A", 0.5), _holding("B", 0.3), _holding("C", 0.2)]
    sectors = {"A": "tech", "B": "tech", "C": "energy"}
    result = compute_concentration(holdings, sectors, "P", _ts("2020-06-30"), ConcentrationConfig(max_sector_concentration=0.5))
    assert result.sector_concentration["tech"] == pytest.approx(0.8)
    assert any("tech" in w for w in result.warnings)


def test_concentration_uses_gross_weight_for_a_long_short_book():
    holdings = [_holding("A", 0.8), _holding("B", -0.8)]  # dollar-neutral but very concentrated
    result = compute_concentration(holdings, {}, "P", _ts("2020-06-30"), ConcentrationConfig(min_effective_n_positions=5.0))
    assert result.top_n_weight[1] == pytest.approx(0.8)  # gross, not net
    assert any("effective n positions" in w for w in result.warnings)
