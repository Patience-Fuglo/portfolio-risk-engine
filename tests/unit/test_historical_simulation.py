import pandas as pd
import pytest
from factor_model.data.schemas import MarketObservation, Sector

from portfolio_risk.historical import simulate_historical_portfolio_returns


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _market(security_id: str, as_of: str, price: float) -> MarketObservation:
    return MarketObservation(
        security_id=security_id, ticker=security_id, as_of=_ts(as_of), known_as_of=_ts(as_of),
        sector=Sector.TECHNOLOGY, is_active=True, price=price, shares_outstanding=1.0e6,
        market_cap=price * 1.0e6, adv_20d=1.0e6,
    )


def test_simulation_matches_a_hand_computed_weighted_return():
    market = [
        _market("A", "2020-01-31", 100.0), _market("B", "2020-01-31", 50.0),
        _market("A", "2020-02-29", 110.0), _market("B", "2020-02-29", 45.0),
    ]
    weights = {"A": 0.6, "B": 0.4}
    result = simulate_historical_portfolio_returns(weights, market, [_ts("2020-01-31"), _ts("2020-02-29")])
    assert result.iloc[0] == pytest.approx(0.6 * 0.10 + 0.4 * -0.10)


def test_simulation_holds_weights_fixed_across_multiple_periods():
    market = [
        _market("A", "2020-01-31", 100.0), _market("A", "2020-02-29", 110.0), _market("A", "2020-03-31", 99.0),
    ]
    result = simulate_historical_portfolio_returns({"A": 1.0}, market, [_ts("2020-01-31"), _ts("2020-02-29"), _ts("2020-03-31")])
    assert result.iloc[0] == pytest.approx(0.10)
    assert result.iloc[1] == pytest.approx(99.0 / 110.0 - 1.0)


def test_simulation_excludes_a_security_with_no_real_price_rather_than_zero_filling():
    market = [_market("A", "2020-01-31", 100.0), _market("A", "2020-02-29", 120.0)]  # B has no real price at all
    result = simulate_historical_portfolio_returns({"A": 0.5, "B": 0.5}, market, [_ts("2020-01-31"), _ts("2020-02-29")])
    assert result.iloc[0] == pytest.approx(0.5 * 0.20)  # NOT 0.5*0.20 + 0.5*0.0 presented as if B were known to be flat


def test_simulation_returns_empty_series_for_fewer_than_two_periods():
    result = simulate_historical_portfolio_returns({"A": 1.0}, [], [_ts("2020-01-31")])
    assert result.empty
