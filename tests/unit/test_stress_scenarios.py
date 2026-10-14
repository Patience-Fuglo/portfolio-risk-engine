import pandas as pd
import pytest
from factor_model.data.schemas import ExposureSubject, FactorExposureRow, FactorReturnRow

from portfolio_risk.stress import apply_factor_shock, historical_worst_period_scenario


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _exposure(subject_id: str, factor_name: str, value: float) -> FactorExposureRow:
    return FactorExposureRow(
        subject_type=ExposureSubject.PORTFOLIO, subject_id=subject_id, as_of=_ts("2020-06-30"),
        factor_name=factor_name, exposure_value=value, source_known_as_of=_ts("2020-06-30"),
    )


def _return_row(period_start: str, period_end: str, factor_name: str, value: float) -> FactorReturnRow:
    return FactorReturnRow(
        period_start=_ts(period_start), period_end=_ts(period_end), factor_name=factor_name,
        factor_return=value, r_squared=0.1, n_obs=30, known_as_of=_ts(period_end),
    )


def test_apply_factor_shock_matches_the_real_linear_contribution_formula():
    exposures = [_exposure("P", "size", 0.5), _exposure("P", "momentum", -0.2)]
    shock = {"size": -0.10, "momentum": 0.05}
    result = apply_factor_shock(exposures, shock, "P", "manual_shock", _ts("2020-06-30"))
    assert result.portfolio_pnl == pytest.approx(0.5 * -0.10 + -0.2 * 0.05)


def test_apply_factor_shock_treats_a_factor_with_no_real_exposure_as_zero_contribution():
    exposures = [_exposure("P", "size", 0.5)]
    shock = {"size": -0.10, "value": -0.20}  # no real "value" exposure recorded
    result = apply_factor_shock(exposures, shock, "P", "manual_shock", _ts("2020-06-30"))
    assert result.portfolio_pnl == pytest.approx(0.5 * -0.10)


def test_historical_worst_period_scenario_finds_the_real_worst_period():
    exposures = [_exposure("P", "size", 1.0), _exposure("P", "momentum", 0.5)]
    returns = [
        _return_row("2019-01-31", "2019-02-28", "size", -0.02),
        _return_row("2019-01-31", "2019-02-28", "momentum", 0.01),
        _return_row("2019-02-28", "2019-03-31", "size", -0.15),  # the real worst
        _return_row("2019-02-28", "2019-03-31", "momentum", -0.03),
        _return_row("2019-03-31", "2019-04-30", "size", 0.05),
        _return_row("2019-03-31", "2019-04-30", "momentum", 0.02),
    ]
    result = historical_worst_period_scenario(exposures, returns, "size", "P", _ts("2020-06-30"))
    assert result.shocked_factor_returns == {"size": -0.15, "momentum": -0.03}
    assert result.portfolio_pnl == pytest.approx(1.0 * -0.15 + 0.5 * -0.03)
    assert "2019-02-28" in result.scenario_name


def test_historical_worst_period_scenario_raises_for_an_unknown_factor():
    with pytest.raises(ValueError):
        historical_worst_period_scenario([], [], "nonexistent_factor", "P", _ts("2020-06-30"))
