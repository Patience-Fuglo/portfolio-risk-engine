import numpy as np
import pandas as pd
import pytest
from factor_model.config import RiskConfig
from factor_model.data.schemas import ExposureSubject, FactorExposureRow, PortfolioHolding
from portfolio_construction.config import RiskModelConfig
from portfolio_construction.risk_model import build_stock_covariance_matrix

from portfolio_risk.volatility import compute_volatility


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _exposure(subject_id: str, factor_name: str, value: float) -> FactorExposureRow:
    return FactorExposureRow(
        subject_type=ExposureSubject.PORTFOLIO, subject_id=subject_id, as_of=_ts("2020-06-30"),
        factor_name=factor_name, exposure_value=value, source_known_as_of=_ts("2020-06-30"),
    )


def _holding(security_id: str, weight: float) -> PortfolioHolding:
    return PortfolioHolding(portfolio_id="P", security_id=security_id, as_of=_ts("2020-06-30"), weight=weight, known_as_of=_ts("2020-06-30"))


def test_component_contributions_sum_exactly_to_portfolio_volatility():
    holdings = [_holding("A", 0.6), _holding("B", 0.4)]
    stock_cov = pd.DataFrame([[0.05, 0.01], [0.01, 0.03]], index=["A", "B"], columns=["A", "B"])
    exposure_rows = [_exposure("P", "size", 0.1)]
    factor_cov = pd.DataFrame([[0.02]], index=["size"], columns=["size"])
    specific_var = pd.Series({"A": 0.02, "B": 0.015})

    result = compute_volatility(exposure_rows, factor_cov, holdings, stock_cov, specific_var, "P", _ts("2020-06-30"), RiskConfig())
    assert sum(result.component_contribution.values()) == pytest.approx(result.portfolio_volatility)


def test_portfolio_variance_from_stock_covariance_matches_the_real_quadratic_form():
    holdings = [_holding("A", 0.6), _holding("B", 0.4)]
    stock_cov = pd.DataFrame([[0.05, 0.01], [0.01, 0.03]], index=["A", "B"], columns=["A", "B"])
    exposure_rows = [_exposure("P", "size", 0.1)]
    factor_cov = pd.DataFrame([[0.02]], index=["size"], columns=["size"])
    specific_var = pd.Series({"A": 0.02, "B": 0.015})

    result = compute_volatility(exposure_rows, factor_cov, holdings, stock_cov, specific_var, "P", _ts("2020-06-30"), RiskConfig())
    w = np.array([0.6, 0.4])
    expected_variance = float(w @ stock_cov.to_numpy() @ w)
    assert result.portfolio_variance == pytest.approx(expected_variance)
    assert result.portfolio_volatility == pytest.approx(np.sqrt(expected_variance))


def test_marginal_contribution_matches_the_real_formula():
    holdings = [_holding("A", 1.0)]
    stock_cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    exposure_rows = [_exposure("P", "size", 0.1)]
    factor_cov = pd.DataFrame([[0.02]], index=["size"], columns=["size"])
    specific_var = pd.Series({"A": 0.02})

    result = compute_volatility(exposure_rows, factor_cov, holdings, stock_cov, specific_var, "P", _ts("2020-06-30"), RiskConfig())
    # single-asset portfolio: marginal contribution = sqrt(variance) itself
    assert result.marginal_contribution["A"] == pytest.approx(0.2)  # sqrt(0.04)


def test_factor_and_specific_variance_sum_reasonably_close_to_the_real_stock_covariance_total():
    # NOT an exact identity in general (compute_portfolio_risk floors
    # missing specific variances independently), but for a fully
    # specified fixture with no missing data, the two real
    # decompositions must agree
    holdings = [_holding("A", 0.5), _holding("B", 0.5)]
    exposure_matrix = pd.DataFrame({"size": [1.0, -1.0]}, index=["A", "B"])
    factor_cov = pd.DataFrame([[0.02]], index=["size"], columns=["size"])
    specific_var = pd.Series({"A": 0.01, "B": 0.015})

    stock_cov, _ = build_stock_covariance_matrix(exposure_matrix, factor_cov, specific_var, RiskModelConfig())
    exposure_rows = [_exposure("P", "size", 0.5 * 1.0 + 0.5 * -1.0)]  # portfolio's own net exposure

    result = compute_volatility(exposure_rows, factor_cov, holdings, stock_cov, specific_var, "P", _ts("2020-06-30"), RiskConfig())
    assert (result.factor_variance + result.specific_variance) == pytest.approx(result.portfolio_variance, rel=1e-6)


def test_warns_when_portfolio_volatility_is_zero():
    holdings = [_holding("A", 1.0)]
    stock_cov = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
    exposure_rows = [_exposure("P", "size", 0.0)]
    factor_cov = pd.DataFrame([[0.0]], index=["size"], columns=["size"])
    specific_var = pd.Series({"A": 0.0})

    result = compute_volatility(exposure_rows, factor_cov, holdings, stock_cov, specific_var, "P", _ts("2020-06-30"), RiskConfig())
    assert result.marginal_contribution == {}
    assert any("zero" in w for w in result.warnings)
