import pandas as pd
import pytest

from portfolio_risk.config import VaRConfig
from portfolio_risk.data import VolatilityResult
from portfolio_risk.expected_shortfall import compute_historical_es, compute_parametric_es
from portfolio_risk.var import compute_historical_var, compute_parametric_var


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _volatility(vol: float) -> VolatilityResult:
    return VolatilityResult(
        as_of=_ts("2020-06-30"), portfolio_id="P", portfolio_variance=vol**2, portfolio_volatility=vol,
        factor_variance=0.0, specific_variance=vol**2,
    )


def test_parametric_es_is_always_at_least_var():
    cfg = VaRConfig(confidence_level=0.95, horizon_periods=1)
    var = compute_parametric_var(_volatility(0.05), cfg)
    es = compute_parametric_es(_volatility(0.05), cfg)
    assert es.es_value >= var.var_value


def test_parametric_es_matches_the_real_closed_form_gaussian_formula():
    cfg = VaRConfig(confidence_level=0.95, horizon_periods=1)
    es = compute_parametric_es(_volatility(0.05), cfg)
    # z_0.95 ~ 1.6448536, phi(z) ~ 0.1031356 -- real, known constants
    sigma = 0.05
    expected = sigma * 0.1031356 / (1.0 - 0.95)
    assert es.es_value == pytest.approx(expected, rel=1e-4)


def test_historical_es_is_always_at_least_historical_var():
    returns = pd.Series([-0.15, -0.10, -0.08, -0.05, -0.02, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    cfg = VaRConfig(confidence_level=0.90, horizon_periods=1)
    var = compute_historical_var(returns, "P", _ts("2020-06-30"), cfg)
    es = compute_historical_es(returns, "P", _ts("2020-06-30"), cfg)
    assert es.es_value >= var.var_value


def test_historical_es_matches_a_hand_computed_tail_average():
    returns = pd.Series([-0.20, -0.10, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    cfg = VaRConfig(confidence_level=0.80, horizon_periods=1)  # worst 20% = worst 2 of 10
    es = compute_historical_es(returns, "P", _ts("2020-06-30"), cfg)
    # the threshold at the 20th percentile and everything AT or below it
    threshold = returns.quantile(0.20)
    tail = returns[returns <= threshold]
    assert es.es_value == pytest.approx(-tail.mean())


def test_historical_es_is_nan_with_too_few_real_observations():
    cfg = VaRConfig()
    es = compute_historical_es(pd.Series([0.01]), "P", _ts("2020-06-30"), cfg)
    assert pd.isna(es.es_value)
