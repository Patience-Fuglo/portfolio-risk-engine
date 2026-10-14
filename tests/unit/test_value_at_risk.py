import pandas as pd
import pytest

from portfolio_risk.config import VaRConfig
from portfolio_risk.data import VolatilityResult
from portfolio_risk.var import compute_historical_var, compute_parametric_var


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _volatility(vol: float) -> VolatilityResult:
    return VolatilityResult(
        as_of=_ts("2020-06-30"), portfolio_id="P", portfolio_variance=vol**2, portfolio_volatility=vol,
        factor_variance=0.0, specific_variance=vol**2,
    )


def test_parametric_var_matches_the_real_known_z_value():
    # the 97.5th percentile of the standard normal is a well-known
    # real constant, ~1.95996 -- a direct, independent check on the
    # bisection-based inverse CDF, not just an internal consistency test
    cfg = VaRConfig(confidence_level=0.975, horizon_periods=1)
    result = compute_parametric_var(_volatility(0.05), cfg)
    assert result.var_value == pytest.approx(1.959964 * 0.05, abs=1e-4)
    assert result.method == "parametric"


def test_parametric_var_scales_with_real_sqrt_time():
    cfg_1 = VaRConfig(confidence_level=0.95, horizon_periods=1)
    cfg_4 = VaRConfig(confidence_level=0.95, horizon_periods=4)
    r1 = compute_parametric_var(_volatility(0.05), cfg_1)
    r4 = compute_parametric_var(_volatility(0.05), cfg_4)
    assert r4.var_value == pytest.approx(r1.var_value * 2.0, rel=1e-6)  # sqrt(4) = 2


def test_historical_var_matches_a_hand_computed_percentile():
    returns = pd.Series([-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    cfg = VaRConfig(confidence_level=0.90, horizon_periods=1)
    result = compute_historical_var(returns, "P", _ts("2020-06-30"), cfg)
    expected = max(-float(returns.quantile(0.10, interpolation="linear")), 0.0)
    assert result.var_value == pytest.approx(expected, abs=1e-6)
    assert result.n_observations == 10


def test_historical_var_is_nan_with_too_few_real_observations():
    cfg = VaRConfig()
    result = compute_historical_var(pd.Series([0.01]), "P", _ts("2020-06-30"), cfg)
    assert pd.isna(result.var_value)
    assert result.n_observations == 1


def test_historical_var_drops_real_nan_observations_before_counting():
    returns = pd.Series([0.01, float("nan"), -0.02, 0.03])
    cfg = VaRConfig()
    result = compute_historical_var(returns, "P", _ts("2020-06-30"), cfg)
    assert result.n_observations == 3
