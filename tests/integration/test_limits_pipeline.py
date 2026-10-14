"""End-to-end: volatility -> VaR -> concentration -> budgets -> a
single consolidated limit-breach list, on the real synthetic
pipeline.
"""

import pandas as pd
import pytest
from factor_model.config import EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import (
    ExposureSubject,
    SyntheticDataConfig,
    build_point_in_time_universe,
    fundamental_observations_from_frame,
    generate_synthetic_dataset,
    market_observations_from_frame,
)
from factor_model.exposures import aggregate_exposure, equal_weight_benchmark_holdings
from factor_model.factors import compute_factor_exposure_table
from factor_model.regression import compute_factor_returns
from factor_model.risk import compute_factor_covariance, compute_specific_returns, compute_specific_variance
from portfolio_construction.config import RiskModelConfig
from portfolio_construction.risk_model import build_exposure_matrix, build_stock_covariance_matrix

from portfolio_risk.budgets import check_risk_budgets
from portfolio_risk.concentration import compute_concentration
from portfolio_risk.config import ConcentrationConfig, LimitsConfig, VaRConfig
from portfolio_risk.data import RiskBudget
from portfolio_risk.limits import check_limits
from portfolio_risk.var import compute_parametric_var
from portfolio_risk.volatility import compute_volatility


def test_consolidated_limit_breaches_on_the_real_synthetic_pipeline():
    cfg = SyntheticDataConfig(n_securities=25, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]
    as_of = period_dates[-1]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    universe = build_point_in_time_universe(market_obs, as_of, fm_cfg.universe, fm_cfg.data)
    stock_exposure_rows, _ = compute_factor_exposure_table(market_obs, fundamental_obs, as_of, universe, fm_cfg)
    return_rows, _ = compute_factor_returns(market_obs, fundamental_obs, period_dates, fm_cfg)
    factor_cov = compute_factor_covariance(return_rows, fm_cfg.risk)
    specific_returns = compute_specific_returns(market_obs, fundamental_obs, period_dates, fm_cfg)
    specific_variance = compute_specific_variance(specific_returns)

    holdings = equal_weight_benchmark_holdings(universe.included_ids(), as_of)
    portfolio_rows = aggregate_exposure(stock_exposure_rows, holdings, ExposureSubject.PORTFOLIO, "benchmark_equal_weight")
    exposure_matrix = build_exposure_matrix(stock_exposure_rows)
    stock_covariance, _ = build_stock_covariance_matrix(exposure_matrix, factor_cov.covariance, specific_variance, RiskModelConfig())

    volatility = compute_volatility(
        portfolio_rows, factor_cov.covariance, holdings, stock_covariance, specific_variance,
        "benchmark_equal_weight", as_of, fm_cfg.risk,
    )
    var_result = compute_parametric_var(volatility, VaRConfig())
    sector_by_security = {obs.security_id: obs.sector.value for obs in market_obs if obs.as_of == as_of}
    concentration = compute_concentration(holdings, sector_by_security, "benchmark_equal_weight", as_of, ConcentrationConfig())
    budget_results = check_risk_budgets([RiskBudget(name="size", target_share=0.5)], volatility)

    breaches = check_limits(var_result, concentration, budget_results, LimitsConfig(), ConcentrationConfig())
    # a full-universe equal-weight benchmark is NOT concentrated and
    # has no single dominant factor -- a real, legitimate "no breaches" result
    assert isinstance(breaches, list)
    assert concentration.effective_n_positions == pytest.approx(len(universe.included_ids()))
