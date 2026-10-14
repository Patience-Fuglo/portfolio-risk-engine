"""End-to-end: a real historical worst-period stress scenario, found
from factor-model-engine's own real factor-return history and applied
to a real portfolio's current exposure.
"""

import pandas as pd
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

from portfolio_risk.stress import historical_worst_period_scenario


def test_real_historical_worst_period_stress_scenario_on_the_synthetic_pipeline():
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

    holdings = equal_weight_benchmark_holdings(universe.included_ids(), as_of)
    portfolio_rows = aggregate_exposure(stock_exposure_rows, holdings, ExposureSubject.PORTFOLIO, "benchmark_equal_weight")

    result = historical_worst_period_scenario(portfolio_rows, return_rows, "momentum", "benchmark_equal_weight", as_of)
    # confirm it's genuinely the worst real period for momentum
    momentum_rows = [r for r in return_rows if r.factor_name == "momentum"]
    worst = min(r.factor_return for r in momentum_rows)
    assert result.shocked_factor_returns["momentum"] == worst
