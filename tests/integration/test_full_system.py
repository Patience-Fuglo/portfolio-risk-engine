"""Stage 10 -- Hardening: the single most comprehensive point-in-time
check in this repo, mirroring factor-model-engine's and portfolio-
construction-engine's own Stage 10. Unlike portfolio-construction-
engine, this engine never re-runs an optimizer -- every computation
here is a deterministic analysis of a GIVEN, fixed portfolio, so a
full report should reproduce EXACTLY (no SLSQP-class floating-point
caveat to work around), both when re-run on identical data and when
the dataset is extended into the future.
"""

import pandas as pd
from factor_model.config import EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, build_point_in_time_universe, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame
from factor_model.exposures import equal_weight_benchmark_holdings

from portfolio_risk.config import EngineConfig
from portfolio_risk.data import RiskBudget
from portfolio_risk.pipeline import run_full_risk_analysis
from portfolio_risk.reporting import report_to_dict


def _run(market_obs, fundamental_obs, as_of, period_dates, fm_cfg, risk_cfg):
    universe = build_point_in_time_universe(market_obs, as_of, fm_cfg.universe, fm_cfg.data)
    holdings = equal_weight_benchmark_holdings(universe.included_ids(), as_of)
    sector_by_security = {obs.security_id: obs.sector.value for obs in market_obs if obs.as_of == as_of}
    result = run_full_risk_analysis(
        market_obs, fundamental_obs, as_of, period_dates, holdings, sector_by_security,
        budgets=[RiskBudget(name="size", target_share=0.5)], stress_factor_names=["momentum", "value"],
        fm_engine_cfg=fm_cfg, risk_engine_cfg=risk_cfg, portfolio_id="benchmark_equal_weight",
    )
    return report_to_dict(result)


def test_extending_the_dataset_into_the_future_never_changes_an_earlier_report():
    short_cfg = SyntheticDataConfig(n_securities=20, n_sectors=5, n_periods_months=36)
    long_cfg = SyntheticDataConfig(n_securities=20, n_sectors=5, n_periods_months=48)  # same seed, 12 extra REAL future months
    short_dataset = generate_synthetic_dataset(short_cfg)
    long_dataset = generate_synthetic_dataset(long_cfg)

    short_market = market_observations_from_frame(short_dataset.market)
    short_fundamentals = fundamental_observations_from_frame(short_dataset.fundamentals)
    long_market = market_observations_from_frame(long_dataset.market)  # includes 12 months AFTER as_of
    long_fundamentals = fundamental_observations_from_frame(long_dataset.fundamentals)

    short_dates = sorted(short_dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in short_dates[12:]]
    as_of = period_dates[-1]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    risk_cfg = EngineConfig()

    report_short = _run(short_market, short_fundamentals, as_of, period_dates, fm_cfg, risk_cfg)
    report_long = _run(long_market, long_fundamentals, as_of, period_dates, fm_cfg, risk_cfg)
    assert report_short == report_long


def test_full_risk_analysis_is_exactly_reproducible():
    # unlike portfolio-construction-engine's optimizer, nothing in this
    # engine calls a numerical solver -- every real computation here
    # should reproduce EXACTLY, not just within a bounded tolerance
    cfg = SyntheticDataConfig()  # real default scale
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]
    as_of = period_dates[-1]

    fm_cfg = FactorModelConfig()  # real default liquidity filters, not relaxed
    risk_cfg = EngineConfig()

    report_a = _run(market_obs, fundamental_obs, as_of, period_dates, fm_cfg, risk_cfg)
    report_b = _run(market_obs, fundamental_obs, as_of, period_dates, fm_cfg, risk_cfg)
    assert report_a == report_b
