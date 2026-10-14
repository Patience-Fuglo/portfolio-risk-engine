"""End-to-end: the full Stage 10 orchestration (Stages 2-9) and its
reproducible report, on the real synthetic dataset.
"""

import json

import pandas as pd
import pytest
from factor_model.config import EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, build_point_in_time_universe, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame
from factor_model.exposures import equal_weight_benchmark_holdings

from portfolio_risk.config import EngineConfig
from portfolio_risk.data import RiskBudget
from portfolio_risk.pipeline import run_full_risk_analysis
from portfolio_risk.reporting import render_markdown_report, report_to_dict


@pytest.fixture(scope="module")
def risk_analysis_result():
    cfg = SyntheticDataConfig(n_securities=25, n_sectors=5, n_periods_months=36)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]
    as_of = period_dates[-1]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    universe = build_point_in_time_universe(market_obs, as_of, fm_cfg.universe, fm_cfg.data)
    holdings = equal_weight_benchmark_holdings(universe.included_ids(), as_of)
    sector_by_security = {obs.security_id: obs.sector.value for obs in market_obs if obs.as_of == as_of}

    risk_cfg = EngineConfig()
    return run_full_risk_analysis(
        market_obs, fundamental_obs, as_of, period_dates, holdings, sector_by_security,
        budgets=[RiskBudget(name="size", target_share=0.5)], stress_factor_names=["momentum", "value"],
        fm_engine_cfg=fm_cfg, risk_engine_cfg=risk_cfg, portfolio_id="benchmark_equal_weight",
    )


def test_pipeline_result_satisfies_the_real_es_at_least_var_invariant(risk_analysis_result):
    r = risk_analysis_result
    assert r.parametric_es.es_value >= r.parametric_var.var_value
    assert r.historical_es.es_value >= r.historical_var.var_value


def test_report_to_dict_is_json_serializable(risk_analysis_result):
    json.dumps(report_to_dict(risk_analysis_result))


def test_render_markdown_report_contains_every_real_section(risk_analysis_result):
    markdown = render_markdown_report(risk_analysis_result)
    for heading in ["## Volatility", "## VaR / Expected Shortfall", "## Stress Scenarios", "## Concentration", "## Risk Budgets", "## Limit Breaches"]:
        assert heading in markdown
    assert "synthetic" in markdown.lower()


def test_report_rendering_is_deterministic(risk_analysis_result):
    first = render_markdown_report(risk_analysis_result)
    second = render_markdown_report(risk_analysis_result)
    assert first == second
    assert report_to_dict(risk_analysis_result) == report_to_dict(risk_analysis_result)


def test_pipeline_rejects_mismatched_as_of():
    cfg = SyntheticDataConfig(n_securities=10, n_sectors=5, n_periods_months=20, restatement_enabled=False)
    dataset = generate_synthetic_dataset(cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates]
    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    universe = build_point_in_time_universe(market_obs, period_dates[-1], fm_cfg.universe, fm_cfg.data)
    holdings = equal_weight_benchmark_holdings(universe.included_ids(), period_dates[-1])
    with pytest.raises(ValueError):
        run_full_risk_analysis(
            market_obs, fundamental_obs, period_dates[-2], period_dates, holdings, {},
            budgets=[], stress_factor_names=[], fm_engine_cfg=fm_cfg, risk_engine_cfg=EngineConfig(),
        )
