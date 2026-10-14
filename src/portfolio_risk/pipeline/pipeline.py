"""Stage 10 -- the end-to-end orchestration: runs Stages 2-9 once, in
dependency order, for one real portfolio. Every step reuses the exact
function that stage already built and tested -- nothing here
re-derives any real computation. Depends on BOTH ``factor-model-
engine`` (universe, exposures, factor returns/covariance, specific
risk) and ``portfolio-construction-engine`` (the stock covariance
matrix) for everything upstream of this engine's own nine stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from factor_model.config import EngineConfig as FactorModelConfig
from factor_model.data.schemas import ExposureSubject, FundamentalObservation, MarketObservation, PortfolioHolding
from factor_model.data.universe import build_point_in_time_universe
from factor_model.exposures import aggregate_exposure
from factor_model.factors import compute_factor_exposure_table
from factor_model.regression import compute_factor_returns
from factor_model.risk import compute_factor_covariance, compute_specific_returns, compute_specific_variance
from portfolio_construction.config import RiskModelConfig as StockRiskModelConfig
from portfolio_construction.risk_model import build_exposure_matrix, build_stock_covariance_matrix

from ..budgets import check_risk_budgets
from ..concentration import compute_concentration
from ..config import EngineConfig
from ..data.schemas import (
    ConcentrationResult,
    ExpectedShortfallResult,
    LimitBreach,
    RiskBudget,
    RiskBudgetResult,
    StressScenarioResult,
    VaRResult,
    VolatilityResult,
)
from ..expected_shortfall import compute_historical_es, compute_parametric_es
from ..historical import simulate_historical_portfolio_returns
from ..limits import check_limits
from ..stress import historical_worst_period_scenario
from ..var import compute_historical_var, compute_parametric_var
from ..volatility import compute_volatility


@dataclass
class RiskAnalysisResult:
    as_of: pd.Timestamp
    portfolio_id: str
    volatility: VolatilityResult
    parametric_var: VaRResult
    historical_var: VaRResult
    parametric_es: ExpectedShortfallResult
    historical_es: ExpectedShortfallResult
    stress_scenarios: list[StressScenarioResult] = field(default_factory=list)
    concentration: ConcentrationResult | None = None
    budget_results: list[RiskBudgetResult] = field(default_factory=list)
    breaches: list[LimitBreach] = field(default_factory=list)


def run_full_risk_analysis(
    market_observations: Sequence[MarketObservation],
    fundamental_observations: Sequence[FundamentalObservation],
    as_of: pd.Timestamp,
    period_dates: Sequence[pd.Timestamp],
    holdings: Sequence[PortfolioHolding],
    sector_by_security: dict[str, str],
    budgets: Sequence[RiskBudget],
    stress_factor_names: Sequence[str],
    fm_engine_cfg: FactorModelConfig,
    risk_engine_cfg: EngineConfig,
    portfolio_id: str = "portfolio",
) -> RiskAnalysisResult:
    """``period_dates`` must have ``as_of`` as its last (sorted)
    element -- the real estimation window for factor returns/
    covariance/specific risk ends exactly where the analysis is dated,
    never later.
    """
    sorted_dates = sorted(period_dates)
    if sorted_dates[-1] != as_of:
        raise ValueError("as_of must be the last date in period_dates")

    universe = build_point_in_time_universe(market_observations, as_of, fm_engine_cfg.universe, fm_engine_cfg.data)
    stock_exposure_rows, _ = compute_factor_exposure_table(market_observations, fundamental_observations, as_of, universe, fm_engine_cfg)
    return_rows, _ = compute_factor_returns(market_observations, fundamental_observations, sorted_dates, fm_engine_cfg)
    factor_cov_result = compute_factor_covariance(return_rows, fm_engine_cfg.risk)
    specific_returns = compute_specific_returns(market_observations, fundamental_observations, sorted_dates, fm_engine_cfg)
    specific_variance = compute_specific_variance(specific_returns)

    portfolio_rows = aggregate_exposure(stock_exposure_rows, holdings, ExposureSubject.PORTFOLIO, portfolio_id)
    exposure_matrix = build_exposure_matrix(stock_exposure_rows)
    stock_covariance, _ = build_stock_covariance_matrix(exposure_matrix, factor_cov_result.covariance, specific_variance, StockRiskModelConfig())

    volatility = compute_volatility(
        portfolio_rows, factor_cov_result.covariance, holdings, stock_covariance, specific_variance,
        portfolio_id, as_of, fm_engine_cfg.risk,
    )
    parametric_var = compute_parametric_var(volatility, risk_engine_cfg.var)

    weights = {h.security_id: h.weight for h in holdings if h.portfolio_id == portfolio_id}
    historical_returns = simulate_historical_portfolio_returns(weights, market_observations, sorted_dates)
    historical_var = compute_historical_var(historical_returns, portfolio_id, as_of, risk_engine_cfg.var)
    parametric_es = compute_parametric_es(volatility, risk_engine_cfg.var)
    historical_es = compute_historical_es(historical_returns, portfolio_id, as_of, risk_engine_cfg.var)

    stress_scenarios = [
        historical_worst_period_scenario(portfolio_rows, return_rows, factor_name, portfolio_id, as_of)
        for factor_name in stress_factor_names
    ]

    concentration = compute_concentration(holdings, sector_by_security, portfolio_id, as_of, risk_engine_cfg.concentration)
    budget_results = check_risk_budgets(budgets, volatility)
    breaches = check_limits(parametric_var, concentration, budget_results, risk_engine_cfg.limits, risk_engine_cfg.concentration)

    return RiskAnalysisResult(
        as_of=as_of,
        portfolio_id=portfolio_id,
        volatility=volatility,
        parametric_var=parametric_var,
        historical_var=historical_var,
        parametric_es=parametric_es,
        historical_es=historical_es,
        stress_scenarios=stress_scenarios,
        concentration=concentration,
        budget_results=budget_results,
        breaches=breaches,
    )
