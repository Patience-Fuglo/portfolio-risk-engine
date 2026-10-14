"""Core data schemas for the portfolio risk model.

This engine never re-estimates a factor exposure, factor covariance,
specific variance, or stock covariance matrix -- it consumes
``factor-model-engine``'s and ``portfolio-construction-engine``'s own
real outputs (``factor_model.data.PortfolioHolding`` is reused
directly for portfolio weights, never redefined). Every schema here is
a real ANALYSIS result over an already-given, fixed portfolio -- this
engine answers "how risky is this portfolio," never "which portfolio
should we hold" (Engine 2's job) or "what explains these stocks"
(Engine 1's job).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class VolatilityResult:
    """Real portfolio volatility, decomposed into factor and specific
    risk (reusing ``portfolio_construction.risk.compute_portfolio_risk``'s
    own real `w^T Sigma w` decomposition -- never re-derived), plus
    real per-security MARGINAL contribution to risk (``(Sigma w)_i /
    portfolio_volatility``) and COMPONENT contribution
    (``w_i * marginal_contribution_i``, which sums exactly to total
    portfolio volatility -- a genuine decomposition, not an
    approximation).
    """

    as_of: pd.Timestamp
    portfolio_id: str
    portfolio_variance: float
    portfolio_volatility: float
    factor_variance: float
    specific_variance: float
    factor_risk_contribution: dict[str, float] = field(default_factory=dict)
    marginal_contribution: dict[str, float] = field(default_factory=dict)
    component_contribution: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VaRResult:
    """A real Value-at-Risk estimate: the loss magnitude (reported as a
    POSITIVE number, by convention) such that real portfolio losses are
    expected to exceed it with probability ``1 - confidence_level`` over
    ``horizon_periods`` real periods. ``method`` is ``"parametric"``
    (Gaussian, from ``VolatilityResult``'s own real volatility) or
    ``"historical"`` (from a real replayed historical return series --
    see ``historical.simulate_historical_portfolio_returns``).
    """

    as_of: pd.Timestamp
    portfolio_id: str
    confidence_level: float
    horizon_periods: int
    method: str
    var_value: float
    n_observations: int | None = None  # real sample size, for the historical method only

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.horizon_periods < 1:
            raise ValueError("horizon_periods must be >= 1")
        if self.method not in ("parametric", "historical"):
            raise ValueError(f"unknown method {self.method!r}")


@dataclass(frozen=True)
class ExpectedShortfallResult:
    """Real Expected Shortfall (CVaR): the real AVERAGE loss, GIVEN
    that the loss already exceeds the matching ``VaRResult``'s
    ``var_value`` -- a genuinely different, always-at-least-as-large
    real quantity than VaR itself, never conflated with it.
    """

    as_of: pd.Timestamp
    portfolio_id: str
    confidence_level: float
    horizon_periods: int
    method: str
    es_value: float
    n_observations: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.horizon_periods < 1:
            raise ValueError("horizon_periods must be >= 1")
        if self.method not in ("parametric", "historical"):
            raise ValueError(f"unknown method {self.method!r}")


@dataclass(frozen=True)
class StressScenarioResult:
    """A real portfolio P&L under one hypothetical or historical shock.
    ``shocked_factor_returns`` is the REAL shock applied (a factor
    name -> hypothetical return mapping, or the real historical
    period's actual factor returns for a historical replay);
    ``portfolio_pnl`` is the real resulting return, computed from the
    portfolio's own real exposure vector -- never a hand-waved estimate.
    """

    as_of: pd.Timestamp
    portfolio_id: str
    scenario_name: str
    shocked_factor_returns: dict[str, float]
    portfolio_pnl: float
    specific_pnl_assumed_zero: bool = True  # documented: stress scenarios shock FACTOR returns only


@dataclass(frozen=True)
class ConcentrationResult:
    as_of: pd.Timestamp
    portfolio_id: str
    effective_n_positions: float
    top_n_weight: dict[int, float] = field(default_factory=dict)  # e.g. {5: 0.42} = top 5 positions hold 42%
    sector_concentration: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RiskBudget:
    """A real, researcher-set target: this factor (or security) should
    account for AT MOST ``target_share`` of total real portfolio
    variance. Pure configuration -- see ``budgets.check_risk_budgets``
    for the real check against ``VolatilityResult``.
    """

    name: str
    target_share: float

    def __post_init__(self) -> None:
        if not 0 < self.target_share <= 1:
            raise ValueError("target_share must be in (0, 1]")


@dataclass(frozen=True)
class RiskBudgetResult:
    name: str
    target_share: float
    actual_share: float
    breached: bool


@dataclass(frozen=True)
class LimitBreach:
    limit_name: str
    limit_value: float
    actual_value: float
    severity: str  # "warning" | "critical"
    detail: str

    def __post_init__(self) -> None:
        if self.severity not in ("warning", "critical"):
            raise ValueError(f"unknown severity {self.severity!r}")
