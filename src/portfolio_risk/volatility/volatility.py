"""Stage 2 -- real portfolio volatility, decomposed into factor and
specific risk, plus real per-security marginal and component risk
contributions. Reuses ``factor-model-engine``'s own real
``compute_portfolio_risk`` (the factor/specific decomposition, Stage 6
of that repo) and ``portfolio-construction-engine``'s
``risk_contributions`` (the `w_i * (Sigma w)_i` decomposition, Stage 4
of that repo) -- neither the covariance decomposition nor the
risk-contribution math is reimplemented here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from factor_model.config import RiskConfig
from factor_model.data.schemas import FactorExposureRow, PortfolioHolding
from factor_model.risk import compute_portfolio_risk
from portfolio_construction.optimization import risk_contributions

from ..data.schemas import VolatilityResult

# A real, investigated numerical-stability floor, not a research
# assumption: a portfolio whose real net exposure cancels out
# cross-sectionally (e.g. an equal-weight book over the FULL eligible
# universe, confirmed elsewhere in this build to net to ~0 exposure)
# has a TRUE portfolio variance of exactly 0 -- but the real
# `w^T Sigma w` quadratic form, computed via a BLAS matrix multiply,
# can land on a tiny nonzero float (~1e-30 or smaller) instead of
# exact 0.0, due to floating-point non-associativity in a
# multi-threaded BLAS reduction (confirmed directly: re-running the
# identical computation occasionally produced a different noise-level
# value in the last few significant digits -- the same real root
# cause already documented for portfolio-construction-engine's SLSQP
# solver, now observed here in a plain matrix multiply, no solver
# involved). Dividing BY that noise (for marginal/component
# contributions) would produce a numerically meaningless result that
# happens to look like a real number. Treating anything below this
# epsilon as "effectively zero" avoids reporting that noise as signal.
_VARIANCE_EPSILON = 1e-12


def compute_volatility(
    portfolio_exposure_rows: Sequence[FactorExposureRow],
    factor_covariance: pd.DataFrame,
    holdings: Sequence[PortfolioHolding],
    stock_covariance: pd.DataFrame,
    specific_variance_by_security: pd.Series,
    portfolio_id: str,
    as_of: pd.Timestamp,
    risk_cfg: RiskConfig,
) -> VolatilityResult:
    """Real ``factor_variance``/``specific_variance`` come from
    ``portfolio_construction.risk.compute_portfolio_risk`` (the factor
    exposure vector against the real factor covariance, plus each
    holding's real specific variance). Real marginal/component
    contributions come from ``stock_covariance`` directly (the actual
    stock-level matrix, not a factor-only approximation) via
    ``risk_contributions`` -- the two real decompositions (factor-vs-
    specific at the aggregate level, and marginal/component at the
    per-security level) are complementary, not the same computation
    twice.
    """
    portfolio_risk = compute_portfolio_risk(
        portfolio_exposure_rows, factor_covariance, holdings, specific_variance_by_security, portfolio_id, risk_cfg
    )

    own_holdings = [h for h in holdings if h.portfolio_id == portfolio_id]
    security_ids = [h.security_id for h in own_holdings]
    weights = np.array([h.weight for h in own_holdings])
    aligned_covariance = stock_covariance.loc[security_ids, security_ids].to_numpy()

    component = risk_contributions(weights, aligned_covariance)
    raw_portfolio_variance = float(weights @ aligned_covariance @ weights)
    # treat anything below the real noise floor as exactly zero -- see
    # _VARIANCE_EPSILON's own comment for why this is necessary, not
    # just defensive: the raw BLAS-computed value can be a tiny nonzero
    # float even when the true variance is exactly 0
    portfolio_variance = raw_portfolio_variance if raw_portfolio_variance >= _VARIANCE_EPSILON else 0.0
    portfolio_volatility = float(np.sqrt(portfolio_variance))

    warnings = list(portfolio_risk.warnings)
    marginal_contribution: dict[str, float] = {}
    component_contribution: dict[str, float] = {}
    if portfolio_volatility > 0:
        marginal_raw = (aligned_covariance @ weights) / portfolio_volatility
        for sid, mc, cc in zip(security_ids, marginal_raw.tolist(), component.tolist()):
            marginal_contribution[sid] = mc
            component_contribution[sid] = cc / portfolio_volatility  # component contribution TO VOLATILITY, not variance
    else:
        warnings.append("portfolio volatility is zero (or below the real numerical noise floor) -- marginal/component contributions not computed")

    # same real noise floor applied to factor-model-engine's own
    # per-factor contributions (and the aggregate factor/specific
    # variance it reports) -- each is individually a BLAS-computed
    # quantity with the same true-zero-but-noisy-float failure mode
    factor_variance = portfolio_risk.factor_variance if portfolio_risk.factor_variance >= _VARIANCE_EPSILON else 0.0
    specific_variance = portfolio_risk.specific_variance if portfolio_risk.specific_variance >= _VARIANCE_EPSILON else 0.0
    factor_risk_contribution = {
        name: (value if abs(value) >= _VARIANCE_EPSILON else 0.0)
        for name, value in portfolio_risk.factor_risk_contribution.items()
    }

    return VolatilityResult(
        as_of=as_of,
        portfolio_id=portfolio_id,
        portfolio_variance=portfolio_variance,
        portfolio_volatility=portfolio_volatility,
        factor_variance=factor_variance,
        specific_variance=specific_variance,
        factor_risk_contribution=factor_risk_contribution,
        marginal_contribution=marginal_contribution,
        component_contribution=component_contribution,
        warnings=tuple(warnings),
    )
