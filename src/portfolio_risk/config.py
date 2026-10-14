"""Engine configuration: every ambiguous research decision lives here,
in YAML, with a recommended default and reasoning in
``docs/research_assumptions.md`` -- never hardcoded in source. Same
plain-``@dataclass`` house style as ``factor-model-engine`` and
``portfolio-construction-engine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class RunConfig:
    output_dir: str = "outputs"
    random_seed: int = 1337


@dataclass(frozen=True)
class VaRConfig:
    # RESEARCH ASSUMPTION -- the standard real risk-management convention
    confidence_level: float = 0.95
    # RESEARCH ASSUMPTION -- VaR/ES horizon, in real periods (matching
    # whatever periodicity the underlying returns/covariance use --
    # monthly for this engine's synthetic demonstrations)
    horizon_periods: int = 1
    # RESEARCH ASSUMPTION -- below this many real historical
    # observations, historical VaR/ES is flagged unreliable, not
    # silently trusted
    min_historical_observations: int = 24

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.horizon_periods < 1:
            raise ValueError("horizon_periods must be >= 1")
        if self.min_historical_observations < 2:
            raise ValueError("min_historical_observations must be >= 2")


@dataclass(frozen=True)
class ConcentrationConfig:
    # RESEARCH ASSUMPTION -- below this many real effective positions
    # (Herfindahl-based), a portfolio is flagged concentrated
    min_effective_n_positions: float = 5.0
    # RESEARCH ASSUMPTION -- no single real sector may exceed this
    # share of total portfolio weight
    max_sector_concentration: float = 0.40

    def __post_init__(self) -> None:
        if self.min_effective_n_positions <= 0:
            raise ValueError("min_effective_n_positions must be positive")
        if not 0 < self.max_sector_concentration <= 1:
            raise ValueError("max_sector_concentration must be in (0, 1]")


@dataclass(frozen=True)
class LimitsConfig:
    # RESEARCH ASSUMPTION -- real VaR, as a fraction of portfolio value,
    # above which a limit breach is flagged "critical" rather than "warning"
    max_var_fraction_critical: float = 0.15
    # RESEARCH ASSUMPTION -- the "warning" threshold, below the critical one
    max_var_fraction_warning: float = 0.08

    def __post_init__(self) -> None:
        if self.max_var_fraction_critical <= 0 or self.max_var_fraction_warning <= 0:
            raise ValueError("limit fractions must be positive")
        if self.max_var_fraction_warning > self.max_var_fraction_critical:
            raise ValueError("max_var_fraction_warning cannot exceed max_var_fraction_critical")


@dataclass(frozen=True)
class EngineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    var: VaRConfig = field(default_factory=VaRConfig)
    concentration: ConcentrationConfig = field(default_factory=ConcentrationConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)


def load_config(path: str) -> EngineConfig:
    """Load a real YAML config file into a validated EngineConfig.
    Missing sections fall back to each dataclass's own documented
    default -- never a silent empty/zero value."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    run = RunConfig(**raw.get("run", {}))
    var = VaRConfig(**raw.get("var", {}))
    concentration = ConcentrationConfig(**raw.get("concentration", {}))
    limits = LimitsConfig(**raw.get("limits", {}))

    return EngineConfig(run=run, var=var, concentration=concentration, limits=limits)
