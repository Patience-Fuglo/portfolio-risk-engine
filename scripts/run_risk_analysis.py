#!/usr/bin/env python3
"""End-to-end CLI: generates factor-model-engine's own synthetic
dataset in-process, builds an equal-weight benchmark portfolio, runs
the full Stage 2-9 risk analysis, and writes one reproducible JSON +
Markdown report.

Usage:
    python scripts/run_risk_analysis.py --n-periods-months 36 [--confidence-level 0.95] [--output-dir outputs/reports]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from factor_model.config import EngineConfig as FactorModelConfig
from factor_model.data import SyntheticDataConfig, build_point_in_time_universe, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame
from factor_model.exposures import equal_weight_benchmark_holdings

from portfolio_risk.config import EngineConfig, VaRConfig
from portfolio_risk.data import RiskBudget
from portfolio_risk.pipeline import run_full_risk_analysis
from portfolio_risk.reporting import render_markdown_report, report_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-securities", type=int, default=25)
    parser.add_argument("--n-periods-months", type=int, default=36)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--portfolio-id", default="benchmark_equal_weight")
    parser.add_argument("--output-dir", default="outputs/reports")
    args = parser.parse_args()

    dataset = generate_synthetic_dataset(SyntheticDataConfig(n_securities=args.n_securities, n_sectors=5, n_periods_months=args.n_periods_months))
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates[12:]]
    as_of = period_dates[-1]

    fm_cfg = FactorModelConfig()
    universe = build_point_in_time_universe(market_obs, as_of, fm_cfg.universe, fm_cfg.data)
    holdings = equal_weight_benchmark_holdings(universe.included_ids(), as_of, portfolio_id=args.portfolio_id)
    sector_by_security = {obs.security_id: obs.sector.value for obs in market_obs if obs.as_of == as_of}

    risk_cfg = EngineConfig(var=VaRConfig(confidence_level=args.confidence_level))
    result = run_full_risk_analysis(
        market_obs, fundamental_obs, as_of, period_dates, holdings, sector_by_security,
        budgets=[RiskBudget(name="size", target_share=0.5), RiskBudget(name="momentum", target_share=0.5)],
        stress_factor_names=["momentum", "value", "size", "quality"],
        fm_engine_cfg=fm_cfg, risk_engine_cfg=risk_cfg, portfolio_id=args.portfolio_id,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"risk_report_{args.portfolio_id}.json"
    md_path = out_dir / f"risk_report_{args.portfolio_id}.md"

    with open(json_path, "w") as f:
        json.dump(report_to_dict(result), f, indent=2, sort_keys=True)
        f.write("\n")
    with open(md_path, "w") as f:
        f.write(render_markdown_report(result))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"portfolio volatility: {result.volatility.portfolio_volatility:.4%}")
    print(f"parametric VaR: {result.parametric_var.var_value:.4%}  historical VaR: {result.historical_var.var_value:.4%}")
    print(f"{len(result.breaches)} limit breach(es)")
    for b in result.breaches:
        print(f"  [{b.severity.upper()}] {b.limit_name}: {b.detail}")


if __name__ == "__main__":
    main()
