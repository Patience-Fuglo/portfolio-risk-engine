"""Stage 10 -- renders a real ``RiskAnalysisResult`` into a
reproducible report: a JSON-serializable dict, and a professional
Markdown document. Both derived from the SAME object -- never
recomputed separately, so they can't silently disagree. Mirrors
factor-model-engine's and portfolio-construction-engine's own
reporting convention.
"""

from __future__ import annotations

from ..pipeline import RiskAnalysisResult


def report_to_dict(result: RiskAnalysisResult) -> dict:
    return {
        "as_of": str(result.as_of.date()),
        "portfolio_id": result.portfolio_id,
        "volatility": {
            "portfolio_volatility": result.volatility.portfolio_volatility,
            "factor_variance": result.volatility.factor_variance,
            "specific_variance": result.volatility.specific_variance,
            "factor_risk_contribution": result.volatility.factor_risk_contribution,
        },
        "var": {
            "parametric": result.parametric_var.var_value,
            "historical": result.historical_var.var_value,
            "confidence_level": result.parametric_var.confidence_level,
        },
        "expected_shortfall": {
            "parametric": result.parametric_es.es_value,
            "historical": result.historical_es.es_value,
        },
        "stress_scenarios": [
            {"scenario_name": s.scenario_name, "portfolio_pnl": s.portfolio_pnl, "shocked_factor_returns": s.shocked_factor_returns}
            for s in result.stress_scenarios
        ],
        "concentration": {
            "effective_n_positions": result.concentration.effective_n_positions if result.concentration else None,
            "sector_concentration": result.concentration.sector_concentration if result.concentration else {},
        },
        "budget_results": [
            {"name": b.name, "target_share": b.target_share, "actual_share": b.actual_share, "breached": b.breached}
            for b in result.budget_results
        ],
        "breaches": [
            {"limit_name": b.limit_name, "limit_value": b.limit_value, "actual_value": b.actual_value, "severity": b.severity, "detail": b.detail}
            for b in result.breaches
        ],
    }


def render_markdown_report(result: RiskAnalysisResult) -> str:
    payload = report_to_dict(result)
    lines: list[str] = []
    lines.append("# Portfolio Risk Model -- Research Report")
    lines.append("")
    lines.append(f"**As of:** {payload['as_of']}  **Portfolio:** `{payload['portfolio_id']}`")
    lines.append("")
    lines.append("All data in this report is synthetic and non-proprietary.")
    lines.append("")

    vol = payload["volatility"]
    lines.append("## Volatility")
    lines.append(f"- Portfolio volatility: {vol['portfolio_volatility']:.4%}")
    lines.append(f"- Factor variance: {vol['factor_variance']:.6f}  Specific variance: {vol['specific_variance']:.6f}")
    lines.append("")

    var = payload["var"]
    es = payload["expected_shortfall"]
    lines.append(f"## VaR / Expected Shortfall (confidence {var['confidence_level']:.0%})")
    lines.append("| Method | VaR | Expected Shortfall |")
    lines.append("|---|---|---|")
    lines.append(f"| Parametric | {var['parametric']:.4%} | {es['parametric']:.4%} |")
    lines.append(f"| Historical | {var['historical']:.4%} | {es['historical']:.4%} |")
    lines.append("")

    if payload["stress_scenarios"]:
        lines.append("## Stress Scenarios")
        lines.append("| Scenario | Portfolio P&L |")
        lines.append("|---|---|")
        for s in payload["stress_scenarios"]:
            lines.append(f"| {s['scenario_name']} | {s['portfolio_pnl']:.4%} |")
        lines.append("")

    conc = payload["concentration"]
    lines.append("## Concentration")
    lines.append(f"- Effective number of positions: {conc['effective_n_positions']:.1f}" if conc["effective_n_positions"] else "- Effective number of positions: n/a")
    for sector, weight in sorted(conc["sector_concentration"].items()):
        lines.append(f"- {sector}: {weight:.1%}")
    lines.append("")

    if payload["budget_results"]:
        lines.append("## Risk Budgets")
        lines.append("| Name | Target | Actual | Breached |")
        lines.append("|---|---|---|---|")
        for b in payload["budget_results"]:
            lines.append(f"| {b['name']} | {b['target_share']:.1%} | {b['actual_share']:.1%} | {b['breached']} |")
        lines.append("")

    lines.append("## Limit Breaches")
    if payload["breaches"]:
        for b in payload["breaches"]:
            lines.append(f"- **[{b['severity'].upper()}]** {b['limit_name']}: {b['detail']}")
    else:
        lines.append("No real limit breaches.")
    lines.append("")

    return "\n".join(lines)
