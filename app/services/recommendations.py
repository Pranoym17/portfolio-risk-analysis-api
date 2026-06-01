from __future__ import annotations

import pandas as pd


def _recommendation(
    *,
    rec_type: str,
    severity: str,
    title: str,
    explanation: str,
    suggested_action: str,
    related_assets: list[str] | None = None,
) -> dict:
    return {
        "type": rec_type,
        "severity": severity,
        "title": title,
        "explanation": explanation,
        "suggested_action": suggested_action,
        "related_assets": related_assets or [],
    }


def _severity_rank(severity: str) -> int:
    return {"critical": 0, "warning": 1, "info": 2}.get(severity, 3)


def build_portfolio_recommendations(
    *,
    current_weights: dict[str, float],
    risk_attribution: list[dict],
    concentration: dict,
    correlation: pd.DataFrame,
    optimized_weights: list[dict],
    optimization_summary: dict,
    rebalance_result: dict,
    correlation_threshold: float,
) -> dict:
    recommendations: list[dict] = []

    if concentration.get("concentration_level") in {"moderate", "high"}:
        top_asset = concentration.get("top_asset_ticker")
        top_asset_pct = float(concentration.get("top_asset_trc_pct") or 0.0)
        severity = "critical" if concentration.get("concentration_level") == "high" else "warning"
        recommendations.append(
            _recommendation(
                rec_type="risk_concentration",
                severity=severity,
                title="Reduce concentration in the largest risk driver",
                explanation=(
                    f"{top_asset} contributes {top_asset_pct:.0%} of total portfolio volatility, "
                    "which is higher than its role should be in a balanced allocation."
                ),
                suggested_action="Review the top risk contributor and consider shifting weight toward lower-correlated holdings.",
                related_assets=[top_asset] if top_asset else [],
            )
        )

    if concentration.get("top_sector_trc_pct", 0.0) >= 0.45:
        recommendations.append(
            _recommendation(
                rec_type="sector_concentration",
                severity="warning",
                title="Reduce sector-level risk concentration",
                explanation=(
                    f"{concentration.get('top_sector')} contributes "
                    f"{float(concentration.get('top_sector_trc_pct') or 0.0):.0%} of total volatility."
                ),
                suggested_action="Add exposure outside the dominant sector or lower weights in holdings from that sector.",
                related_assets=[],
            )
        )

    correlated_pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(correlation.columns):
        for right in correlation.columns[i + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= correlation_threshold:
                correlated_pairs.append((left, right, float(value)))

    correlated_pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    if correlated_pairs:
        left, right, value = correlated_pairs[0]
        recommendations.append(
            _recommendation(
                rec_type="redundant_correlation",
                severity="warning",
                title="Review highly correlated holdings",
                explanation=(
                    f"{left} and {right} have a correlation of {value:.2f}, "
                    "so they may be adding overlapping exposure rather than diversification."
                ),
                suggested_action="Consider keeping the stronger conviction holding or lowering one of the two weights.",
                related_assets=[left, right],
            )
        )

    target_weights = {item["ticker"]: float(item["weight"]) for item in optimized_weights}
    optimizer_gap_assets = []
    for ticker in sorted(set(current_weights) | set(target_weights)):
        current = float(current_weights.get(ticker, 0.0))
        target = float(target_weights.get(ticker, 0.0))
        if abs(target - current) >= 0.05:
            optimizer_gap_assets.append(ticker)

    if optimizer_gap_assets:
        recommendations.append(
            _recommendation(
                rec_type="optimization_gap",
                severity="info",
                title="Move closer to the optimized allocation",
                explanation=(
                    f"The optimized portfolio has an expected Sharpe of "
                    f"{float(optimization_summary.get('expected_sharpe') or 0.0):.2f} under current assumptions."
                ),
                suggested_action="Use the optimizer output as the target allocation for the next rebalance review.",
                related_assets=optimizer_gap_assets[:5],
            )
        )

    if rebalance_result.get("rebalance_needed"):
        trade_assets = [item["ticker"] for item in rebalance_result.get("trades", []) if item["action"] != "hold"]
        recommendations.append(
            _recommendation(
                rec_type="rebalance_needed",
                severity="warning",
                title="Rebalance portfolio toward target weights",
                explanation=(
                    f"Maximum allocation drift is {float(rebalance_result.get('max_abs_drift') or 0.0):.0%}, "
                    "above the configured drift tolerance."
                ),
                suggested_action="Review the generated buy/sell list and rebalance positions outside the tolerance band.",
                related_assets=trade_assets,
            )
        )

    if not recommendations:
        recommendations.append(
            _recommendation(
                rec_type="portfolio_health",
                severity="info",
                title="Portfolio is within current risk and allocation tolerances",
                explanation="No major concentration, redundancy, or rebalance issues were detected under the current settings.",
                suggested_action="Keep monitoring drift and rerun analysis when holdings or market conditions change.",
                related_assets=[],
            )
        )

    recommendations.sort(key=lambda item: (_severity_rank(item["severity"]), item["type"]))

    return {
        "recommendations": recommendations,
        "diagnostics": {
            "recommendation_count": len(recommendations),
            "concentration_level": concentration.get("concentration_level"),
            "top_asset_ticker": concentration.get("top_asset_ticker"),
            "top_asset_trc_pct": concentration.get("top_asset_trc_pct"),
            "top_sector": concentration.get("top_sector"),
            "top_sector_trc_pct": concentration.get("top_sector_trc_pct"),
            "correlated_pair_count": len(correlated_pairs),
            "rebalance_needed": bool(rebalance_result.get("rebalance_needed")),
            "max_abs_drift": rebalance_result.get("max_abs_drift"),
            "optimizer_expected_sharpe": optimization_summary.get("expected_sharpe"),
        },
    }
