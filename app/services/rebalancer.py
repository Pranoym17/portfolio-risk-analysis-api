from __future__ import annotations


def aggregate_weights(weight_items) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in weight_items:
        ticker = (item.ticker or "").strip().upper()
        if not ticker:
            continue
        weights[ticker] = weights.get(ticker, 0.0) + float(item.weight)
    return weights


def build_rebalance_recommendation(
    *,
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    drift_threshold: float,
    portfolio_value: float | None = None,
    minimum_trade_value: float = 0.0,
) -> dict:
    tickers = sorted(set(current_weights) | set(target_weights))
    trades = []
    max_abs_drift = 0.0
    turnover_weight = 0.0

    for ticker in tickers:
        current = float(current_weights.get(ticker, 0.0))
        target = float(target_weights.get(ticker, 0.0))
        drift = target - current
        abs_drift = abs(drift)
        max_abs_drift = max(max_abs_drift, abs_drift)

        trade_value = None
        if portfolio_value is not None:
            trade_value = drift * portfolio_value

        action = "hold"
        if abs_drift > drift_threshold:
            action = "buy" if drift > 0 else "sell"

        if trade_value is not None and abs(trade_value) < minimum_trade_value:
            action = "hold"

        if action != "hold":
            turnover_weight += abs_drift

        trades.append(
            {
                "ticker": ticker,
                "current_weight": current,
                "target_weight": target,
                "drift": drift,
                "abs_drift": abs_drift,
                "action": action,
                "trade_value": trade_value,
            }
        )

    trades.sort(key=lambda item: item["abs_drift"], reverse=True)
    turnover_estimate = turnover_weight / 2.0
    rebalance_needed = any(item["action"] != "hold" for item in trades)

    if rebalance_needed:
        trade_count = sum(1 for item in trades if item["action"] != "hold")
        summary = (
            f"Rebalance recommended: {trade_count} position"
            f"{'' if trade_count == 1 else 's'} exceed the {drift_threshold:.0%} drift threshold."
        )
    else:
        summary = f"No rebalance needed: all positions are within the {drift_threshold:.0%} drift threshold."

    return {
        "rebalance_needed": rebalance_needed,
        "drift_threshold": drift_threshold,
        "max_abs_drift": max_abs_drift,
        "turnover_estimate": turnover_estimate,
        "portfolio_value": portfolio_value,
        "current_weights": current_weights,
        "target_weights": target_weights,
        "trades": trades,
        "summary": summary,
    }
