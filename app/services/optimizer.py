from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analytics import (
    AnalysisAssetDrop,
    compute_correlation_matrix,
    compute_covariance_matrix,
    compute_expected_returns,
)


@dataclass
class OptimizationInputs:
    tickers: list[str]
    expected_returns: pd.Series
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    standalone_volatility: pd.Series
    standalone_sharpe: pd.Series
    dropped_assets: list[AnalysisAssetDrop]


def _require_scipy():
    try:
        from scipy.optimize import minimize
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optimization requires scipy. Install it with `pip install scipy` or add it to the environment."
        ) from exc
    return minimize


def prepare_optimization_inputs(
    returns: pd.DataFrame,
    *,
    risk_free: float,
    trading_days: int,
    filter_high_correlation: bool,
    correlation_threshold: float,
    filter_low_sharpe: bool,
    min_asset_sharpe: float,
) -> OptimizationInputs:
    if returns.empty or len(returns.columns) < 2:
        raise ValueError("At least two assets with usable return history are required for optimization")

    expected_returns = compute_expected_returns(returns, trading_days=trading_days)
    covariance = compute_covariance_matrix(returns, trading_days=trading_days)
    correlation = compute_correlation_matrix(returns)
    standalone_volatility = pd.Series(np.sqrt(np.diag(covariance.values)), index=covariance.index)
    standalone_sharpe = (expected_returns - risk_free) / standalone_volatility.replace(0.0, np.nan)

    selected = list(returns.columns)
    dropped_assets: list[AnalysisAssetDrop] = []

    if filter_low_sharpe:
        kept = []
        for ticker in selected:
            sharpe = standalone_sharpe.get(ticker)
            if pd.notna(sharpe) and float(sharpe) >= float(min_asset_sharpe):
                kept.append(ticker)
            else:
                detail = (
                    f"Standalone Sharpe {float(sharpe):.3f} was below the threshold {min_asset_sharpe:.3f}"
                    if pd.notna(sharpe)
                    else "Standalone Sharpe could not be computed"
                )
                dropped_assets.append(
                    AnalysisAssetDrop(
                        ticker=ticker,
                        reason="low_standalone_sharpe",
                        detail=detail,
                    )
                )
        selected = kept

    if len(selected) < 2:
        raise ValueError("Too few assets remain after Sharpe filtering to run optimization")

    if filter_high_correlation:
        survivors: list[str] = []
        ordered = sorted(selected, key=lambda ticker: float(standalone_sharpe.get(ticker, -np.inf)), reverse=True)
        dropped_corr: set[str] = set()
        for ticker in ordered:
            if ticker in dropped_corr:
                continue
            survivors.append(ticker)
            for other in ordered:
                if other == ticker or other in dropped_corr or other in survivors:
                    continue
                corr_value = correlation.loc[ticker, other]
                if pd.notna(corr_value) and abs(float(corr_value)) > correlation_threshold:
                    dropped_corr.add(other)
                    dropped_assets.append(
                        AnalysisAssetDrop(
                            ticker=other,
                            reason="high_correlation",
                            detail=(
                                f"Correlation with {ticker} was {float(corr_value):.3f}, "
                                f"above the threshold {correlation_threshold:.3f}"
                            ),
                        )
                    )
        selected = survivors

    if len(selected) < 2:
        raise ValueError("Too few assets remain after correlation filtering to run optimization")

    return OptimizationInputs(
        tickers=selected,
        expected_returns=expected_returns[selected],
        covariance=covariance.loc[selected, selected],
        correlation=correlation.loc[selected, selected],
        standalone_volatility=standalone_volatility[selected],
        standalone_sharpe=standalone_sharpe[selected],
        dropped_assets=dropped_assets,
    )


def optimize_portfolio(
    optimization_inputs: OptimizationInputs,
    *,
    objective: str,
    risk_free: float,
    max_weight: float,
) -> dict:
    minimize = _require_scipy()
    tickers = optimization_inputs.tickers
    mu = optimization_inputs.expected_returns.values.astype(float)
    cov = optimization_inputs.covariance.values.astype(float)
    n_assets = len(tickers)

    if n_assets < 2:
        raise ValueError("At least two assets are required for optimization")
    if max_weight * n_assets < 1.0:
        raise ValueError("max_weight is too small to allocate 100% across the available assets")

    def portfolio_volatility(weights: np.ndarray) -> float:
        variance = float(weights.T @ cov @ weights)
        return float(np.sqrt(max(variance, 0.0)))

    def objective_fn(weights: np.ndarray) -> float:
        port_return = float(weights @ mu)
        port_vol = portfolio_volatility(weights)
        if objective == "min_variance":
            return port_vol
        if port_vol <= 0:
            return 1e9
        sharpe = (port_return - risk_free) / port_vol
        return -float(sharpe)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight) for _ in range(n_assets)]
    initial = np.full(n_assets, 1.0 / n_assets)

    result = minimize(
        objective_fn,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    weights = np.clip(result.x, 0.0, max_weight)
    weights = weights / weights.sum()

    port_return = float(weights @ mu)
    port_vol = portfolio_volatility(weights)
    port_sharpe = None if port_vol == 0 else float((port_return - risk_free) / port_vol)

    covariance_matrix = {
        row: {
            col: float(value) if np.isfinite(value) else 0.0
            for col, value in cols.items()
        }
        for row, cols in optimization_inputs.covariance.to_dict().items()
    }

    optimal_weights = []
    for idx, ticker in enumerate(tickers):
        optimal_weights.append(
            {
                "ticker": ticker,
                "weight": float(weights[idx]),
                "expected_return": float(optimization_inputs.expected_returns[ticker]),
                "volatility": float(optimization_inputs.standalone_volatility[ticker]),
                "standalone_sharpe": (
                    float(optimization_inputs.standalone_sharpe[ticker])
                    if pd.notna(optimization_inputs.standalone_sharpe[ticker])
                    else None
                ),
            }
        )

    optimal_weights.sort(key=lambda item: item["weight"], reverse=True)

    return {
        "tickers_selected": tickers,
        "optimal_weights": optimal_weights,
        "summary": {
            "objective": objective,
            "expected_return": port_return,
            "expected_volatility": port_vol,
            "expected_sharpe": port_sharpe,
            "max_weight_used": max_weight,
            "selected_asset_count": len(tickers),
        },
        "covariance_matrix": covariance_matrix,
    }
