from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loader import get_price_history


WEIGHT_TOL = 1e-6


@dataclass
class AnalysisAssetDrop:
    ticker: str
    reason: str
    detail: str


@dataclass
class AnalysisUniverse:
    requested_tickers: list[str]
    usable_tickers: list[str]
    dropped_assets: list[AnalysisAssetDrop]
    ticker_to_weight: dict[str, float]
    prices: pd.DataFrame
    benchmark_ticker: str | None
    benchmark_prices: pd.Series | None


def normalize_ticker_weights(holdings) -> dict[str, float]:
    ticker_to_weight: dict[str, float] = {}
    for holding in holdings:
        ticker = (holding.ticker or "").strip().upper()
        if not ticker:
            continue
        ticker_to_weight[ticker] = ticker_to_weight.get(ticker, 0.0) + float(holding.weight)
    return ticker_to_weight


def ensure_weights_sum_to_one(ticker_to_weight: dict[str, float], *, context: str = "Weights") -> None:
    total = sum(ticker_to_weight.values())
    if abs(total - 1.0) > WEIGHT_TOL:
        raise ValueError(f"{context} must sum to 1.0 (got {total:.6f})")


def clean_prices(prices: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
    if prices is None:
        return pd.DataFrame()
    if hasattr(prices, "ndim") and prices.ndim == 1:
        prices = prices.to_frame()
    if not isinstance(prices, pd.DataFrame):
        return pd.DataFrame()
    cleaned = prices.copy()
    cleaned.columns = [str(col).strip().upper() for col in cleaned.columns]
    cleaned = cleaned.sort_index()
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.dropna(axis=1, how="all")
    return cleaned


def compute_returns(price_df: pd.DataFrame, return_type: str = "simple") -> pd.DataFrame:
    clean = clean_prices(price_df).ffill().dropna(how="all")
    if clean.empty:
        return pd.DataFrame()
    if return_type == "log":
        returns = np.log(clean / clean.shift(1))
    else:
        returns = clean.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=1, how="all")
    returns = returns.dropna(how="all")
    return returns


def compute_expected_returns(returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.mean() * trading_days


def compute_covariance_matrix(returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    return returns.cov() * trading_days


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    return returns.corr()


def filter_columns_by_return_history(
    returns: pd.DataFrame,
    *,
    minimum_points: int,
) -> tuple[list[str], list[AnalysisAssetDrop]]:
    usable: list[str] = []
    dropped: list[AnalysisAssetDrop] = []
    for column in returns.columns:
        count = int(returns[column].notna().sum())
        if count >= minimum_points:
            usable.append(column)
        else:
            dropped.append(
                AnalysisAssetDrop(
                    ticker=column,
                    reason="insufficient_return_history",
                    detail=f"Only {count} return observations were available after alignment",
                )
            )
    return usable, dropped


def build_analysis_universe(
    ticker_to_weight: dict[str, float],
    *,
    period: str,
    interval: str,
    minimum_price_rows: int = 11,
    benchmark: str | None = None,
) -> AnalysisUniverse:
    ensure_weights_sum_to_one(ticker_to_weight, context="Portfolio weights")

    requested_tickers = list(ticker_to_weight.keys())
    benchmark_ticker = (benchmark or "").strip().upper() or None
    all_tickers = requested_tickers + (
        [benchmark_ticker] if benchmark_ticker and benchmark_ticker not in requested_tickers else []
    )

    raw_prices = clean_prices(get_price_history(all_tickers, period=period, interval=interval))
    if raw_prices.empty:
        raise ValueError("No price data returned for tickers")

    if len(raw_prices.index) < minimum_price_rows:
        raise ValueError(
            f"Not enough price history returned to analyze this request (got {len(raw_prices.index)} rows)"
        )

    dropped_assets: list[AnalysisAssetDrop] = []
    usable_tickers: list[str] = []
    for ticker in requested_tickers:
        if ticker not in raw_prices.columns:
            dropped_assets.append(
                AnalysisAssetDrop(
                    ticker=ticker,
                    reason="missing_price_data",
                    detail="Ticker was not present in provider response",
                )
            )
            continue

        row_count = int(raw_prices[ticker].dropna().shape[0])
        if row_count < minimum_price_rows:
            dropped_assets.append(
                AnalysisAssetDrop(
                    ticker=ticker,
                    reason="insufficient_price_history",
                    detail=f"Only {row_count} non-empty price rows were returned",
                )
            )
            continue

        usable_tickers.append(ticker)

    if not usable_tickers:
        raise ValueError("No price data returned for tickers")

    usable_prices = raw_prices[usable_tickers]
    benchmark_prices = raw_prices[benchmark_ticker] if benchmark_ticker and benchmark_ticker in raw_prices.columns else None

    return AnalysisUniverse(
        requested_tickers=requested_tickers,
        usable_tickers=usable_tickers,
        dropped_assets=dropped_assets,
        ticker_to_weight=ticker_to_weight,
        prices=usable_prices,
        benchmark_ticker=benchmark_ticker,
        benchmark_prices=benchmark_prices,
    )
