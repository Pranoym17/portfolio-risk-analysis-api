import numpy as np
import pandas as pd

_Z_MAP = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.97: 1.8808,
    0.99: 2.3263,
}

def _to_returns(price_df: pd.DataFrame, return_type: str) -> pd.DataFrame:
    if return_type == "log":
        return np.log(price_df / price_df.shift(1)).dropna()
    return price_df.pct_change().dropna()


def compute_portfolio_metrics(
    price_df: pd.DataFrame,
    weights: list[float],
    benchmark_prices: pd.Series | None = None,
    risk_free: float = 0.02,
    var_level: float = 0.95,
    trading_days: int = 252,
    return_type: str = "simple",
    benchmark_ticker: str = "SPY",
) -> dict:
    """
    Computes portfolio risk metrics.
    Assumes weights are aligned to price_df.columns and sum to 1.
    """
    returns = _to_returns(price_df, return_type=return_type)

    # Annualized mean + cov
    mean_returns = returns.mean() * trading_days
    cov = returns.cov() * trading_days

    w = np.array(weights, dtype=float)

    port_return = float(np.dot(w, mean_returns.values))
    variance = float(w.T @ cov.values @ w)
    volatility = float(np.sqrt(variance))
    sharpe = (port_return - risk_free) / volatility if volatility != 0 else None

    z = _Z_MAP.get(round(float(var_level), 2), _Z_MAP[0.95])
    var_parametric = port_return - z * volatility

    # Daily portfolio return series (for drawdown / worst day)
    port_daily = (returns.values @ w)
    port_daily = pd.Series(port_daily, index=returns.index)
    
    downside = port_daily[port_daily < 0]
    downside_std = downside.std() * np.sqrt(trading_days) if len(downside) > 0 else None
    sortino = None
    if downside_std and downside_std > 0:
         sortino = (port_return - risk_free) / downside_std
    
    worst_day = float(port_daily.min())

    # Max drawdown from cumulative equity curve
    equity = (1.0 + port_daily).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    max_drawdown = float(drawdown.min())

    beta = None
    if benchmark_prices is not None:
        bench_df = benchmark_prices.to_frame("BENCH")
        bench_ret = _to_returns(bench_df, return_type=return_type)["BENCH"]

        aligned = pd.concat([port_daily.rename("PORT"), bench_ret.rename("BENCH")], axis=1).dropna()
        if len(aligned) > 5 and aligned["BENCH"].var() != 0:
            beta = float(aligned["PORT"].cov(aligned["BENCH"]) / aligned["BENCH"].var())

    return {
        "annual_return": port_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "value_at_risk": var_parametric,
        "var_level": float(var_level),
        "max_drawdown": max_drawdown,
        "worst_day": worst_day,
        "beta_vs_benchmark": beta,
        "benchmark_ticker": benchmark_ticker,
        "covariance_matrix": cov.to_dict(),
        "sortino_ratio": sortino,
        "rolling_window": rolling_window,
    }

def compute_rolling_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    risk_free: float,
    window: int,
    trading_days: int,
):
    rolling_vol = returns.rolling(window).std() * np.sqrt(trading_days)
    rolling_mean = returns.rolling(window).mean() * trading_days
    rolling_sharpe = (rolling_mean - risk_free) / rolling_vol

    beta_series = None
    if benchmark_returns is not None:
        cov = returns.rolling(window).cov(benchmark_returns)
        var = benchmark_returns.rolling(window).var()
        beta_series = cov / var

    def series_to_points(s: pd.Series):
        return [
            {"date": idx.strftime("%Y-%m-%d"), "value": float(val)}
            for idx, val in s.dropna().items()
        ]

    return {
        "volatility": series_to_points(rolling_vol),
        "sharpe": series_to_points(rolling_sharpe),
        "beta": series_to_points(beta_series) if beta_series is not None else [],
    }

def compute_risk_attribution(
    cov: pd.DataFrame,
    weights: list[float],
    tickers: list[str],
) -> dict:

    w = np.array(weights, dtype=float)

    # portfolio variance and vol
    port_var = float(w.T @ cov.values @ w)
    port_vol = float(np.sqrt(port_var))
    if port_vol == 0:
        # Edge case: zero vol portfolio (rare). Avoid divide-by-zero.
        mrc = np.zeros_like(w)
    else:
        sigma_w = cov.values @ w
        mrc = sigma_w / port_vol

    trc = w * mrc
    trc_pct = trc / port_vol if port_vol != 0 else np.zeros_like(w)

    attribution = []
    for i, t in enumerate(tickers):
        attribution.append({
            "ticker": t,
            "weight": float(w[i]),
            "mrc": float(mrc[i]),
            "trc": float(trc[i]),
            "trc_pct": float(trc_pct[i]),
        })

    # Sort by biggest risk contributor 
    attribution.sort(key=lambda x: x["trc_pct"], reverse=True)

    return {
        "portfolio_volatility": port_vol,
        "attribution": attribution,
    }