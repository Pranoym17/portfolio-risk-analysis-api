import numpy as np
import pandas as pd

from .analytics import compute_covariance_matrix, compute_expected_returns, compute_returns

_Z_MAP = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.97: 1.8808,
    0.99: 2.3263,
}


def _ensure_finite_scalar(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        raise ValueError(f"{field_name} could not be computed from available price history")
    return float(value)


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
    returns = compute_returns(price_df, return_type=return_type)
    if returns.empty or len(returns.index) < 2:
        raise ValueError("Not enough return history to compute portfolio metrics")

    mean_returns = compute_expected_returns(returns, trading_days=trading_days)
    cov = compute_covariance_matrix(returns, trading_days=trading_days)

    w = np.array(weights, dtype=float)

    port_return = _ensure_finite_scalar(float(np.dot(w, mean_returns.values)), "annual_return")
    variance = float(w.T @ cov.values @ w)
    volatility = _ensure_finite_scalar(float(np.sqrt(variance)), "volatility")
    sharpe = (port_return - risk_free) / volatility if volatility != 0 else None

    z = _Z_MAP.get(round(float(var_level), 2), _Z_MAP[0.95])
    var_parametric = port_return - z * volatility

    port_daily = pd.Series(returns.values @ w, index=returns.index)

    downside = port_daily[port_daily < 0]
    downside_std = downside.std() * np.sqrt(trading_days) if len(downside) > 0 else None
    sortino = None
    if downside_std and downside_std > 0:
        sortino = (port_return - risk_free) / downside_std

    worst_day = _ensure_finite_scalar(float(port_daily.min()), "worst_day")

    equity = (1.0 + port_daily).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    max_drawdown = _ensure_finite_scalar(float(drawdown.min()), "max_drawdown")

    beta = None
    if benchmark_prices is not None:
        bench_df = benchmark_prices.to_frame("BENCH")
        bench_ret = compute_returns(bench_df, return_type=return_type)["BENCH"]
        aligned = pd.concat([port_daily.rename("PORT"), bench_ret.rename("BENCH")], axis=1).dropna()
        if len(aligned) > 5 and aligned["BENCH"].var() != 0:
            beta = float(aligned["PORT"].cov(aligned["BENCH"]) / aligned["BENCH"].var())

    sharpe = _ensure_finite_scalar(sharpe, "sharpe_ratio")
    sortino = _ensure_finite_scalar(sortino, "sortino_ratio")
    beta = _ensure_finite_scalar(beta, "beta_vs_benchmark")
    var_parametric = _ensure_finite_scalar(var_parametric, "value_at_risk")

    covariance_matrix = {
        row: {
            col: float(value) if np.isfinite(value) else 0.0
            for col, value in cols.items()
        }
        for row, cols in cov.to_dict().items()
    }

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
        "covariance_matrix": covariance_matrix,
        "sortino_ratio": sortino,
    }


def compute_rolling_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    risk_free: float,
    window: int,
    trading_days: int,
):
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_returns.empty:
        return {"volatility": [], "sharpe": [], "beta": []}

    rolling_vol = clean_returns.rolling(window).std() * np.sqrt(trading_days)
    rolling_mean = clean_returns.rolling(window).mean() * trading_days
    rolling_sharpe = (rolling_mean - risk_free) / rolling_vol

    beta_series = None
    if benchmark_returns is not None:
        clean_benchmark = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()
        aligned = pd.concat([clean_returns.rename("PORT"), clean_benchmark.rename("BENCH")], axis=1).dropna()
        if not aligned.empty:
            cov = aligned["PORT"].rolling(window).cov(aligned["BENCH"])
            var = aligned["BENCH"].rolling(window).var()
            beta_series = cov / var

    def series_to_points(s: pd.Series):
        return [
            {
                "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                "value": float(val),
            }
            for idx, val in s.dropna().items()
            if np.isfinite(val)
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
    sectors: dict[str, str],
) -> dict:
    w = np.array(weights, dtype=float)
    if cov.empty or cov.shape[0] == 0:
        raise ValueError("Not enough return history to compute attribution")

    port_var = float(w.T @ cov.values @ w)
    port_vol = float(np.sqrt(port_var))
    if not np.isfinite(port_vol):
        raise ValueError("Portfolio volatility could not be computed from available price history")

    if port_vol == 0:
        mrc = np.zeros_like(w)
    else:
        sigma_w = cov.values @ w
        mrc = sigma_w / port_vol

    trc = w * mrc
    trc_pct = trc / port_vol if port_vol != 0 else np.zeros_like(w)

    attribution = []
    for i, ticker in enumerate(tickers):
        attribution.append(
            {
                "ticker": ticker,
                "weight": float(w[i]),
                "mrc": float(mrc[i]) if np.isfinite(mrc[i]) else 0.0,
                "trc": float(trc[i]) if np.isfinite(trc[i]) else 0.0,
                "trc_pct": float(trc_pct[i]) if np.isfinite(trc_pct[i]) else 0.0,
                "sector": sectors.get(ticker, "Unknown"),
            }
        )
    attribution.sort(key=lambda item: item["trc_pct"], reverse=True)

    sector_map: dict[str, dict] = {}
    for item in attribution:
        sector = sectors.get(item["ticker"], "Unknown")
        sector_map.setdefault(sector, {"trc": 0.0, "tickers": []})
        sector_map[sector]["trc"] += item["trc"]
        sector_map[sector]["tickers"].append(item["ticker"])

    sector_attribution = []
    for sector, data in sector_map.items():
        sector_attribution.append(
            {
                "sector": sector,
                "trc": data["trc"],
                "trc_pct": data["trc"] / port_vol if port_vol != 0 else 0.0,
                "tickers": data["tickers"],
            }
        )

    sector_attribution.sort(key=lambda item: item["trc_pct"], reverse=True)

    return {
        "portfolio_volatility": port_vol,
        "attribution": attribution,
        "sector_attribution": sector_attribution,
    }


def compute_concentration_summary(
    attribution: list[dict],
    sector_attribution: list[dict],
) -> dict:
    if not attribution:
        return {
            "top_asset_ticker": None,
            "top_asset_trc_pct": 0.0,
            "top_3_assets_trc_pct": 0.0,
            "top_sector": None,
            "top_sector_trc_pct": 0.0,
            "diversification_score": 0.0,
            "concentration_level": "unknown",
        }

    asset_contribs = [max(0.0, float(item["trc_pct"])) for item in attribution]
    top_asset = attribution[0]
    top_asset_trc_pct = asset_contribs[0] if asset_contribs else 0.0
    top_3_assets_trc_pct = float(sum(asset_contribs[:3]))
    top_sector = sector_attribution[0] if sector_attribution else None
    top_sector_name = top_sector["sector"] if top_sector else None
    top_sector_trc_pct = float(max(0.0, top_sector["trc_pct"])) if top_sector else 0.0

    hhi = float(sum(value * value for value in asset_contribs))
    diversification_score = max(0.0, 1.0 - hhi)

    if top_asset_trc_pct >= 0.45 or top_3_assets_trc_pct >= 0.85:
        concentration_level = "high"
    elif top_asset_trc_pct >= 0.30 or top_3_assets_trc_pct >= 0.65:
        concentration_level = "moderate"
    else:
        concentration_level = "low"

    return {
        "top_asset_ticker": top_asset["ticker"],
        "top_asset_trc_pct": top_asset_trc_pct,
        "top_3_assets_trc_pct": top_3_assets_trc_pct,
        "top_sector": top_sector_name,
        "top_sector_trc_pct": top_sector_trc_pct,
        "diversification_score": diversification_score,
        "concentration_level": concentration_level,
    }


def generate_concentration_insights(
    attribution: list[dict],
    sector_attribution: list[dict],
) -> list[dict]:
    if not attribution:
        return []

    insights: list[dict] = []
    concentration = compute_concentration_summary(attribution, sector_attribution)
    top_asset = attribution[0]

    if concentration["top_asset_trc_pct"] >= 0.35:
        insights.append(
            {
                "level": "warning" if concentration["top_asset_trc_pct"] < 0.5 else "critical",
                "code": "single_asset_concentration",
                "title": "A single holding dominates portfolio risk",
                "detail": (
                    f"{top_asset['ticker']} contributes {top_asset['trc_pct']:.0%} of total volatility, "
                    "which indicates concentrated risk exposure."
                ),
                "related_assets": [top_asset["ticker"]],
            }
        )

    if concentration["top_3_assets_trc_pct"] >= 0.7:
        insights.append(
            {
                "level": "warning",
                "code": "top_three_risk_cluster",
                "title": "Risk is concentrated in a small cluster of holdings",
                "detail": (
                    f"The top three holdings account for {concentration['top_3_assets_trc_pct']:.0%} "
                    "of total portfolio volatility."
                ),
                "related_assets": [item["ticker"] for item in attribution[:3]],
            }
        )

    if sector_attribution:
        top_sector = sector_attribution[0]
        if top_sector["trc_pct"] >= 0.45:
            insights.append(
                {
                    "level": "warning",
                    "code": "sector_concentration",
                    "title": "Sector exposure is a major risk driver",
                    "detail": (
                        f"{top_sector['sector']} contributes {top_sector['trc_pct']:.0%} of total volatility "
                        "through a concentrated set of holdings."
                    ),
                    "related_assets": top_sector["tickers"],
                }
            )

    if not insights:
        insights.append(
            {
                "level": "info",
                "code": "risk_balanced",
                "title": "Risk contributions appear relatively balanced",
                "detail": "No single holding or sector dominates portfolio volatility based on current inputs.",
                "related_assets": [],
            }
        )

    return insights


def generate_risk_summary(
    attribution: list[dict],
    sector_attribution: list[dict],
) -> str:
    if not attribution:
        return "No attribution data available."

    lines = []
    concentration = compute_concentration_summary(attribution, sector_attribution)
    top_assets = attribution[:3]
    asset_text = ", ".join(f"{asset['ticker']} ({asset['trc_pct']:.0%})" for asset in top_assets)
    lines.append(f"The portfolio's risk is primarily driven by {asset_text}.")

    if sector_attribution:
        top_sector = sector_attribution[0]
        lines.append(
            f"Sector exposure is concentrated in {top_sector['sector']}, "
            f"which contributes {top_sector['trc_pct']:.0%} of total volatility."
        )

    if len(attribution) > 1:
        if concentration["concentration_level"] == "low":
            lines.append(
                "Risk is reasonably diversified, with no single asset dominating portfolio volatility."
            )
        elif concentration["concentration_level"] == "moderate":
            lines.append(
                "Risk is moderately concentrated, so the largest contributors should be monitored closely."
            )
        else:
            lines.append("Risk is highly concentrated in a small number of assets.")

    return " ".join(lines)
