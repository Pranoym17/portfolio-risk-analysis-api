import numpy as np

_Z_MAP = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.97: 1.8808,
    0.99: 2.3263,
}

def compute_portfolio_metrics(price_df, weights, risk_free=0.02, var_level=0.95):
    returns = price_df.pct_change().dropna()

    mean_returns = returns.mean() * 252
    cov = returns.cov() * 252

    w = np.array(weights, dtype=float)

    port_return = float(np.dot(w, mean_returns.values))
    variance = float(w.T @ cov.values @ w)
    volatility = float(np.sqrt(variance))

    sharpe = (port_return - risk_free) / volatility if volatility != 0 else None

    z = _Z_MAP.get(round(float(var_level), 2), _Z_MAP[0.95])
    var_parametric = port_return - z * volatility

    return {
        "annual_return": port_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "value_at_risk": var_parametric,
        "var_level": float(var_level),
        "covariance_matrix": cov.to_dict(),
    }
