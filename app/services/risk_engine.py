import numpy as np

def compute_portfolio_metrics(price_df, weights, risk_free=0.02, var_z=1.65):
    returns = price_df.pct_change().dropna()

    # Annualized stats 
    mean_returns = returns.mean() * 252
    cov = returns.cov() * 252

    w = np.array(weights, dtype=float)

    # Portfolio expected return (weighted)
    port_return = float(np.dot(w, mean_returns.values))

    # Portfolio volatility
    variance = float(w.T @ cov.values @ w)
    volatility = float(np.sqrt(variance))

    sharpe = (port_return - risk_free) / volatility if volatility != 0 else None

    # Parametric VaR 
    var_95 = port_return - var_z * volatility

    return {
        "annual_return": port_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "value_at_risk_95": var_95,
        "covariance_matrix": cov.to_dict()
    }
