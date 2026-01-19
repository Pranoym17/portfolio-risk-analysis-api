from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from .database import SessionLocal, engine
from . import models
from . import schemas
from .services.data_loader import get_price_history
from .services.risk_engine import compute_portfolio_metrics, compute_rolling_metrics

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Portfolio Risk Analysis API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {"status": "ok", "message": "Portfolio Risk API Running"}


@app.post("/portfolios", response_model=schemas.PortfolioOut, status_code=201)
def create_portfolio(payload: schemas.PortfolioCreate, db: Session = Depends(get_db)):
    portfolio = models.Portfolio(name=payload.name.strip())
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio


@app.put("/portfolios/{portfolio_id}/holdings", response_model=schemas.PortfolioOut)
def replace_holdings(
    portfolio_id: int,
    payload: schemas.HoldingsReplace,
    db: Session = Depends(get_db),
):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    ticker_to_weight = {}
    for h in payload.holdings:
        ticker_to_weight[h.ticker] = ticker_to_weight.get(h.ticker, 0.0) + h.weight

    total = sum(ticker_to_weight.values())
    if abs(total - 1.0) > 1e-6:
        raise HTTPException(
            status_code=400,
            detail=f"Holdings weights must sum to 1.0 after aggregation (got {total:.6f})",
        )

    try:
        db.query(models.Holding).filter(models.Holding.portfolio_id == portfolio_id).delete()

        for ticker, weight in ticker_to_weight.items():
            db.add(models.Holding(ticker=ticker, weight=float(weight), portfolio_id=portfolio_id))

        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(portfolio)
    return portfolio


@app.get("/portfolios", response_model=list[schemas.PortfolioOut])
def list_portfolios(db: Session = Depends(get_db)):
    return db.query(models.Portfolio).all()


@app.get("/portfolios/{portfolio_id}", response_model=schemas.PortfolioOut)
def get_portfolio(portfolio_id: int, db: Session = Depends(get_db)):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio


@app.delete("/portfolios/{portfolio_id}", status_code=204)
def delete_portfolio(portfolio_id: int, db: Session = Depends(get_db)):
    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    db.delete(portfolio)
    db.commit()
    return


@app.get("/portfolios/{portfolio_id}/risk", response_model=schemas.RiskResponse)
def portfolio_risk(
    portfolio_id: int,
    period: str = "1y",
    interval: str = "1d",
    risk_free: float = 0.02,
    var_level: float = 0.95,
    trading_days: int = 252,
    return_type: str = "simple",
    benchmark: str = "SPY",
    rolling_window: int = 30,
    db: Session = Depends(get_db),
):
    return_type = return_type.lower()
    if return_type not in ("simple", "log"):
        raise HTTPException(status_code=400, detail="return_type must be 'simple' or 'log'")

    if not (0.5 < var_level < 1.0):
        raise HTTPException(status_code=400, detail="var_level must be between 0.5 and 1.0")

    if trading_days <= 0:
        raise HTTPException(status_code=400, detail="trading_days must be > 0")

    portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    if not portfolio.holdings:
        raise HTTPException(status_code=400, detail="Portfolio has no holdings")

    ticker_to_weight = {}
    for h in portfolio.holdings:
        t = (h.ticker or "").strip().upper()
        if not t:
            continue
        ticker_to_weight[t] = ticker_to_weight.get(t, 0.0) + float(h.weight)

    total = sum(ticker_to_weight.values())
    if abs(total - 1.0) > 1e-6:
        raise HTTPException(status_code=400, detail=f"DB weights do not sum to 1 (got {total:.6f})")

    tickers = list(ticker_to_weight.keys())
    bench = (benchmark or "SPY").strip().upper()

    all_tickers = tickers + ([bench] if bench not in tickers else [])
    prices_all = get_price_history(all_tickers, period=period, interval=interval)

    if hasattr(prices_all, "ndim") and prices_all.ndim == 1:
        prices_all = prices_all.to_frame()

    benchmark_prices = prices_all[bench] if bench in prices_all.columns else None

    returned_cols = [c for c in prices_all.columns if c in ticker_to_weight]
    dropped = [t for t in tickers if t not in returned_cols]

    if not returned_cols:
        raise HTTPException(status_code=400, detail="No price data returned for tickers")

    prices = prices_all[returned_cols]
    weights = [ticker_to_weight[c] for c in returned_cols]

    wsum = sum(weights)
    if abs(wsum - 1.0) > 1e-6:
        raise HTTPException(
            status_code=400,
            detail=f"Some tickers had no data, remaining weights sum to {wsum:.6f}. Fix holdings or tickers.",
        )

    metrics_dict = compute_portfolio_metrics(
        price_df=prices,
        weights=weights,
        benchmark_prices=benchmark_prices,
        risk_free=risk_free,
        var_level=var_level,
        trading_days=trading_days,
        return_type=return_type,
        benchmark_ticker=bench,
    )
    
    # daily returns for portfolio assets
    returns_df = prices.pct_change().dropna()

    # portfolio daily returns series: (N_days x N_assets) @ (N_assets,)
    port_daily = pd.Series(returns_df.values @ pd.Series(weights).values, index=returns_df.index)

    # benchmark daily returns series
    benchmark_returns = None
    if benchmark_prices is not None:
        benchmark_returns = benchmark_prices.pct_change().dropna()

    rolling_dict = compute_rolling_metrics(
         returns=port_daily,
         benchmark_returns=benchmark_returns,
         risk_free=risk_free,
          window=rolling_window,
         trading_days=trading_days,
     )

    return {
        "portfolio_id": portfolio_id,
        "period": period,
        "tickers_used": returned_cols,
        "tickers_dropped": dropped,
        "weights_used": dict(zip(returned_cols, weights)),
        "config": {
            "period": period,
            "interval": interval,
            "risk_free": risk_free,
            "var_level": var_level,
            "trading_days": trading_days,
            "return_type": return_type,
            "benchmark": bench,
        },
        "metrics": metrics_dict,
        "rolling": {
        "window": rolling_window,
        **rolling_dict,
    },
    }


@app.get("/tickers/validate")
def validate_ticker(ticker: str, period: str = "1y", interval: str = "1d"):
    t = (ticker or "").strip().upper()
    if not t:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")

    try:
        prices = get_price_history([t], period=period, interval=interval)

        if hasattr(prices, "ndim") and prices.ndim == 1:
            prices = prices.to_frame()

        ok = prices is not None and len(prices.index) > 10
        return {
            "ticker": t,
            "period": period,
            "interval": interval,
            "is_valid": bool(ok),
            "rows_returned": int(len(prices.index)) if prices is not None else 0,
        }
    except Exception as e:
        return {
            "ticker": t,
            "period": period,
            "interval": interval,
            "is_valid": False,
            "rows_returned": 0,
            "error": str(e),
        }
