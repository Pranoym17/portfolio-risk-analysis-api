from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from .database import SessionLocal, engine
from . import models
from .services.data_loader import get_price_history
from .services.risk_engine import compute_portfolio_metrics

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


@app.get("/portfolio/{portfolio_id}/risk")
def portfolio_risk(portfolio_id: int, db: Session = Depends(get_db)):
    portfolio = (
        db.query(models.Portfolio)
        .filter(models.Portfolio.id == portfolio_id)
        .first()
    )
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    if not portfolio.holdings:
        raise HTTPException(status_code=400, detail="Portfolio has no holdings")

   #duplicate tickers sum weights
    ticker_to_weight = {}
    for h in portfolio.holdings:
        t = (h.ticker or "").strip().upper()
        if not t:
            continue
        ticker_to_weight[t] = ticker_to_weight.get(t, 0.0) + float(h.weight)

    tickers = list(ticker_to_weight.keys())
    if not tickers:
        raise HTTPException(status_code=400, detail="No valid tickers found in holdings")


    prices = get_price_history(tickers)

   
    if hasattr(prices, "to_frame") and prices.ndim == 1:
        prices = prices.to_frame(name=tickers[0])

    
    returned_cols = [c for c in prices.columns if c in ticker_to_weight]
    if not returned_cols:
        raise HTTPException(
            status_code=400,
            detail="No price data returned for portfolio tickers (check symbols / availability)",
        )

    prices = prices[returned_cols]
    weights = [ticker_to_weight[c] for c in returned_cols]


    total_w = sum(weights)
    if total_w <= 0:
        raise HTTPException(status_code=400, detail="Total weight must be > 0")
    weights = [w / total_w for w in weights]

    Compute metrics
    return compute_portfolio_metrics(prices, weights)
