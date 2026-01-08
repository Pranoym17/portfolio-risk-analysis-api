from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from .database import SessionLocal, engine
from . import models
from . import schemas
from .services.data_loader import get_price_history
from .services.risk_engine import compute_portfolio_metrics
import pandas as pd

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

    # Aggregate duplicates
    ticker_to_weight = {}
    for h in payload.holdings:
        ticker_to_weight[h.ticker] = ticker_to_weight.get(h.ticker, 0.0) + h.weight

    # Enforce sum to 1
    total = sum(ticker_to_weight.values())
    if abs(total - 1.0) > 1e-6:
        raise HTTPException(
            status_code=400,
            detail=f"Holdings weights must sum to 1.0 after aggregation (got {total:.6f})",
        )

    # Replace holdings atomically:
    # delete old -> add new -> commit

    db.query(models.Holding).filter(models.Holding.portfolio_id == portfolio_id).delete()

    for ticker, weight in ticker_to_weight.items():
        db.add(models.Holding(ticker=ticker, weight=float(weight), portfolio_id=portfolio_id))

    db.commit()
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

@app.get("/portfolios/{portfolio_id}/risk")
def portfolio_risk(
    portfolio_id: int,
    period: str = "1y",
    risk_free: float = 0.02,
    var_level: float = 0.95,
    db: Session = Depends(get_db),
):
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

    prices = get_price_history(tickers, period=period)
    if hasattr(prices, "ndim") and prices.ndim == 1:
         prices = prices.to_frame()

    returned_cols = [c for c in prices.columns if c in ticker_to_weight]
    dropped = [t for t in tickers if t not in returned_cols]

    if not returned_cols:
        raise HTTPException(status_code=400, detail="No price data returned for tickers")

    prices = prices[returned_cols]
    weights = [ticker_to_weight[c] for c in returned_cols]

    wsum = sum(weights)
    if abs(wsum - 1.0) > 1e-6:
        raise HTTPException(
            status_code=400,
            detail=f"Some tickers had no data, remaining weights sum to {wsum:.6f}. Fix holdings or tickers.",
        )

    metrics = compute_portfolio_metrics(prices, weights, risk_free=risk_free, var_level=var_level)
    return {
        "portfolio_id": portfolio_id,
        "period": period,
        "tickers_used": returned_cols,
        "tickers_dropped": dropped,
        "weights_used": dict(zip(returned_cols, weights)),
        "metrics": metrics,
    }
