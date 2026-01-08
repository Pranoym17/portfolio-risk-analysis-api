from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from .database import SessionLocal, engine
from . import models
from . import schemas

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

    # Aggregate duplicates (AAPL appears twice -> sum weights)
    ticker_to_weight = {}
    for h in payload.holdings:
        ticker_to_weight[h.ticker] = ticker_to_weight.get(h.ticker, 0.0) + h.weight

    # Enforce sum-to-1 again after aggregation (duplicates could break it)
    total = sum(ticker_to_weight.values())
    if abs(total - 1.0) > 1e-6:
        raise HTTPException(
            status_code=400,
            detail=f"Holdings weights must sum to 1.0 after aggregation (got {total:.6f})",
        )

    # Replace holdings atomically:
    # delete old -> add new -> commit
    # (relationship has cascade delete-orphan, but we'll do explicit delete for clarity)
    db.query(models.Holding).filter(models.Holding.portfolio_id == portfolio_id).delete()

    for ticker, weight in ticker_to_weight.items():
        db.add(models.Holding(ticker=ticker, weight=float(weight), portfolio_id=portfolio_id))

    db.commit()
    db.refresh(portfolio)
    return portfolio
