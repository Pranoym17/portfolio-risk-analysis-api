from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
import pandas as pd
from .database import SessionLocal, engine, ensure_schema
from . import models
from . import schemas
from .auth import create_access_token, decode_access_token, hash_password, verify_password
from .services.data_loader import get_price_history
from .services.risk_engine import compute_portfolio_metrics, compute_rolling_metrics
from .services.risk_engine import compute_risk_attribution
from .services.data_loader import get_sector
from .services.risk_engine import generate_risk_summary
from fastapi.middleware.cors import CORSMiddleware

models.Base.metadata.create_all(bind=engine)
ensure_schema()

app = FastAPI(title="Portfolio Risk Analysis API")
bearer_scheme = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])
    except (ValueError, KeyError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


def get_owned_portfolio(db: Session, user_id: int, portfolio_id: int):
    portfolio = (
        db.query(models.Portfolio)
        .filter(
            models.Portfolio.id == portfolio_id,
            models.Portfolio.user_id == user_id,
        )
        .first()
    )
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio


def _ensure_sufficient_prices(prices: pd.DataFrame, minimum_rows: int = 11):
    if hasattr(prices, "ndim") and prices.ndim == 1:
        prices = prices.to_frame()

    if prices is None or prices.empty:
        raise HTTPException(status_code=400, detail="No price data returned for tickers")

    cleaned = prices.dropna(how="all")
    if len(cleaned.index) < minimum_rows:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough price history returned to analyze this request (got {len(cleaned.index)} rows)",
        )

    return cleaned


def _clean_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.ffill().pct_change(fill_method=None)
    returns = returns.replace([float("inf"), float("-inf")], pd.NA)
    returns = returns.dropna(axis=1, how="all")
    returns = returns.dropna(how="all")
    return returns


@app.get("/")
def root():
    return {"status": "ok", "message": "Portfolio Risk API Running"}


@app.post("/auth/signup", response_model=schemas.TokenResponse, status_code=201)
def signup(payload: schemas.UserSignup, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = models.User(
        email=payload.email,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "access_token": create_access_token(user.id, user.email),
        "user": user,
    }


@app.post("/auth/login", response_model=schemas.TokenResponse)
def login(payload: schemas.UserLogin, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "access_token": create_access_token(user.id, user.email),
        "user": user,
    }


@app.get("/auth/me", response_model=schemas.UserOut)
def me(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.post("/portfolios", response_model=schemas.PortfolioOut, status_code=201)
def create_portfolio(
    payload: schemas.PortfolioCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    portfolio = models.Portfolio(name=payload.name.strip(), user_id=current_user.id)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio


@app.put("/portfolios/{portfolio_id}/holdings", response_model=schemas.PortfolioOut)
def replace_holdings(
    portfolio_id: int,
    payload: schemas.HoldingsReplace,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    portfolio = get_owned_portfolio(db, current_user.id, portfolio_id)

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
def list_portfolios(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return (
        db.query(models.Portfolio)
        .filter(models.Portfolio.user_id == current_user.id)
        .all()
    )


@app.get("/portfolios/{portfolio_id}", response_model=schemas.PortfolioOut)
def get_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return get_owned_portfolio(db, current_user.id, portfolio_id)


@app.delete("/portfolios/{portfolio_id}", status_code=204)
def delete_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    portfolio = get_owned_portfolio(db, current_user.id, portfolio_id)
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
    current_user: models.User = Depends(get_current_user),
):
    return_type = return_type.lower()
    if return_type not in ("simple", "log"):
        raise HTTPException(status_code=400, detail="return_type must be 'simple' or 'log'")

    if not (0.5 < var_level < 1.0):
        raise HTTPException(status_code=400, detail="var_level must be between 0.5 and 1.0")

    if trading_days <= 0:
        raise HTTPException(status_code=400, detail="trading_days must be > 0")

    portfolio = get_owned_portfolio(db, current_user.id, portfolio_id)

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
    prices_all = _ensure_sufficient_prices(
        get_price_history(all_tickers, period=period, interval=interval)
    )

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

    try:
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    
    # daily returns for portfolio assets
    returns_df = _clean_returns(prices)
    if returns_df.empty or len(returns_df.index) < 2:
        raise HTTPException(status_code=400, detail="Not enough return history to compute rolling metrics")

    # portfolio daily returns series: (N_days x N_assets) @ (N_assets,)
    port_daily = pd.Series(returns_df.values @ pd.Series(weights).values, index=returns_df.index)

    # benchmark daily returns series
    benchmark_returns = None
    if benchmark_prices is not None:
        benchmark_returns = benchmark_prices.ffill().pct_change(fill_method=None).replace([float("inf"), float("-inf")], pd.NA).dropna()
        benchmark_returns = benchmark_returns.reindex(port_daily.index).dropna()
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

        prices = prices.dropna(how="all") if prices is not None else prices
        rows_returned = int(len(prices.index)) if prices is not None else 0
        available_columns = [str(col).strip().upper() for col in prices.columns] if prices is not None else []
        ok = (
            prices is not None
            and not prices.empty
            and t in available_columns
            and rows_returned > 10
        )
        return {
            "ticker": t,
            "period": period,
            "interval": interval,
            "is_valid": bool(ok),
            "rows_returned": rows_returned,
            "detail": (
                "Ticker returned enough price history for analysis"
                if ok else
                ("Ticker was not present in provider response" if t not in available_columns else f"Only {rows_returned} rows returned")
            ),
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


@app.get("/portfolios/{portfolio_id}/risk/attribution", response_model=schemas.RiskAttributionResponse)
def portfolio_risk_attribution(
    portfolio_id: int,
    period: str = "1y",
    interval: str = "1d",
    trading_days: int = 252,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    portfolio = get_owned_portfolio(db, current_user.id, portfolio_id)

    if not portfolio.holdings:
        raise HTTPException(status_code=400, detail="Portfolio has no holdings")

    # Build ticker -> weight
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

    # Fetch prices
    prices = _ensure_sufficient_prices(
        get_price_history(tickers, period=period, interval=interval)
    )

    returned_cols = [c for c in prices.columns if c in ticker_to_weight]
    dropped = [t for t in tickers if t not in returned_cols]

    if not returned_cols:
        raise HTTPException(status_code=400, detail="No price data returned for tickers")

    prices = prices[returned_cols]
    weights = [ticker_to_weight[c] for c in returned_cols]

    # Strict: if tickers dropped → remaining weights not 1
    wsum = sum(weights)
    if abs(wsum - 1.0) > 1e-6:
        raise HTTPException(
            status_code=400,
            detail=f"Some tickers had no data, remaining weights sum to {wsum:.6f}. Fix holdings or tickers.",
        )

    # Covariance matrix (annualized)
    returns = _clean_returns(prices)
    if returns.empty or len(returns.index) < 2:
        raise HTTPException(status_code=400, detail="Not enough return history to compute attribution")
    valid_columns = [col for col in returns.columns if returns[col].notna().sum() >= 2]
    if not valid_columns:
        raise HTTPException(status_code=400, detail="Not enough overlapping return history to compute attribution")
    if len(valid_columns) != len(returned_cols):
        filtered_weights = [ticker_to_weight[col] for col in valid_columns]
        filtered_weight_sum = sum(filtered_weights)
        if abs(filtered_weight_sum - 1.0) > 1e-6:
            raise HTTPException(
                status_code=400,
                detail=f"Some tickers lacked overlapping return history, remaining weights sum to {filtered_weight_sum:.6f}. Fix holdings or tickers.",
            )
        returned_cols = valid_columns
        prices = prices[returned_cols]
        weights = filtered_weights
        returns = returns[returned_cols]
    cov = returns.cov() * trading_days
    sectors = {t: get_sector(t) for t in returned_cols}
    try:
        attr = compute_risk_attribution(cov=cov, weights=weights, tickers=returned_cols,sectors=sectors)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    summary = generate_risk_summary(attribution=attr["attribution"],sector_attribution=attr["sector_attribution"])

    return {
        "portfolio_id": portfolio_id,
        "period": period,
        "interval": interval,
        "trading_days": trading_days,
        "tickers_used": returned_cols,
        "tickers_dropped": dropped,
        "portfolio_volatility": attr["portfolio_volatility"],
        "attribution": attr["attribution"],
        "sector_attribution": attr["sector_attribution"],
        "summary": summary,
    }
