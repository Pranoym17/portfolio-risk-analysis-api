from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Union

WEIGHT_TOL = 1e-6


class HoldingIn(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    weight: float = Field(..., gt=0)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        t = v.strip().upper()
        if not t:
            raise ValueError("Ticker cannot be empty")
        return t


class HoldingOut(BaseModel):
    id: int
    ticker: str
    weight: float

    class Config:
        from_attributes = True


class PortfolioCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class PortfolioOut(BaseModel):
    id: int
    name: str
    holdings: List[HoldingOut] = []

    class Config:
        from_attributes = True


class HoldingsReplace(BaseModel):
    holdings: List[HoldingIn] = Field(..., min_length=1)

    @field_validator("holdings")
    @classmethod
    def validate_sum_to_one(cls, holdings: List[HoldingIn]) -> List[HoldingIn]:
        total = sum(h.weight for h in holdings)
        if abs(total - 1.0) > WEIGHT_TOL:
            raise ValueError(f"Holdings weights must sum to 1.0 (got {total:.6f})")
        return holdings


class RiskMetrics(BaseModel):
    annual_return: float
    volatility: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    value_at_risk: float
    var_level: float

    max_drawdown: float
    worst_day: float

    beta_vs_benchmark: Optional[float] = None
    benchmark_ticker: str

    covariance_matrix: Dict[str, Dict[str, float]]

class RollingMetricPoint(BaseModel):
    date: str
    value: float


class RollingMetrics(BaseModel):
    window: int
    volatility: List[RollingMetricPoint]
    sharpe: List[RollingMetricPoint]
    beta: List[RollingMetricPoint]

class RiskResponse(BaseModel):
    portfolio_id: int
    period: str
    tickers_used: List[str]
    tickers_dropped: List[str]
    weights_used: Dict[str, float]
    config: Dict[str, Union[float, str]]
    metrics: RiskMetrics
    rolling: Optional[RollingMetrics] = None

class RiskAttributionItem(BaseModel):
    ticker: str
    weight: float
    mrc: float                 # marginal risk contribution (to volatility)
    trc: float                 # total/component risk contribution (to volatility)
    trc_pct: float             # % contribution to total volatility


class RiskAttributionResponse(BaseModel):
    portfolio_id: int
    period: str
    interval: str
    trading_days: int
    tickers_used: List[str]
    tickers_dropped: List[str]
    portfolio_volatility: float
    attribution: List[RiskAttributionItem]