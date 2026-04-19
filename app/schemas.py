from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Union

WEIGHT_TOL = 1e-6


class UserSignup(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=8, max_length=256)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        email = v.strip().lower()
        if "@" not in email or email.startswith("@") or email.endswith("@"):
            raise ValueError("Valid email required")
        return email


class UserLogin(UserSignup):
    pass


class UserOut(BaseModel):
    id: int
    email: str

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


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
    user_id: Optional[int] = None
    holdings: List[HoldingOut] = Field(default_factory=list)

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

class AnalysisAssetDrop(BaseModel):
    ticker: str
    reason: str
    detail: str

class RiskResponse(BaseModel):
    portfolio_id: int
    period: str
    tickers_used: List[str]
    tickers_dropped: List[str]
    tickers_dropped_details: List[AnalysisAssetDrop] = Field(default_factory=list)
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
    sector: str

class SectorRiskContribution(BaseModel):
    sector: str
    trc: float
    trc_pct: float
    tickers: List[str]

class RiskConcentrationInsight(BaseModel):
    level: str
    code: str
    title: str
    detail: str
    related_assets: List[str] = Field(default_factory=list)

class RiskConcentrationSummary(BaseModel):
    top_asset_ticker: Optional[str] = None
    top_asset_trc_pct: float
    top_3_assets_trc_pct: float
    top_sector: Optional[str] = None
    top_sector_trc_pct: float
    diversification_score: float
    concentration_level: str

class RiskAttributionResponse(BaseModel):
    portfolio_id: int
    period: str
    interval: str
    trading_days: int
    tickers_used: List[str]
    tickers_dropped: List[str]
    tickers_dropped_details: List[AnalysisAssetDrop] = Field(default_factory=list)
    portfolio_volatility: float
    concentration: RiskConcentrationSummary
    insights: List[RiskConcentrationInsight] = Field(default_factory=list)
    attribution: List[RiskAttributionItem]
    sector_attribution: List[SectorRiskContribution]
    summary: str
