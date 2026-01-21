import time
import yfinance as yf
import pandas as pd
from typing import Iterable


_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}
DEFAULT_TTL_SECONDS = 300  # 5 minutes


def _make_cache_key(tickers: Iterable[str], period: str, interval: str) -> tuple:
    tickers_sorted = tuple(sorted(set([t.strip().upper() for t in tickers if t])))
    return (tickers_sorted, period, interval)


def get_price_history(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> pd.DataFrame:
    """
    Returns a DataFrame of Adj Close prices with columns = tickers.
    Uses a TTL cache to avoid repeated yfinance calls.
    """
    key = _make_cache_key(tickers, period, interval)
    now = time.time()

    if key in _CACHE:
        expires_at, df = _CACHE[key]
        if now < expires_at:
            return df.copy()

    data = yf.download(list(key[0]), period=period, interval=interval, auto_adjust=False)

    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        prices = data


    if hasattr(prices, "ndim") and prices.ndim == 1:
        prices = prices.to_frame(name=list(key[0])[0])

    prices = prices.dropna()

    _CACHE[key] = (now + ttl_seconds, prices.copy())
    return prices
