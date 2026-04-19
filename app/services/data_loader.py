import os
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf


_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}
DEFAULT_TTL_SECONDS = 300  # 5 minutes
_SECTOR_CACHE: dict[str, str] = {}
_SESSION = requests.Session()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
FINNHUB_BASE_URL = os.getenv("FINNHUB_BASE_URL", "https://finnhub.io/api/v1").rstrip("/")


def _finnhub_enabled() -> bool:
    return bool(FINNHUB_API_KEY)


def _is_finnhub_symbol(symbol: str) -> bool:
    """
    Finnhub is strongest for standard exchange symbols and weaker for
    exchange-suffixed names like BN.TO. Keep those on fallback.
    """
    symbol = symbol.strip().upper()
    return bool(symbol) and "." not in symbol and "/" not in symbol

def _make_cache_key(tickers: Iterable[str], period: str, interval: str) -> tuple:
    tickers_sorted = tuple(sorted(set([t.strip().upper() for t in tickers if t])))
    return (tickers_sorted, period, interval)


def _normalize_prices(prices: pd.DataFrame | pd.Series | None, tickers: Iterable[str]) -> pd.DataFrame:
    if prices is None:
        return pd.DataFrame(columns=[t.strip().upper() for t in tickers if t])

    if hasattr(prices, "ndim") and prices.ndim == 1:
        ticker_list = [t.strip().upper() for t in tickers if t]
        prices = prices.to_frame(name=ticker_list[0] if ticker_list else "VALUE")

    if not isinstance(prices, pd.DataFrame):
        return pd.DataFrame()

    normalized = prices.copy()
    normalized.columns = [str(col).strip().upper() for col in normalized.columns]
    normalized = normalized.sort_index()
    normalized = normalized.dropna(how="all")
    normalized = normalized.dropna(axis=1, how="all")
    return normalized


def _merge_price_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    merged = pd.concat(non_empty, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.sort_index()
    merged = merged.dropna(how="all")
    return merged


def _period_to_start(period: str) -> datetime:
    now = datetime.now(timezone.utc)
    mapping = {
        "1mo": timedelta(days=31),
        "3mo": timedelta(days=93),
        "6mo": timedelta(days=186),
        "1y": timedelta(days=366),
        "2y": timedelta(days=730),
        "5y": timedelta(days=1826),
    }
    return now - mapping.get(period, timedelta(days=366))


def _interval_to_timeframe(interval: str) -> str | None:
    mapping = {
        "1d": "1Day",
        "1wk": "1Week",
    }
    return mapping.get(interval)


def _fetch_finnhub_prices(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    if not tickers or not _finnhub_enabled():
        return pd.DataFrame()

    resolution = {
        "1d": "D",
        "1wk": "W",
        "1mo": "M",
        "60m": "60",
        "30m": "30",
        "15m": "15",
        "5m": "5",
        "1m": "1",
    }.get(interval)
    if resolution is None:
        return pd.DataFrame()

    start = int(_period_to_start(period).timestamp())
    end = int(datetime.now(timezone.utc).timestamp())
    frames: list[pd.DataFrame] = []
    for symbol in tickers:
        response = _SESSION.get(
            f"{FINNHUB_BASE_URL}/stock/candle",
            params={
                "symbol": symbol,
                "resolution": resolution,
                "from": start,
                "to": end,
                "token": FINNHUB_API_KEY,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("s") != "ok":
            continue
        timestamps = payload.get("t", [])
        closes = payload.get("c", [])
        if not timestamps or not closes or len(timestamps) != len(closes):
            continue
        frame = pd.DataFrame(
            {
                "t": pd.to_datetime(timestamps, unit="s", utc=True),
                symbol.strip().upper(): closes,
            }
        ).set_index("t")
        frames.append(frame)

    return _merge_price_frames(frames)


def _fetch_yfinance_prices(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    data = yf.download(tickers, period=period, interval=interval, auto_adjust=False, progress=False)

    if (data is None or (isinstance(data, pd.DataFrame) and data.empty)) and len(tickers) == 1:
        single = tickers[0]
        try:
            fallback = yf.Ticker(single).history(period=period, interval=interval, auto_adjust=False)
            if isinstance(fallback, pd.DataFrame) and not fallback.empty:
                if "Adj Close" in fallback.columns:
                    data = fallback[["Adj Close"]].rename(columns={"Adj Close": single})
                elif "Close" in fallback.columns:
                    data = fallback[["Close"]].rename(columns={"Close": single})
        except Exception:
            pass

    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        prices = data

    return _normalize_prices(prices, tickers)


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

    all_tickers = list(key[0])
    finnhub_tickers = [ticker for ticker in all_tickers if _is_finnhub_symbol(ticker)]
    fallback_tickers = [ticker for ticker in all_tickers if ticker not in finnhub_tickers]

    finnhub_prices = pd.DataFrame()
    if finnhub_tickers:
        try:
            finnhub_prices = _fetch_finnhub_prices(finnhub_tickers, period, interval)
        except Exception:
            finnhub_prices = pd.DataFrame()

    missing_after_finnhub = [ticker for ticker in finnhub_tickers if ticker not in finnhub_prices.columns]
    yfinance_tickers = sorted(set(fallback_tickers + missing_after_finnhub))
    yfinance_prices = _fetch_yfinance_prices(yfinance_tickers, period, interval)

    prices = _merge_price_frames([finnhub_prices, yfinance_prices])

    _CACHE[key] = (now + ttl_seconds, prices.copy())
    return prices

def get_sector(ticker: str) -> str:
    """
    Returns sector name for a ticker using Finnhub first, with yfinance fallback.
    Cached to avoid repeated calls.
    """
    t = ticker.strip().upper()
    if t in _SECTOR_CACHE:
        return _SECTOR_CACHE[t]

    sector = "Unknown"
    if _finnhub_enabled() and _is_finnhub_symbol(t):
        try:
            response = _SESSION.get(
                f"{FINNHUB_BASE_URL}/stock/profile2",
                params={"symbol": t, "token": FINNHUB_API_KEY},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            sector = payload.get("finnhubIndustry") or payload.get("industry") or sector
        except Exception:
            sector = "Unknown"

    if sector == "Unknown":
        try:
            info = yf.Ticker(t).info
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"

    _SECTOR_CACHE[t] = sector
    return sector
