import yfinance as yf
import pandas as pd

def get_price_history(tickers, period="1y"):
    data = yf.download(tickers, period=period, auto_adjust=False)

    # yfinance returns different shapes depending on tickers count
    # We want a DataFrame of Adj Close columns per ticker
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        # fallback (rare formatting cases)
        prices = data

    # If single ticker, it might come back as a Series
    if hasattr(prices, "to_frame") and getattr(prices, "ndim", 2) == 1:
        prices = prices.to_frame()

    return prices.dropna()
