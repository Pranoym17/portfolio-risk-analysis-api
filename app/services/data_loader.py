import yfinance as yf

def get_price_history(tickers, period="1y"):
    data = yf.download(tickers, period=period, auto_adjust=False)["Adj Close"]
    return data.dropna()
