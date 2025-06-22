# src/data_loader.py
# Download subset of Historical OHLCV Data
import yfinance as yf
import pandas as pd

def download_data(tickers, start="2015-01-01", end="2024-12-31"):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    return data

tickers = ["SPY", "QQQ", "GLD", "TLT", "AAPL"]
data = download_data(tickers)
data.to_csv("data/ohlcv_data.csv")