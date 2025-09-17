import pandas as pd

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / down.ewm(alpha=1/n, adjust=False).mean()
    return 100 - (100/(1+rs))

def macd(close, fast=12, slow=26, signal=9):
    fast_ema, slow_ema = ema(close, fast), ema(close, slow)
    macd_val = fast_ema - slow_ema
    signal_line = ema(macd_val, signal)
    hist = macd_val - signal_line
    return macd_val, signal_line, hist

def atr(df, n=14):
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    macd_val, signal, hist = macd(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_val, signal, hist
    df["atr14"] = atr(df, 14)
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df = df.dropna().reset_index(drop=True)
    return df