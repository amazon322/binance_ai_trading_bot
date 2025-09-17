import ccxt
import pandas as pd

def fetch_ohlcv_binance(symbol: str, timeframe: str = "5m", limit: int = 2000, testnet: bool = False):
    ex = ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"}
    })
    if testnet:
        ex.set_sandbox_mode(True)
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df, ex