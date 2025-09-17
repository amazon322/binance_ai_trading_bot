import numpy as np
import pandas as pd

def triple_barrier(df: pd.DataFrame, horizon:int=24, tp_atr:float=1.5, sl_atr:float=1.0):
    y_class = np.zeros(len(df), dtype=int)
    y_reg = np.zeros(len(df), dtype=float)
    for i in range(len(df)-horizon-1):
        price = df.loc[i, "close"]
        atr = df.loc[i, "atr14"]
        tp = price * (1 + tp_atr * atr / price)
        sl = price * (1 - sl_atr * atr / price)
        segment = df.loc[i+1:i+horizon, "close"]
        hit_tp = (segment >= tp).any()
        hit_sl = (segment <= sl).any()
        if hit_tp and not hit_sl:
            y_class[i] = 1
            hit_idx = segment[segment >= tp].index[0]
            y_reg[i] = (df.loc[hit_idx, "close"] - price) / price
        elif hit_sl and not hit_tp:
            y_class[i] = -1
            hit_idx = segment[segment <= sl].index[0]
            y_reg[i] = (df.loc[hit_idx, "close"] - price) / price
        else:
            y_class[i] = 0
            y_reg[i] = (df.loc[i+horizon, "close"] - price) / price
    return y_class, y_reg