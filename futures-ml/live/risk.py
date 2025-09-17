def position_size(equity_usdt, risk_pct, entry, sl, contract_value=1.0):
    risk_per_trade = equity_usdt * risk_pct
    stop_dist = abs(entry - sl)
    if stop_dist <= 0: return 0
    qty = risk_per_trade / stop_dist
    return max(0.0, qty / contract_value)

def dynamic_tp_sl(entry, atr, side, tp_k=1.5, sl_k=1.0):
    if side == "buy":
        tp = entry + tp_k*atr; sl = entry - sl_k*atr
    else:
        tp = entry - tp_k*atr; sl = entry + sl_k*atr
    return tp, sl