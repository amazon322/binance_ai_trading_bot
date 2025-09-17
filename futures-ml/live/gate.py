from loguru import logger

def price_gate(mark_price: float, ref_price: float, threshold_pct: float = 0.3):
    avg = 0.5 * (mark_price + ref_price)
    diff = abs(mark_price - ref_price) / (avg + 1e-12) * 100
    return ("OK" if diff <= threshold_pct else "ALERT"), diff

def liquidity_guard(spread_pct: float, max_spread_pct: float = 0.25):
    return spread_pct is not None and spread_pct * 100 <= max_spread_pct

def depth_guard(bids, asks, mid, window_pct=0.3):
    if not bids or not asks or mid is None:
        return False
    lo, hi = mid * (1 - window_pct/100), mid * (1 + window_pct/100)
    b_vol = sum(v for p, v in bids if lo <= p <= mid)
    a_vol = sum(v for p, v in asks if mid <= p <= hi)
    ok = b_vol > 0 and a_vol > 0
    if not ok:
        logger.info(f"Depth fail: b_vol={b_vol:.4f} a_vol={a_vol:.4f}")
    return ok