import os, asyncio, time
import numpy as np
import torch
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

from data.fetch_ccxt import fetch_ohlcv_binance
from data.features import build_features
from model.hybrid_model import HybridModel
from live.exchanges import BinanceUSDM
from live.gate import price_gate, liquidity_guard, depth_guard
from live.risk import dynamic_tp_sl, position_size
from live.orphan_cleaner import cleanup_orphans
from live.telegram import TGBot

logger.add("bot.log", rotation="5 MB", retention=7)

TIMEFRAME = os.getenv("TIMEFRAME", "5m")
LOOKBACK = int(os.getenv("LOOKBACK", 128))
RISK_PCT = float(os.getenv("RISK_PCT", 0.005))
SPREAD_GUARD_PCT = float(os.getenv("SPREAD_GUARD_PCT", 0.25))
PRICE_GATE_PCT = float(os.getenv("PRICE_GATE_PCT", 0.3))
DEPTH_WINDOW_PCT = float(os.getenv("DEPTH_WINDOW_PCT", 0.3))
MAX_OPEN = int(os.getenv("MAX_OPEN_TRADES", 10))
DEDUP_HOURS = int(os.getenv("DEDUP_HOURS", 24))
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(',')]
TESTNET = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

FEATURES = ["open","high","low","close","volume","ema50","ema200","rsi14","macd","macd_signal","macd_hist","atr14","ret1","ret5"]

dedup = {}  # symbol -> last_ts

async def main():
    tg = TGBot()
    broker = BinanceUSDM(key=os.getenv("BINANCE_KEY"), secret=os.getenv("BINANCE_SECRET"), testnet=TESTNET)
    model = HybridModel(in_features=len(FEATURES))
    model.eval()  # TODO: зареди тежести ако имаш: model.load_state_dict(torch.load(...))
    equity_usdt = 1000.0

    while True:
        try:
            cleanup_orphans(broker)
            open_pos = broker.list_open_positions()
            active = [p for p in open_pos if float(p.get('contracts') or p.get('info',{}).get('positionAmt') or 0) != 0]
            if len(active) >= MAX_OPEN:
                logger.info(f"Max open positions reached: {len(active)}/{MAX_OPEN}")
                await asyncio.sleep(30); continue

            for symbol in SYMBOLS:
                last_ts = dedup.get(symbol, 0)
                if time.time() - last_ts < DEDUP_HOURS*3600:
                    continue

                df, ex = fetch_ohlcv_binance(symbol, timeframe=TIMEFRAME, limit=LOOKBACK+300, testnet=TESTNET)
                df = build_features(df)
                if len(df) < LOOKBACK + 5:
                    continue
                X = torch.tensor(df[FEATURES].values[-LOOKBACK:], dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    logits, yreg = model(X)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    side = "flat"; conf = probs[1]
                    if probs[2] >= 0.45 and yreg.item() > 0.001: side, conf = "buy", probs[2]
                    if probs[0] >= 0.45 and yreg.item() < -0.001: side, conf = "sell", probs[0]

                entry = float(df.iloc[-1]["close"])  # референтна
                atr = float(df.iloc[-1]["atr14"])    # за TP/SL

                spread = broker.spread_pct(symbol)
                if not liquidity_guard(spread, SPREAD_GUARD_PCT):
                    logger.info(f"{symbol} NO_TRADE: spread {spread*100:.2f}% > {SPREAD_GUARD_PCT}%"); continue

                mark = float(broker.fetch_mark_price(symbol))
                status, diff = price_gate(mark, entry, PRICE_GATE_PCT)
                if status != "OK":
                    logger.info(f"{symbol} NO_TRADE: price gate diff={diff:.2f}%"); continue

                bids, asks = broker.fetch_l2(symbol, limit=100)
                mid = 0.5 * (bids[0][0] + asks[0][0]) if bids and asks else None
                if not depth_guard(bids, asks, mid, window_pct=DEPTH_WINDOW_PCT):
                    logger.info(f"{symbol} NO_TRADE: insufficient L2 depth"); continue

                if side == "flat":
                    logger.info(f"{symbol} WAIT conf={conf:.2f}")
                    continue

                tp, sl = dynamic_tp_sl(entry=entry, atr=atr, side=("buy" if side=="buy" else "sell"))
                qty = position_size(equity_usdt, RISK_PCT, entry, sl)
                if qty <= 0:
                    logger.info(f"{symbol} NO_TRADE: qty=0"); continue

                od = broker.place_order(symbol, side, qty, tp=tp, sl=sl)
                dedup[symbol] = time.time()
                msg = (f"✅ {symbol} {side.upper()}\n"
                       f"Entry≈{entry:.4f} | TP={tp:.4f} | SL={sl:.4f}\n"
                       f"ATR={atr:.6f} | conf={conf:.2f} | qty={qty:.4f}\n"
                       f"Gate: spread={spread*100:.2f}% diff={diff:.2f}%")
                logger.info(msg)
                await tg.send(msg)

            await asyncio.sleep(30)
        except Exception as e:
            logger.exception(e); await asyncio.sleep(10)

if __name__ == "__main__":
    import uvloop
    uvloop.install()
    asyncio.run(main())