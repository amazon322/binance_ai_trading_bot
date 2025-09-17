import ccxt
from loguru import logger

class BinanceUSDM:
    def __init__(self, key, secret, testnet=False):
        self.ex = ccxt.binanceusdm({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })
        if testnet:
            self.ex.set_sandbox_mode(True)
        self.ex.load_markets()

    def norm_symbol(self, symbol: str) -> str:
        return symbol  # Binance USDM: 'BTC/USDT'

    def round_to_market(self, symbol, price=None, qty=None):
        s = self.norm_symbol(symbol)
        if price is not None:
            price = float(self.ex.price_to_precision(s, price))
        if qty is not None:
            qty = float(self.ex.amount_to_precision(s, qty))
        return price, qty

    def fetch_mark_price(self, symbol):
        t = self.ex.fetch_ticker(self.norm_symbol(symbol))
        return float(t.get("info", {}).get("markPrice") or t["last"])

    def fetch_l2(self, symbol, limit=100):
        ob = self.ex.fetch_order_book(self.norm_symbol(symbol), limit=limit)
        return ob.get("bids", []), ob.get("asks", [])

    def spread_pct(self, symbol):
        t = self.ex.fetch_ticker(self.norm_symbol(symbol))
        bid, ask = t.get("bid"), t.get("ask")
        if not bid or not ask:
            return None
        mid = 0.5 * (bid + ask)
        return (ask - bid) / (mid + 1e-12)

    def place_order(self, symbol, side, qty, tp=None, sl=None, reduce_only=False):
        s = self.norm_symbol(symbol)
        _, qty = self.round_to_market(s, qty=qty)
        od = self.ex.create_order(s, type="market", side=side, amount=qty, params={"reduceOnly": reduce_only})
        try:
            if tp:
                tp,_ = self.round_to_market(s, price=tp)
                self.ex.create_order(s, type="TAKE_PROFIT_MARKET", side=("sell" if side=="buy" else "buy"), amount=qty, params={"stopPrice": tp, "reduceOnly": True})
            if sl:
                sl,_ = self.round_to_market(s, price=sl)
                self.ex.create_order(s, type="STOP_MARKET", side=("sell" if side=="buy" else "buy"), amount=qty, params={"stopPrice": sl, "reduceOnly": True})
        except Exception as e:
            logger.error(f"TP/SL attach error: {e}")
        return od

    def list_open_positions(self):
        try:
            return self.ex.fetch_positions()
        except Exception as e:
            logger.error(e); return []

    def list_open_orders(self, symbol=None):
        try:
            return self.ex.fetch_open_orders(self.norm_symbol(symbol) if symbol else None)
        except Exception as e:
            logger.error(e); return []

    def cancel_order(self, order_id, symbol):
        try:
            return self.ex.cancel_order(order_id, self.norm_symbol(symbol))
        except Exception as e:
            logger.error(e); return None