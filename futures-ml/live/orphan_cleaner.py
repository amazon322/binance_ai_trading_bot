from loguru import logger

def cleanup_orphans(broker, symbol=None):
    positions = broker.list_open_positions()
    pos_symbols = set()
    for p in positions:
        try:
            sym = p.get('symbol') or p.get('info', {}).get('symbol')
            qty = float(p.get('contracts') or p.get('info', {}).get('positionAmt') or 0)
            if qty != 0:
                pos_symbols.add(sym)
        except Exception:
            continue
    open_orders = broker.list_open_orders(symbol)
    for o in open_orders:
        sym = o.get('symbol')
        if symbol and sym != symbol:
            continue
        if sym not in pos_symbols:
            try:
                broker.cancel_order(o['id'], sym)
                logger.info(f"Cancelled orphan order {o['id']} for {sym}")
            except Exception as e:
                logger.error(e)