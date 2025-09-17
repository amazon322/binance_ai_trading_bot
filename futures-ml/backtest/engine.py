class Backtester:
    def __init__(self, fee_pct=0.05/100, slip_pct=0.05/100):
        self.fee_pct = fee_pct
        self.slip_pct = slip_pct
    def exec_trade(self, side, entry, tp, sl):
        entry_exec = entry * (1 + self.slip_pct if side=='buy' else 1 - self.slip_pct)
        tp_exec = tp * (1 - self.slip_pct if side=='buy' else 1 + self.slip_pct)
        sl_exec = sl * (1 + self.slip_pct if side=='buy' else 1 - self.slip_pct)
        return entry_exec, tp_exec, sl_exec