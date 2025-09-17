# Placeholder for backtest metrics utilities
# This file can be extended with comprehensive backtesting metrics
# such as Sharpe ratio, maximum drawdown, win rate, etc.

import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Placeholder for Sharpe ratio calculation
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) - risk_free_rate) / np.std(returns)

def calculate_max_drawdown(equity_curve):
    """
    Placeholder for maximum drawdown calculation
    """
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)

def calculate_win_rate(trades):
    """
    Placeholder for win rate calculation
    """
    if len(trades) == 0:
        return 0.0
    winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
    return winning_trades / len(trades)