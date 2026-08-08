# utils/portfolio_metrics.py
import numpy as np
import pandas as pd

def calculate_portfolio_metrics(portfolio_values: list, risk_free_rate: float = 0.0) -> dict:
    """
    Takes a list of daily portfolio values and calculates institutional-grade risk/reward metrics.
    """
    values = np.array(portfolio_values)
    
    # Calculate daily returns
    # Add a tiny epsilon to avoid division by zero if the portfolio wiped out entirely
    daily_returns = (values[1:] - values[:-1]) / (values[:-1] + 1e-8)
    
    # 1. Total Return & CAGR
    total_return = (values[-1] / values[0]) - 1.0
    trading_days = len(values)
    cagr = (values[-1] / values[0]) ** (252.0 / trading_days) - 1.0 if trading_days > 0 else 0.0
    
    # 2. Risk Metrics (Volatility)
    daily_vol = np.std(daily_returns)
    annual_vol = daily_vol * np.sqrt(252)
    
    # 3. Sharpe Ratio
    mean_return = np.mean(daily_returns)
    excess_return = mean_return - (risk_free_rate / 252.0)
    sharpe = (excess_return / (daily_vol + 1e-8)) * np.sqrt(252)
    
    # 4. Sortino Ratio (Downside Risk)
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
    sortino = (excess_return / (downside_vol + 1e-8)) * np.sqrt(252)
    
    # 5. Drawdowns
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / (running_max + 1e-8)
    max_drawdown = np.min(drawdowns)
    
    return {
        "Final_Value": values[-1],
        "Total_Return_Pct": total_return * 100,
        "CAGR_Pct": cagr * 100,
        "Ann_Volatility_Pct": annual_vol * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown_Pct": max_drawdown * 100
    }