# post04_golden_cross.py
# "I Backtested the Golden Cross So You Don't Have To"
# Trash Panda Analytics - https://trashpandaanalytics.substack.com

import yfinance as yf
import pandas as pd
import numpy as np

# pull S&P 500 data
sp500 = yf.download('^GSPC', start='1970-01-01', end='2025-01-01')
sp500 = sp500[['Close']].copy()
sp500.columns = ['close']

# calculate moving averages
sp500['sma_50'] = sp500['close'].rolling(window=50).mean()
sp500['sma_200'] = sp500['close'].rolling(window=200).mean()

# generate signals
sp500['signal'] = 0
sp500.loc[sp500['sma_50'] > sp500['sma_200'], 'signal'] = 1
sp500.loc[sp500['sma_50'] <= sp500['sma_200'], 'signal'] = 0

# calculate returns
sp500['market_return'] = sp500['close'].pct_change()
sp500['strategy_return'] = sp500['market_return'] * sp500['signal'].shift(1)

# cumulative returns
sp500['market_cumulative'] = (1 + sp500['market_return']).cumprod()
sp500['strategy_cumulative'] = (1 + sp500['strategy_return']).cumprod()

# results
sp500_clean = sp500.dropna()

total_market = sp500_clean['market_cumulative'].iloc[-1]
total_strategy = sp500_clean['strategy_cumulative'].iloc[-1]

crosses = sp500_clean['signal'].diff().abs()
n_trades = crosses.sum() / 2
pct_in_market = sp500_clean['signal'].mean() * 100

def max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min() * 100

market_dd = max_drawdown(sp500_clean['market_cumulative'])
strategy_dd = max_drawdown(sp500_clean['strategy_cumulative'])

n_years = len(sp500_clean) / 252
market_ann = (total_market ** (1/n_years) - 1) * 100
strategy_ann = (total_strategy ** (1/n_years) - 1) * 100

market_vol = sp500_clean['market_return'].std() * np.sqrt(252) * 100
strategy_vol = sp500_clean['strategy_return'].std() * np.sqrt(252) * 100

market_sharpe = market_ann / market_vol
strategy_sharpe = strategy_ann / strategy_vol

print("=== GOLDEN CROSS BACKTEST: S&P 500 (1970-2025) ===")
print(f"")
print(f"Buy & Hold:")
print(f"  Total return:       {total_market:.1f}x")
print(f"  Annualized return:  {market_ann:.1f}%")
print(f"  Max drawdown:       {market_dd:.1f}%")
print(f"  Volatility (ann):   {market_vol:.1f}%")
print(f"  Sharpe ratio:       {market_sharpe:.2f}")
print(f"")
print(f"Golden Cross Strategy:")
print(f"  Total return:       {total_strategy:.1f}x")
print(f"  Annualized return:  {strategy_ann:.1f}%")
print(f"  Max drawdown:       {strategy_dd:.1f}%")
print(f"  Volatility (ann):   {strategy_vol:.1f}%")
print(f"  Sharpe ratio:       {strategy_sharpe:.2f}")
print(f"")
print(f"  Round-trip trades:  {n_trades:.0f}")
print(f"  Time in market:     {pct_in_market:.1f}%")

# whipsaw analysis
print(f"\n\n=== WHIPSAW ANALYSIS ===")
sp500_clean['cross'] = sp500_clean['signal'].diff().abs()
cross_dates = sp500_clean[sp500_clean['cross'] == 1].index
gaps = [(cross_dates[i+1] - cross_dates[i]).days for i in range(len(cross_dates)-1)]
gaps_series = pd.Series(gaps)

print(f"Total signals:     {len(cross_dates)}")
print(f"Median gap:        {gaps_series.median():.0f} days")
print(f"Signals < 30 days: {(gaps_series < 30).sum()}")
print(f"Signals < 14 days: {(gaps_series < 14).sum()}")
print(f"Shortest gap:      {gaps_series.min()} days")
