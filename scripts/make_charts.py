# make_charts.py
# Generates post04_golden_cross_stacked.gif
# Trash Panda Analytics - https://trashpandaanalytics.substack.com

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import yfinance as yf
import os

# style
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'
plt.rcParams['text.color'] = '#e0e0e0'
plt.rcParams['axes.labelcolor'] = '#e0e0e0'
plt.rcParams['xtick.color'] = '#999999'
plt.rcParams['ytick.color'] = '#999999'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 10

# download real S&P 500 data from Yahoo Finance
print("Downloading S&P 500 data from Yahoo Finance...")
ticker = yf.Ticker("^GSPC")
data = ticker.history(start="1990-01-01", end="2025-01-01")
data.index = data.index.tz_localize(None)  # remove timezone info for compatibility

sp500 = pd.DataFrame({'close': data['Close']})
sp500['sma_50'] = sp500['close'].rolling(window=50).mean()
sp500['sma_200'] = sp500['close'].rolling(window=200).mean()
sp500['signal'] = 0
sp500.loc[sp500['sma_50'] > sp500['sma_200'], 'signal'] = 1
sp500['market_return'] = sp500['close'].pct_change()
sp500['strategy_return'] = sp500['market_return'] * sp500['signal'].shift(1)
sp500['market_cum'] = (1 + sp500['market_return']).cumprod()
sp500['strategy_cum'] = (1 + sp500['strategy_return']).cumprod()
sp500 = sp500.dropna()

# find crossover points
sp500['cross'] = sp500['signal'].diff()
golden_crosses = sp500[sp500['cross'] == 1].index
death_crosses = sp500[sp500['cross'] == -1].index

print(f"Data ready. {len(sp500)} days, {len(golden_crosses)} golden crosses, {len(death_crosses)} death crosses")

# --- STACKED CHART: Crossovers on top, Cumulative on bottom ---
print("Building stacked animation...")

# module-level variables set per sweep
weekly = None
frame_indices = None
x_min = x_max = None
ax1 = ax2 = None

def animate_stacked(frame_num):
    ax1.clear()
    ax2.clear()

    idx = frame_indices[frame_num]
    subset = weekly.iloc[:idx+1]

    if len(subset) < 2:
        return

    dates = subset.index.to_numpy()

    # === TOP: Crossover chart ===
    ax1.plot(dates, subset['close'].values, color='#e0e0e0', linewidth=0.8, alpha=0.9, label='S&P 500')
    ax1.plot(dates, subset['sma_50'].values, color='#4ecdc4', linewidth=1.2, alpha=0.8, label='50-day SMA')
    ax1.plot(dates, subset['sma_200'].values, color='#ff6b6b', linewidth=1.2, alpha=0.8, label='200-day SMA')

    if len(subset) > 1:
        ax1.fill_between(dates, subset['close'].min() * 0.9, subset['close'].max() * 1.05,
                         where=subset['signal'].values == 1, alpha=0.06, color='#4ecdc4')
        ax1.fill_between(dates, subset['close'].min() * 0.9, subset['close'].max() * 1.05,
                         where=subset['signal'].values == 0, alpha=0.06, color='#ff6b6b')

    current_date = subset.index[-1]
    for gc in golden_crosses:
        if gc <= current_date:
            ax1.axvline(gc, color='#4ecdc4', alpha=0.3, linewidth=0.5)
            ax1.scatter([gc], [sp500.loc[gc, 'close']], color='#4ecdc4', s=30, zorder=5, marker='^')

    for dc in death_crosses:
        if dc <= current_date:
            ax1.axvline(dc, color='#ff6b6b', alpha=0.3, linewidth=0.5)
            ax1.scatter([dc], [sp500.loc[dc, 'close']], color='#ff6b6b', s=30, zorder=5, marker='v')

    current_signal = subset['signal'].iloc[-1]
    status = "IN MARKET" if current_signal == 1 else "IN CASH"
    status_color = '#4ecdc4' if current_signal == 1 else '#ff6b6b'
    ax1.text(0.98, 0.95, status, transform=ax1.transAxes, fontsize=14, fontweight='bold',
             color=status_color, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=status_color, alpha=0.9))

    ax1.set_title('Golden Cross Strategy: S&P 500 (1990-2025)', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.5, facecolor='#1a1a2e', edgecolor='#333333')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.set_ylim(subset['close'].min() * 0.9, subset['close'].max() * 1.08)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.1)

    # === BOTTOM: Cumulative portfolio chart ===
    market_norm = (subset['market_cum'] / subset['market_cum'].iloc[0]).values
    strat_norm = (subset['strategy_cum'] / subset['strategy_cum'].iloc[0]).values

    ax2.plot(dates, market_norm, color='#e0e0e0', linewidth=1.5, alpha=0.9, label='Buy & Hold')
    ax2.plot(dates, strat_norm, color='#ffd93d', linewidth=1.5, alpha=0.9, label='Golden Cross')

    ax2.fill_between(dates, market_norm, strat_norm,
                     where=market_norm > strat_norm, alpha=0.15, color='#ff6b6b',
                     interpolate=True)
    ax2.fill_between(dates, market_norm, strat_norm,
                     where=strat_norm > market_norm, alpha=0.15, color='#4ecdc4',
                     interpolate=True)

    m_val = market_norm[-1]
    s_val = strat_norm[-1]
    ax2.text(dates[-1], m_val, f'  ${m_val:.1f}x', color='#e0e0e0', fontsize=9, va='center')
    ax2.text(dates[-1], s_val, f'  ${s_val:.1f}x', color='#ffd93d', fontsize=9, va='center')

    diff_pct = ((s_val / m_val) - 1) * 100
    diff_color = '#4ecdc4' if diff_pct >= 0 else '#ff6b6b'
    diff_text = f"Strategy vs Buy&Hold: {diff_pct:+.0f}%"
    ax2.text(0.98, 0.05, diff_text, transform=ax2.transAxes, fontsize=12, fontweight='bold',
             color=diff_color, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=diff_color, alpha=0.9))

    ax2.set_title('$1 Invested: Buy & Hold vs Golden Cross', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.5, facecolor='#1a1a2e', edgecolor='#333333')
    ax2.set_ylabel('Portfolio Value ($1 start)', fontsize=10)
    ax2.set_yscale('log')
    ax2.set_xlim(x_min, x_max)
    ax2.grid(True, alpha=0.1)

# --- RENDER (daily / weekly / monthly sweeps) ---
chartdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'charts')
os.makedirs(chartdir, exist_ok=True)

SWEEPS = [('daily', None), ('weekly', 'W'), ('monthly', 'ME')]

for sweep_label, freq in SWEEPS:
    weekly = sp500 if freq is None else sp500.resample(freq).last().dropna()
    n = len(weekly)
    step = max(1, n // 300) if freq is None else (4 if freq == 'W' else 2)
    frame_indices = list(range(0, n, step)) + [n - 1]
    all_dates = weekly.index.to_numpy()
    x_min, x_max = all_dates[0], all_dates[-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.06, hspace=0.12)

    outpath = os.path.join(chartdir, f'post04_golden_cross_stacked_{sweep_label}.gif')
    print(f"Rendering {sweep_label}: {len(frame_indices)} frames...")
    ani = animation.FuncAnimation(fig, animate_stacked, frames=len(frame_indices),
                                  interval=80, repeat=True)
    ani.save(outpath, writer='pillow', fps=12, dpi=100)
    plt.close(fig)
    print(f"  Saved: {outpath}")

print("\nDone.")
