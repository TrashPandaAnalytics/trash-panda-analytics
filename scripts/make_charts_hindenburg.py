# make_charts_hindenburg.py
# Generates post03_hindenburg_stacked.gif
# Trash Panda Analytics - https://trashpandaanalytics.substack.com

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yfinance as yf
import os

# style (matches other chart scripts)
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#1a1a2e'
plt.rcParams['text.color'] = '#e0e0e0'
plt.rcParams['axes.labelcolor'] = '#e0e0e0'
plt.rcParams['xtick.color'] = '#999999'
plt.rcParams['ytick.color'] = '#999999'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 10

# stock universe for breadth proxy
TICKERS = [
    'AAPL', 'MSFT', 'INTC', 'CSCO', 'ORCL', 'IBM', 'TXN', 'QCOM',
    'ADBE', 'HPQ', 'AMD', 'MU',
    'JNJ', 'PFE', 'MRK', 'ABT', 'BMY', 'AMGN', 'GILD', 'MDT',
    'UNH', 'LLY',
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BK', 'USB',
    'PNC', 'MET',
    'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'F',
    'PG', 'KO', 'PEP', 'WMT', 'CL', 'GIS', 'MO', 'SYY',
    'GE', 'MMM', 'CAT', 'BA', 'HON', 'UPS', 'LMT', 'DE', 'EMR',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'VLO', 'HAL',
    'DUK', 'SO', 'NEE', 'D', 'AEP', 'SRE', 'EXC', 'ED',
    'T', 'VZ', 'CMCSA', 'DIS',
    'APD', 'ECL', 'FCX', 'NEM', 'NUE', 'SPG', 'PLD',
]

# --- DATA ---
print(f"Downloading data for {len(TICKERS)} stocks...")
raw = yf.download(TICKERS, start='2000-01-01', end='2025-01-01')
closes = raw['Close']

print("Downloading S&P 500 index...")
sp500_raw = yf.download('^GSPC', start='2000-01-01', end='2025-01-01')
sp500_close = sp500_raw['Close'].squeeze()
sma_50 = sp500_close.rolling(50).mean()

print(f"Got data for {closes.shape[1]} stocks")

# --- BREADTH CALCULATION ---
rolling_max = closes.rolling(252, min_periods=252).max()
rolling_min = closes.rolling(252, min_periods=252).min()
new_highs = (closes >= rolling_max) & closes.notna()
new_lows = (closes <= rolling_min) & closes.notna()
stocks_available = closes.notna().sum(axis=1)

pct_new_highs = new_highs.sum(axis=1) / stocks_available
pct_new_lows = new_lows.sum(axis=1) / stocks_available

# McClellan Oscillator
daily_returns = closes.pct_change()
advances = (daily_returns > 0).sum(axis=1)
declines = (daily_returns < 0).sum(axis=1)
ad_diff = advances - declines
mcclellan = ad_diff.ewm(span=19).mean() - ad_diff.ewm(span=39).mean()

# assemble daily dataframe
daily = pd.DataFrame({
    'sp500': sp500_close,
    'sma_50': sma_50,
    'pct_new_highs': pct_new_highs,
    'pct_new_lows': pct_new_lows,
    'mcclellan': mcclellan,
}).dropna()

# Hindenburg Omen
THRESHOLD = 0.022
daily['omen'] = (
    (daily['pct_new_highs'] > THRESHOLD) &
    (daily['pct_new_lows'] > THRESHOLD) &
    (daily['sp500'] > daily['sma_50']) &
    (daily['mcclellan'] < 0) &
    (daily['pct_new_highs'] < 2 * daily['pct_new_lows'] + 0.001)
).astype(int)

# strategy: sell for 30 days on omen
CASH_DAYS = 30
daily['signal'] = 1
for d in daily[daily['omen'] == 1].index:
    loc = daily.index.get_loc(d)
    end = min(loc + CASH_DAYS, len(daily))
    daily.iloc[loc:end, daily.columns.get_loc('signal')] = 0

daily['market_return'] = daily['sp500'].pct_change()
daily['strategy_return'] = daily['market_return'] * daily['signal'].shift(1)
daily['market_cum'] = (1 + daily['market_return']).cumprod()
daily['strategy_cum'] = (1 + daily['strategy_return']).cumprod()
daily = daily.dropna()

# omen dates for markers
omen_dates_all = daily[daily['omen'] == 1].index

print(f"Data ready. {len(daily)} days, {len(omen_dates_all)} omen signals")

# --- BUILD ANIMATION ---
print("Building stacked animation...")

HINDENBURG_AGG = {
    'sp500': 'last', 'sma_50': 'last',
    'pct_new_highs': 'max', 'pct_new_lows': 'max',
    'omen': 'max', 'signal': 'last',
    'market_cum': 'last', 'strategy_cum': 'last',
}

# module-level variables set per sweep
weekly = None
frame_indices = None
x_min = x_max = None
ax1 = ax_mid = ax2 = None


def animate_stacked(frame_num):
    ax1.clear()
    ax_mid.clear()
    ax2.clear()

    idx = frame_indices[frame_num]
    subset = weekly.iloc[:idx+1]

    if len(subset) < 2:
        return

    dates = subset.index.to_numpy()

    # === TOP: S&P 500 price with Omen markers ===
    ax1.plot(dates, subset['sp500'].values, color='#e0e0e0', linewidth=0.8,
             alpha=0.9, label='S&P 500')
    ax1.plot(dates, subset['sma_50'].values, color='#4ecdc4', linewidth=1.0,
             alpha=0.6, label='50-day SMA')

    # shade market/cash periods
    if len(subset) > 1:
        ax1.fill_between(dates, subset['sp500'].min() * 0.9,
                         subset['sp500'].max() * 1.05,
                         where=subset['signal'].values == 0,
                         alpha=0.08, color='#ff6b6b')

    # mark omen signals that have occurred
    current_date = subset.index[-1]
    for od in omen_dates_all:
        if od <= current_date:
            # find the closest weekly date
            ax1.axvline(od, color='#ff4444', alpha=0.25, linewidth=0.7)

    # omen markers on the price line (weekly omen weeks)
    omen_weeks = subset[subset['omen'] == 1]
    if len(omen_weeks) > 0:
        ax1.scatter(omen_weeks.index.to_numpy(), omen_weeks['sp500'].values,
                    color='#ff4444', s=40, zorder=5, marker='v',
                    label='Hindenburg Omen')

    current_signal = subset['signal'].iloc[-1]
    if current_signal == 1:
        status = "IN MARKET"
        status_color = '#4ecdc4'
    else:
        status = "OMEN: IN CASH"
        status_color = '#ff4444'

    ax1.text(0.98, 0.95, status, transform=ax1.transAxes, fontsize=14,
             fontweight='bold', color=status_color, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                       edgecolor=status_color, alpha=0.9))

    ax1.set_title('Hindenburg Omen Strategy: S&P 500 (2000-2025)',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.5,
               facecolor='#1a1a2e', edgecolor='#333333')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.set_ylim(subset['sp500'].min() * 0.9, subset['sp500'].max() * 1.08)
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, alpha=0.1)

    # === MIDDLE: Breadth divergence (new highs % and new lows %) ===
    highs_vals = subset['pct_new_highs'].values * 100
    lows_vals = subset['pct_new_lows'].values * 100

    ax_mid.plot(dates, highs_vals, color='#4ecdc4', linewidth=1.2, alpha=0.9,
                label='New 52w Highs %')
    ax_mid.plot(dates, lows_vals, color='#ff6b6b', linewidth=1.2, alpha=0.9,
                label='New 52w Lows %')

    # threshold line
    ax_mid.axhline(THRESHOLD * 100, color='#ffd93d', linewidth=0.8,
                   linestyle='--', alpha=0.7, label=f'{THRESHOLD*100:.1f}% threshold')

    # fill when both are above threshold (omen territory)
    both_above = (highs_vals > THRESHOLD * 100) & (lows_vals > THRESHOLD * 100)
    ax_mid.fill_between(dates, 0, np.maximum(highs_vals, lows_vals),
                        where=both_above, alpha=0.15, color='#ff4444')

    # current breadth readout
    cur_h = highs_vals[-1]
    cur_l = lows_vals[-1]
    breadth_color = '#ff4444' if (cur_h > THRESHOLD * 100 and cur_l > THRESHOLD * 100) else '#b388ff'
    ax_mid.text(0.98, 0.90,
                f"Highs: {cur_h:.1f}%  Lows: {cur_l:.1f}%",
                transform=ax_mid.transAxes, fontsize=10, fontweight='bold',
                color=breadth_color, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                          edgecolor=breadth_color, alpha=0.9))

    ax_mid.set_title('Market Breadth: New 52-Week Highs & Lows', fontsize=11,
                     fontweight='bold', pad=6)
    ax_mid.legend(loc='upper left', fontsize=7, framealpha=0.5,
                  facecolor='#1a1a2e', edgecolor='#333333')
    ax_mid.set_ylabel('% of Stocks', fontsize=9)
    y_max = max(highs_vals.max(), lows_vals.max(), THRESHOLD * 100 * 2) * 1.1
    ax_mid.set_ylim(0, max(y_max, 5))
    ax_mid.set_xlim(x_min, x_max)
    ax_mid.grid(True, alpha=0.1)

    # === BOTTOM: Cumulative portfolio chart ===
    market_norm = (subset['market_cum'] / subset['market_cum'].iloc[0]).values
    strat_norm = (subset['strategy_cum'] / subset['strategy_cum'].iloc[0]).values

    ax2.plot(dates, market_norm, color='#e0e0e0', linewidth=1.5, alpha=0.9,
             label='Buy & Hold')
    ax2.plot(dates, strat_norm, color='#ffd93d', linewidth=1.5, alpha=0.9,
             label='Hindenburg Strategy')

    ax2.fill_between(dates, market_norm, strat_norm,
                     where=market_norm > strat_norm, alpha=0.15,
                     color='#ff6b6b', interpolate=True)
    ax2.fill_between(dates, market_norm, strat_norm,
                     where=strat_norm > market_norm, alpha=0.15,
                     color='#4ecdc4', interpolate=True)

    m_val = market_norm[-1]
    s_val = strat_norm[-1]
    ax2.text(dates[-1], m_val, f'  ${m_val:.1f}x', color='#e0e0e0',
             fontsize=9, va='center')
    ax2.text(dates[-1], s_val, f'  ${s_val:.1f}x', color='#ffd93d',
             fontsize=9, va='center')

    diff_pct = ((s_val / m_val) - 1) * 100
    diff_color = '#4ecdc4' if diff_pct >= 0 else '#ff6b6b'
    diff_text = f"Strategy vs Buy&Hold: {diff_pct:+.0f}%"
    ax2.text(0.98, 0.05, diff_text, transform=ax2.transAxes, fontsize=12,
             fontweight='bold', color=diff_color, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                       edgecolor=diff_color, alpha=0.9))

    ax2.set_title('$1 Invested: Buy & Hold vs Hindenburg Strategy',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.5,
               facecolor='#1a1a2e', edgecolor='#333333')
    ax2.set_ylabel('Portfolio Value ($1 start)', fontsize=10)
    ax2.set_yscale('log')
    ax2.set_xlim(x_min, x_max)
    ax2.grid(True, alpha=0.1)


# --- RENDER (daily / weekly / monthly sweeps) ---
chartdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'charts')
os.makedirs(chartdir, exist_ok=True)

SWEEPS = [('daily', None), ('weekly', 'W'), ('monthly', 'ME')]

for sweep_label, freq in SWEEPS:
    weekly = daily if freq is None else daily.resample(freq).agg(HINDENBURG_AGG).dropna()
    n = len(weekly)
    step = max(1, n // 300) if freq is None else (4 if freq == 'W' else 2)
    frame_indices = list(range(0, n, step)) + [n - 1]
    all_dates = weekly.index.to_numpy()
    x_min, x_max = all_dates[0], all_dates[-1]

    fig, (ax1, ax_mid, ax2) = plt.subplots(3, 1, figsize=(12, 12), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1.5, 3]})
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05, hspace=0.10)

    outpath = os.path.join(chartdir, f'post03_hindenburg_stacked_{sweep_label}.gif')
    print(f"Rendering {sweep_label}: {len(frame_indices)} frames...")
    ani = animation.FuncAnimation(fig, animate_stacked, frames=len(frame_indices),
                                  interval=80, repeat=True)
    ani.save(outpath, writer='pillow', fps=12, dpi=100)
    plt.close(fig)
    print(f"  Saved: {outpath}")

print("\nDone.")
