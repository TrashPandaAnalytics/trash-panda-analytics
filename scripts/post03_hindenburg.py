# post03_hindenburg.py
# "The Hindenburg Omen: Someone Named a Technical Indicator After a Burning Airship"
# Trash Panda Analytics - https://trashpandaanalytics.substack.com

import yfinance as yf
import pandas as pd
import numpy as np

# ~85 major stocks as a proxy for NYSE breadth.
# the actual omen uses all NYSE traded issues.
# this is imperfect. acknowledged. moving on.
TICKERS = [
    # Technology
    'AAPL', 'MSFT', 'INTC', 'CSCO', 'ORCL', 'IBM', 'TXN', 'QCOM',
    'ADBE', 'HPQ', 'AMD', 'MU',
    # Healthcare
    'JNJ', 'PFE', 'MRK', 'ABT', 'BMY', 'AMGN', 'GILD', 'MDT',
    'UNH', 'LLY',
    # Financials
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BK', 'USB',
    'PNC', 'MET',
    # Consumer Discretionary
    'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'F',
    # Consumer Staples
    'PG', 'KO', 'PEP', 'WMT', 'CL', 'GIS', 'MO', 'SYY',
    # Industrials
    'GE', 'MMM', 'CAT', 'BA', 'HON', 'UPS', 'LMT', 'DE', 'EMR',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'VLO', 'HAL',
    # Utilities
    'DUK', 'SO', 'NEE', 'D', 'AEP', 'SRE', 'EXC', 'ED',
    # Telecom / Media
    'T', 'VZ', 'CMCSA', 'DIS',
    # Materials / REITs
    'APD', 'ECL', 'FCX', 'NEM', 'NUE', 'SPG', 'PLD',
]


def download_data():
    """download individual stocks and S&P 500 index."""
    print(f"Downloading data for {len(TICKERS)} stocks...")
    raw = yf.download(TICKERS, start='2000-01-01', end='2025-01-01')
    closes = raw['Close']

    print("Downloading S&P 500 index...")
    sp500 = yf.download('^GSPC', start='2000-01-01', end='2025-01-01')
    sp500_close = sp500['Close'].squeeze()

    print(f"Got data for {closes.shape[1]} stocks")
    return closes, sp500_close


def compute_breadth(closes, sp500_close):
    """compute all breadth metrics and hindenburg omen signals."""
    sma_50 = sp500_close.rolling(50).mean()

    # 52-week highs and lows
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

    breadth = pd.DataFrame({
        'pct_new_highs': pct_new_highs,
        'pct_new_lows': pct_new_lows,
        'mcclellan': mcclellan,
        'n_stocks': stocks_available,
        'sp500': sp500_close,
        'sma_50': sma_50,
    }).dropna()

    # Hindenburg Omen criteria
    THRESHOLD = 0.022
    breadth['omen'] = (
        (breadth['pct_new_highs'] > THRESHOLD) &
        (breadth['pct_new_lows'] > THRESHOLD) &
        (breadth['sp500'] > breadth['sma_50']) &
        (breadth['mcclellan'] < 0) &
        (breadth['pct_new_highs'] < 2 * breadth['pct_new_lows'] + 0.001)
    ).astype(int)

    return breadth


def cluster_signals(breadth, gap_days=30):
    """group omen dates within gap_days into single clusters."""
    omen_dates = breadth[breadth['omen'] == 1].index
    clusters = []
    cluster_start = None
    for d in omen_dates:
        if cluster_start is None or (d - cluster_start).days > gap_days:
            cluster_start = d
            clusters.append(d)
    return clusters


def measure_forward_returns(prices, signal_dates, windows=[30, 60, 90]):
    """measure forward return and max drawdown after each signal date."""
    results = []
    for signal_date in signal_dates:
        if signal_date not in prices.index:
            continue
        loc = prices.index.get_loc(signal_date)
        for window in windows:
            if loc + window >= len(prices):
                continue
            future = prices.iloc[loc:loc + window]
            start_price = future.iloc[0]
            forward_return = (future.iloc[-1] / start_price - 1) * 100
            max_dd = ((future / future.cummax()) - 1).min() * 100

            results.append({
                'date': signal_date,
                'window': window,
                'forward_return': forward_return,
                'max_drawdown': max_dd,
                'decline_5pct': max_dd < -5,
                'decline_10pct': max_dd < -10,
            })
    return pd.DataFrame(results)


def monte_carlo_test(prices, signal_dates, n_sims=10000, window=60):
    """compare omen forward drawdowns against random date samples."""
    valid_idx = prices.dropna().index[252:-window]

    actual_dds = []
    for d in signal_dates:
        if d not in prices.index:
            continue
        loc = prices.index.get_loc(d)
        if loc + window >= len(prices):
            continue
        future = prices.iloc[loc:loc + window]
        actual_dds.append(((future / future.cummax()) - 1).min() * 100)

    actual_mean = np.mean(actual_dds)

    worse_count = 0
    random_means = []
    for _ in range(n_sims):
        rand_dates = np.random.choice(valid_idx, size=len(signal_dates),
                                      replace=False)
        rand_dds = []
        for d in rand_dates:
            loc = prices.index.get_loc(d)
            if loc + window >= len(prices):
                continue
            future = prices.iloc[loc:loc + window]
            rand_dds.append(((future / future.cummax()) - 1).min() * 100)

        rm = np.mean(rand_dds)
        random_means.append(rm)
        if rm <= actual_mean:
            worse_count += 1

    percentile = (worse_count / n_sims) * 100

    print(f"=== MONTE CARLO: HINDENBURG OMEN vs RANDOM ===")
    print(f"Window: {window} trading days")
    print(f"Simulations: {n_sims}\n")
    print(f"Avg max drawdown after Omen: {actual_mean:.1f}%")
    print(f"Avg max drawdown after random: {np.mean(random_means):.1f}%")
    print(f"Omen percentile: {percentile:.0f}th")
    print(f"(higher = worse drawdowns after omen than random)\n")

    if percentile > 90:
        print("The Omen IS associated with worse forward drawdowns.")
        print("Whether that's tradeable after whipsaw is another question.")
    elif percentile > 70:
        print("Slight tendency toward worse outcomes. Not convincing.")
    else:
        print("The Omen does NOT predict worse returns than random dates.")
        print("The burning airship metaphor is doing more work than the math.")


def backtest_hindenburg(prices, omen_signal, cash_days=30):
    """sell on omen, sit in cash for cash_days, re-enter."""
    df = pd.DataFrame({'close': prices})
    df['market_return'] = df['close'].pct_change()

    df['signal'] = 1
    for d in omen_signal[omen_signal == 1].index:
        loc = df.index.get_loc(d)
        end = min(loc + cash_days, len(df))
        df.iloc[loc:end, df.columns.get_loc('signal')] = 0

    df['strategy_return'] = df['market_return'] * df['signal'].shift(1)
    df['market_cum'] = (1 + df['market_return']).cumprod()
    df['strategy_cum'] = (1 + df['strategy_return']).cumprod()

    clean = df.dropna()
    n_years = len(clean) / 252

    bh_total = clean['market_cum'].iloc[-1]
    strat_total = clean['strategy_cum'].iloc[-1]
    bh_ann = (bh_total ** (1/n_years) - 1) * 100
    strat_ann = (strat_total ** (1/n_years) - 1) * 100
    bh_dd = ((clean['market_cum'] / clean['market_cum'].cummax()) - 1).min() * 100
    strat_dd = ((clean['strategy_cum'] / clean['strategy_cum'].cummax()) - 1).min() * 100
    time_out = (1 - clean['signal'].mean()) * 100

    print(f"=== HINDENBURG OMEN STRATEGY BACKTEST ===")
    print(f"Rule: sell to cash for {cash_days} days when omen fires\n")
    print(f"Buy & Hold:")
    print(f"  Total return: {bh_total:.1f}x")
    print(f"  Annualized:   {bh_ann:.1f}%")
    print(f"  Max drawdown: {bh_dd:.1f}%")
    print(f"\nHindenburg Strategy:")
    print(f"  Total return: {strat_total:.1f}x")
    print(f"  Annualized:   {strat_ann:.1f}%")
    print(f"  Max drawdown: {strat_dd:.1f}%")
    print(f"  Time in cash: {time_out:.1f}%")


# run it
closes, sp500_close = download_data()
breadth = compute_breadth(closes, sp500_close)
clusters = cluster_signals(breadth)

omen_dates = breadth[breadth['omen'] == 1].index

print("\n=== HINDENBURG OMEN DETECTION ===")
print(f"Period: 2000-2025")
print(f"Stocks in universe: {closes.shape[1]}")
print(f"Individual omen days: {len(omen_dates)}")
print(f"Signal clusters (30-day grouping): {len(clusters)}")
print(f"Average omens per year: {len(omen_dates) / 25:.1f}")

# forward returns
print("\n")
omen_fwd = measure_forward_returns(breadth['sp500'], clusters)

np.random.seed(42)
valid_idx = breadth.dropna().index[252:-90]
random_dates = np.random.choice(valid_idx, size=len(clusters) * 20,
                                replace=False)
random_fwd = measure_forward_returns(breadth['sp500'], random_dates)

print("=== FORWARD RETURNS AFTER HINDENBURG OMEN ===\n")
for window in [30, 60, 90]:
    omen_w = omen_fwd[omen_fwd['window'] == window]
    rand_w = random_fwd[random_fwd['window'] == window]
    print(f"--- {window}-day window ---")
    print(f"After Omen (n={len(omen_w)}):")
    print(f"  Avg return:       {omen_w['forward_return'].mean():+.1f}%")
    print(f"  Avg max drawdown: {omen_w['max_drawdown'].mean():.1f}%")
    print(f"  5%+ decline:      {omen_w['decline_5pct'].mean()*100:.0f}%")
    print(f"  10%+ decline:     {omen_w['decline_10pct'].mean()*100:.0f}%")
    print(f"After Random (n={len(rand_w)}):")
    print(f"  Avg return:       {rand_w['forward_return'].mean():+.1f}%")
    print(f"  Avg max drawdown: {rand_w['max_drawdown'].mean():.1f}%")
    print(f"  5%+ decline:      {rand_w['decline_5pct'].mean()*100:.0f}%")
    print(f"  10%+ decline:     {rand_w['decline_10pct'].mean()*100:.0f}%")
    print()

# monte carlo
print("\n")
monte_carlo_test(breadth['sp500'], clusters)

# strategy backtest
print("\n")
backtest_hindenburg(breadth['sp500'], breadth['omen'])
