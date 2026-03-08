# post02_survivorship_bias.py
# "Your Fund Manager's Track Record Is a Magic Trick"
# Trash Panda Analytics - https://trashpandaanalytics.substack.com

import numpy as np
import pandas as pd

np.random.seed(42)

n_managers = 1000
n_years = 10
n_simulations = 1000

def simulate_survivors(n_managers, n_years):
    """
    simulate fund managers with ZERO skill.
    each year, each manager has a 50/50 chance of beating the market.
    if they underperform, 30% of them shut down.
    """
    surviving = np.ones(n_managers, dtype=bool)
    streak = np.zeros(n_managers, dtype=int)
    
    for year in range(n_years):
        beat_market = np.random.binomial(1, 0.5, n_managers).astype(bool)
        streak = np.where(beat_market & surviving, streak + 1, 0)
        underperformed = ~beat_market & surviving
        closes = underperformed & (np.random.random(n_managers) < 0.30)
        surviving[closes] = False
    
    return surviving, streak

results = []
for sim in range(n_simulations):
    np.random.seed(sim)
    surviving, streak = simulate_survivors(n_managers, n_years)
    
    results.append({
        'simulation': sim,
        'survivors': surviving.sum(),
        'streak_5plus': (streak[surviving] >= 5).sum(),
        'streak_7plus': (streak[surviving] >= 7).sum(),
        'max_streak': streak[surviving].max() if surviving.any() else 0,
        'avg_streak_survivors': streak[surviving].mean() if surviving.any() else 0
    })

df = pd.DataFrame(results)

print("=== SURVIVORSHIP BIAS SIMULATION ===")
print(f"Started with: {n_managers} managers (zero skill)")
print(f"Simulations: {n_simulations}")
print(f"")
print(f"After {n_years} years:")
print(f"  Average survivors: {df['survivors'].mean():.0f} / {n_managers}")
print(f"  Avg managers with 5+ year streak: {df['streak_5plus'].mean():.1f}")
print(f"  Avg managers with 7+ year streak: {df['streak_7plus'].mean():.1f}")
print(f"  Average 'best' streak among survivors: {df['max_streak'].mean():.1f} years")
print(f"  Average streak of all survivors: {df['avg_streak_survivors'].mean():.1f} years")
