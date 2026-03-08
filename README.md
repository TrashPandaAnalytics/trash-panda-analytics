# trash-panda-analytics

Code from the [Trash Panda Analytics](https://trashpandaanalytics.substack.com) newsletter (formerly Probably Wrong).

I'm a nobody with Python and trust issues. I check the math on things people say about money. These are the scripts behind each post. Run them. Break them. Find my mistakes. Tell me about my mistakes. I'll fix them and pretend I knew all along.

## Requirements

```
pip install yfinance pandas numpy scipy statsmodels
```

## Posts

| File | Post | What it does |
|------|------|-------------|
| [scripts/post02_survivorship_bias.py](scripts/post02_survivorship_bias.py) | [Your Fund Manager's Track Record Is a Magic Trick](https://trashpandaanalytics.substack.com) | Simulates 1,000 fund managers over 10 years to show how survivorship bias creates fake track records |
| [notebooks/post02_survivorship_bias.ipynb](notebooks/post02_survivorship_bias.ipynb) | ↳ notebook version | Same simulation with inline output |
| [scripts/post03_hindenburg.py](scripts/post03_hindenburg.py) | [The Hindenburg Omen](https://trashpandaanalytics.substack.com) | Backtests every Hindenburg Omen trigger since 1996 against actual forward returns |
| [scripts/make_charts_hindenburg.py](scripts/make_charts_hindenburg.py) | ↳ chart generator | Builds animated stacked bar charts (daily/weekly/monthly windows) |
| [notebooks/post03_hindenburg.ipynb](notebooks/post03_hindenburg.ipynb) | ↳ notebook version | Same backtest with inline output |

More posts coming soon. Subscribe to the [newsletter](https://trashpandaanalytics.substack.com) to follow along.

## Notes

Nothing here is financial advice. I'm probably wrong about something in every file. If you find it, open an issue.

## License

MIT. Do whatever you want with it. If you make money using any of these strategies I will be genuinely shocked and would like to hear about it.
