# TastyAlgo

Regime-aware algorithmic trading dashboard that actually works (most of the time)

## what this does

Built this to experiment with different trading strategies and see how they perform across various market conditions. turns out market regime matters way more than i initially thought.

- fetches real market data from yahoo finance
- detects market regimes using statistical methods or k-means clustering
- tests 4 different trading strategies with proper backtesting
- shows performance breakdown by market regime
- interactive charts and csv export for deeper analysis

## strategies included

**MA crossover** - classic trend following using moving averages. buy when short MA crosses above long MA, sell when it crosses below. works great in trends, gets chopped up in sideways markets.

**momentum** - buys the strong stuff, sells the weak stuff. uses n-day returns vs configurable thresholds. can work really well in trending environments but watch out for nasty reversals.

**vol breakout** - trades when volatility spikes above historical norms. idea is that high vol often precedes big moves. catches some good ones but also plenty of false signals.

**pairs trading** - market neutral approach that trades mean reversion between correlated stocks. uses cointegration analysis to find statistical arbitrage opportunities.

## quick setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## project structure

```
TastyAlgo/
├── app.py                     # main streamlit dashboard
├── data_fetcher.py            # yahoo finance data handler
├── regime_detection.py        # market regime classification  
├── backtesting.py            # trade execution and portfolio tracking
├── metrics.py                # performance calculations
├── visualizations.py         # plotly chart generation
├── utils.py                  # helper functions and utilities
└── strategies/               # individual strategy implementations
    ├── moving_average.py
    ├── momentum.py
    ├── volatility_breakout.py
    └── pairs_trading.py
```

## key features

- **real market data** - uses yahoo finance, not synthetic/fake data
- **regime detection** - statistical and machine learning approaches to classify market states
- **proper backtesting** - includes commissions, proper trade execution, realistic assumptions
- **comprehensive metrics** - sharpe ratio, max drawdown, win rate, etc. all split by regime
- **interactive charts** - plotly-powered visualizations with hover info and zooming
- **csv export** - download detailed results for further analysis

## how regime detection works

**statistical method** - uses rolling returns and volatility thresholds to classify market states into bull/bear/sideways

**k-means clustering** - groups similar market conditions automatically using multiple features (returns, vol, rsi, volume)

the regime stuff actually helps a lot - most strategies perform very differently depending on market conditions.

## performance metrics

calculates the usual suspects:
- total and annualized returns
- sharpe ratio for risk-adjusted returns
- maximum drawdown (worst peak-to-trough decline)
- win rate (percentage of profitable trades)
- volatility and other risk measures
- all metrics broken down by market regime

also compares against simple buy & hold benchmark.

## data notes

uses yahoo finance via the yfinance package. data gets cached for 1 hour to avoid hitting rate limits. 

for educational purposes only - dont use this for actual trading without doing your own research and testing.

## known issues

- sometimes yahoo finance data is flaky or missing
- pairs trading strategy could use more sophisticated hedge ratio calculation
- regime detection can be a bit noisy during transition periods
- commission model is simplified
- probably some other bugs i haven't found yet

## possible extensions

could add:
- more sophisticated regime detection (maybe using ML)
- portfolio optimization across multiple assets
- real-time data feeds
- more trading strategies (mean reversion, breakout, etc)
- walk-forward analysis and parameter optimization
- better transaction cost modeling
- risk management overlays

## disclaimer

this is for educational and research purposes only. Do your own research if want to use it for real trading.

Built this while learning about quantitative finance - definitely learned a lot in the process. probably has some quirks but seems to work reasonably well for what it is.