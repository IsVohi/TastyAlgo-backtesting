"""
moving average crossover strategy
probably the most basic trend following strategy there is
"""

import pandas as pd
import numpy as np

class MAcross:
  
  def __init__(self):
      self.name = "MA Crossover"
      self.description = """
      classic trend following strategy. buy when short ma crosses above long ma, 
      sell when short ma crosses below long ma. simple but can be effective.
      """
  
  def genSigs(self, data, shortWin=20, longWin=50):
      # generate signals based on moving average crossover
      
      # input validation
      if shortWin >= longWin:
          raise ValueError("short window must be less than long window")
      
      if len(data) < longWin:
          raise ValueError(f"need at least {longWin} data points")
      
      # initialize signals dataframe
      signals = pd.DataFrame(index=data.index)
      signals['price'] = data['Close']
      
      # calculate the moving averages
      signals['short_ma'] = self._calcSMA(data['Close'], shortWin)
      signals['long_ma'] = self._calcSMA(data['Close'], longWin)
      
      # generate trading signals
      signals['signal'] = self._genCrossoverSignals(
          signals['short_ma'], signals['long_ma'], longWin
      )
      
      # calculate position changes
      signals['positions'] = signals['signal'].diff()
      
      # add some extra analysis indicators
      signals['ma_spread'] = ((signals['short_ma'] - signals['long_ma']) / signals['long_ma'] * 100)
      signals['price_vs_short'] = ((signals['price'] - signals['short_ma']) / signals['short_ma'] * 100)
      signals['price_vs_long'] = ((signals['price'] - signals['long_ma']) / signals['long_ma'] * 100)
      
      return signals
  
  def _calcSMA(self, prices, window):
      # calculate simple moving average
      return prices.rolling(window=window, min_periods=window).mean()
  
  def _genCrossoverSignals(self, shortMa, longMa, minPeriods):
      # generate crossover signals
      signals = pd.Series(0, index=shortMa.index)
      
      # only generate signals after we have enough data
      validIndex = minPeriods
      signals.iloc[validIndex:] = np.where(
          shortMa.iloc[validIndex:] > longMa.iloc[validIndex:], 1, 0
      )
      
      return signals
  
  def analyzeSignals(self, signals):
      # analyze the generated signals
      
      # count signal changes
      positionChanges = signals['positions'].dropna()
      buySignals = (positionChanges > 0).sum()
      sellSignals = (positionChanges < 0).sum()
      
      # moving average spread statistics
      maSpreadStats = {
          'mean_spread': signals['ma_spread'].mean(),
          'std_spread': signals['ma_spread'].std(),
          'max_spread': signals['ma_spread'].max(),
          'min_spread': signals['ma_spread'].min()
      }
      
      # signal timing analysis
      longPeriods = signals[signals['signal'] == 1]
      shortPeriods = signals[signals['signal'] == 0]
      
      return {
          'total_buy_signals': buySignals,
          'total_sell_signals': sellSignals,
          'long_position_days': len(longPeriods),
          'out_of_market_days': len(shortPeriods),
          'signal_frequency': (buySignals + sellSignals) / len(signals) * 100,
          'ma_spread_statistics': maSpreadStats,
          'average_signal_strength': abs(signals['ma_spread']).mean()
      }
  
  def optimizeParams(self, data, shortRange=(5, 30), longRange=(30, 100), step=5):
      # basic parameter optimization for MA periods
      bestSharpe = -np.inf
      bestParams = {}
      results = []
      
      for shortWin in range(shortRange[0], shortRange[1] + 1, step):
          for longWin in range(longRange[0], longRange[1] + 1, step):
              if shortWin >= longWin:
                  continue
              
              try:
                  # generate signals
                  signals = self.genSigs(data, shortWin, longWin)
                  
                  # calculate basic performance
                  returns = signals['price'].pct_change()
                  strategyReturns = returns * signals['signal'].shift(1)
                  
                  if strategyReturns.std() != 0:
                      sharpe = strategyReturns.mean() / strategyReturns.std() * np.sqrt(252)
                  else:
                      sharpe = 0
                  
                  result = {
                      'short_window': shortWin,
                      'long_window': longWin,
                      'sharpe_ratio': sharpe,
                      'total_return': strategyReturns.sum(),
                      'volatility': strategyReturns.std() * np.sqrt(252)
                  }
                  
                  results.append(result)
                  
                  if sharpe > bestSharpe:
                      bestSharpe = sharpe
                      bestParams = {
                          'shortWin': shortWin,
                          'longWin': longWin,
                          'expected_sharpe': sharpe
                      }
                      
              except Exception:
                  continue
      
      return {
          'best_parameters': bestParams,
          'all_results': pd.DataFrame(results),
          'optimization_summary': {
              'total_combinations': len(results),
              'best_sharpe_ratio': bestSharpe
          }
      }
  
  def getStrategyInfo(self):
      # return strategy information
      return {
          'name': self.name,
          'description': self.description,
          'strategy_type': 'Trend Following',
          'parameters': ['shortWin', 'longWin'],
          'strengths': [
              'simple and intuitive to understand',
              'works well in trending markets',
              'risk management through trend following',
              'low implementation complexity'
          ],
          'weaknesses': [
              'whipsaws in sideways markets',
              'lagging indicator nature',
              'false signals in choppy conditions',
              'requires trending market conditions'
          ],
          'best_market_conditions': [
              'strong trending markets',
              'low noise trading environments',
              'clear bull or bear market phases'
          ]
      }