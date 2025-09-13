"""
volatility breakout strategy
trades vol spikes - sometimes works, sometimes doesnt
"""

import pandas as pd
import numpy as np

class VolBreak:
  
  def __init__(self):
      self.name = "Volatility Breakout Strategy"
      self.description = """
      volatility breakout strategy that trades when volatility exceeds historical norms.
      the idea is that vol spikes often precede significant price moves.
      """
  
  def genSigs(self, data, volWin=20, volMult=2.0):
      # generate signals based on volatility breakouts
      
      # parameter validation
      if volWin <= 0:
          raise ValueError("volatility window must be positive")
      
      if volMult <= 0:
          raise ValueError("volatility multiplier must be positive")
      
      if len(data) < volWin * 2:
          raise ValueError(f"need at least {volWin * 2} data points")
      
      # initialize signals dataframe
      signals = pd.DataFrame(index=data.index)
      signals['price'] = data['Close']
      
      # calculate returns and volatility metrics
      signals['returns'] = data['Close'].pct_change()
      signals['high_low_ratio'] = (data['High'] - data['Low']) / data['Close']
      
      # calculate different volatility estimators
      signals['return_vol'] = self._calcReturnVol(signals['returns'], volWin)
      signals['parkinson_vol'] = self._calcParkinsonVol(data, volWin)
      signals['gk_vol'] = self._calcGKVol(data, volWin)
      
      # use return volatility as primary signal
      signals['volatility'] = signals['return_vol']
      
      # calculate volatility threshold and regime
      signals['vol_threshold'] = self._calcVolThreshold(
          signals['volatility'], volWin, volMult
      )
      signals['vol_regime'] = signals['volatility'] / signals['vol_threshold']
      
      # generate trading signals
      signals['signal'] = self._genVolatilitySignals(
          signals['volatility'], signals['vol_threshold']
      )
      
      # calculate position changes
      signals['positions'] = signals['signal'].diff()
      
      # add analytical indicators
      signals['vol_percentile'] = signals['volatility'].rolling(volWin * 2).rank(pct=True)
      signals['vol_zscore'] = self._calcVolZscore(signals['volatility'], volWin)
      
      return signals
  
  def _calcReturnVol(self, returns, window):
      # calculate rolling volatility from returns
      return returns.rolling(window).std()
  
  def _calcParkinsonVol(self, data, window):
      # parkinson volatility estimator using high-low prices
      
      # parkinson vol: sqrt(1/(4*ln(2)) * (ln(H/L))^2)
      hlRatio = np.log(data['High'] / data['Low'])
      parkinsonVar = (hlRatio ** 2).rolling(window).mean() / (4 * np.log(2))
      return np.sqrt(parkinsonVar)
  
  def _calcGKVol(self, data, window):
      # garman-klass volatility estimator
      
      # more sophisticated vol estimator using OHLC
      logHL = np.log(data['High'] / data['Low'])
      logCO = np.log(data['Close'] / data['Open'])
      
      gkVar = (0.5 * logHL ** 2 - (2 * np.log(2) - 1) * logCO ** 2).rolling(window).mean()
      return np.sqrt(gkVar)
  
  def _calcVolThreshold(self, volatility, window, multiplier):
      # calculate dynamic volatility threshold
      
      # use rolling mean as baseline
      volBaseline = volatility.rolling(window).mean()
      return volBaseline * multiplier
  
  def _calcVolZscore(self, volatility, window):
      # calculate volatility z-score for normalization
      volMean = volatility.rolling(window).mean()
      volStd = volatility.rolling(window).std()
      
      return (volatility - volMean) / volStd
  
  def _genVolatilitySignals(self, volatility, threshold):
      # generate signals based on volatility breakout
      signals = pd.Series(0, index=volatility.index)
      
      # buy signal when volatility exceeds threshold
      signals.loc[volatility > threshold] = 1
      
      return signals
  
  def analyzeVolatilityClusters(self, signals, minClusterSize=3):
      # analyze volatility clustering patterns
      highVolPeriods = signals['signal'] == 1
      
      # find clusters of high volatility
      clusters = []
      currentCluster = []
      
      for i, isHighVol in enumerate(highVolPeriods):
          if isHighVol:
              currentCluster.append(i)
          else:
              if len(currentCluster) >= minClusterSize:
                  clusters.append(currentCluster)
              currentCluster = []
      
      # don't forget the last cluster
      if len(currentCluster) >= minClusterSize:
          clusters.append(currentCluster)
      
      if not clusters:
          return {'cluster_count': 0, 'analysis': 'no significant volatility clusters found'}
      
      clusterLengths = [len(cluster) for cluster in clusters]
      
      return {
          'cluster_count': len(clusters),
          'average_cluster_length': np.mean(clusterLengths),
          'max_cluster_length': max(clusterLengths),
          'min_cluster_length': min(clusterLengths),
          'total_high_vol_days': sum(clusterLengths),
          'clustering_ratio': len(clusters) / sum(clusterLengths) if sum(clusterLengths) > 0 else 0
      }
  
  def calcVolRegimeTransitions(self, signals):
      # analyze volatility regime transitions
      
      # define volatility regimes based on percentiles
      volPercentiles = signals['volatility'].quantile([0.33, 0.67])
      
      def classifyVolRegime(vol):
          if vol <= volPercentiles.iloc[0]:
              return 'Low'
          elif vol <= volPercentiles.iloc[1]:
              return 'Medium'
          else:
              return 'High'
      
      volRegimes = signals['volatility'].apply(classifyVolRegime)
      regimeChanges = (volRegimes != volRegimes.shift()).sum()
      
      # calculate transition matrix
      transitions = {}
      for i in range(1, len(volRegimes)):
          fromRegime = volRegimes.iloc[i-1]
          toRegime = volRegimes.iloc[i]
          
          if fromRegime != toRegime:
              transKey = f"{fromRegime}_to_{toRegime}"
              transitions[transKey] = transitions.get(transKey, 0) + 1
      
      return {
          'total_regime_changes': regimeChanges,
          'regime_transitions': transitions,
          'regime_persistence': {
              regime: (volRegimes == regime).sum() for regime in ['Low', 'Medium', 'High']
          }
      }
  
  def optimizeParams(self, data, volWinRange=(10, 50), multRange=(1.0, 4.0), stepWin=5, stepMult=0.2):
      # optimize volatility breakout parameters
      bestSharpe = -np.inf
      bestParams = {}
      results = []
      
      windows = range(volWinRange[0], volWinRange[1] + 1, stepWin)
      multipliers = np.arange(multRange[0], multRange[1] + stepMult, stepMult)
      
      for volWin in windows:
          for volMult in multipliers:
              try:
                  # generate signals
                  signals = self.genSigs(data, volWin, volMult)
                  
                  # calculate performance
                  returns = signals['price'].pct_change()
                  strategyReturns = returns * signals['signal'].shift(1)
                  
                  if strategyReturns.std() != 0:
                      sharpe = strategyReturns.mean() / strategyReturns.std() * np.sqrt(252)
                  else:
                      sharpe = 0
                  
                  # calculate signal frequency
                  totalSignals = (signals['signal'] == 1).sum()
                  signalFreq = totalSignals / len(signals) * 100
                  
                  result = {
                      'vol_window': volWin,
                      'vol_multiplier': volMult,
                      'sharpe_ratio': sharpe,
                      'total_return': strategyReturns.sum(),
                      'volatility': strategyReturns.std() * np.sqrt(252),
                      'signal_frequency': signalFreq,
                      'total_signals': totalSignals
                  }
                  
                  results.append(result)
                  
                  if sharpe > bestSharpe and totalSignals > 10:  # minimum signal requirement
                      bestSharpe = sharpe
                      bestParams = {
                          'volWin': volWin,
                          'volMult': volMult,
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
          'strategy_type': 'Volatility/Breakout',
          'parameters': ['volWin', 'volMult'],
          'strengths': [
              'captures market stress periods effectively',
              'works during regime transitions',
              'profits from volatility expansion',
              'responsive to changing market conditions'
          ],
          'weaknesses': [
              'false signals during normal volatility',
              'requires careful multiplier calibration',
              'whipsawed in choppy but low-vol markets',
              'performance varies significantly by regime'
          ],
          'best_market_conditions': [
              'market transition periods',
              'crisis or high uncertainty environments',
              'earnings announcement periods',
              'periods of significant regime change'
          ]
      }