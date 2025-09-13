"""
pairs trading strategy - market neutral mean reversion
works when it works, doesnt when it doesnt
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from scipy import stats

class PairsTrade:
  
  def __init__(self):
      self.name = "Pairs Trading Strategy"
      self.description = """
      market neutral strategy exploiting price divergences between correlated stocks.
      uses cointegration analysis and mean reversion on price spreads.
      """
  
  def genSigs(self, data1, data2, pairsWin=30, entryZ=2.0, exitZ=0.5):
      # generate pairs trading signals using cointegration
      
      # parameter validation
      if pairsWin <= 0:
          raise ValueError("pairs window must be positive")
      
      if entryZ <= exitZ:
          raise ValueError("entry z-score must be greater than exit z-score")
      
      # align the two datasets
      alignedData = pd.concat([data1['Close'], data2['Close']], axis=1, keys=['stock1', 'stock2']).dropna()
      
      if len(alignedData) < pairsWin * 2:
          raise ValueError(f"need at least {pairsWin * 2} aligned data points")
      
      # initialize signals dataframe
      signals = pd.DataFrame(index=alignedData.index)
      signals['stock1'] = alignedData['stock1']
      signals['stock2'] = alignedData['stock2']
      signals['price'] = alignedData['stock1']  # use first stock for portfolio calculations
      
      # calculate price ratio and spread
      signals['price_ratio'] = alignedData['stock1'] / alignedData['stock2']
      signals['log_ratio'] = np.log(signals['price_ratio'])
      signals['spread'] = alignedData['stock1'] - alignedData['stock2']
      
      # calculate rolling statistics for mean reversion
      signals['spread_mean'] = signals['spread'].rolling(pairsWin).mean()
      signals['spread_std'] = signals['spread'].rolling(pairsWin).std()
      signals['zscore'] = (signals['spread'] - signals['spread_mean']) / signals['spread_std']
      
      # calculate additional cointegration metrics
      signals['correlation'] = self._calcRollingCorr(
          alignedData['stock1'], alignedData['stock2'], pairsWin
      )
      signals['coint_pvalue'] = self._calcRollingCoint(
          alignedData, pairsWin
      )
      
      # generate trading signals
      signals['signal'] = self._genPairsSignals(
          signals['zscore'], entryZ, exitZ
      )
      
      # calculate position changes
      signals['positions'] = signals['signal'].diff()
      
      # add signal strength and confidence indicators
      signals['signal_strength'] = self._calcSignalStrength(
          signals['zscore'], entryZ, exitZ
      )
      signals['mean_reversion_probability'] = self._calcMeanReversionProb(
          signals['zscore'], signals['correlation']
      )
      
      return signals
  
  def _calcRollingCorr(self, series1, series2, window):
      # calculate rolling correlation between the two series
      return series1.rolling(window).corr(series2)
  
  def _calcRollingCoint(self, alignedData, window):
      # calculate rolling cointegration p-value
      pvalues = []
      
      for i in range(len(alignedData)):
          if i < window:
              pvalues.append(np.nan)
          else:
              # get window of data
              windowData = alignedData.iloc[i-window:i]
              
              try:
                  # perform cointegration test
                  _, pvalue, _ = coint(windowData['stock1'], windowData['stock2'])
                  pvalues.append(pvalue)
              except:
                  pvalues.append(np.nan)
      
      return pd.Series(pvalues, index=alignedData.index)
  
  def _genPairsSignals(self, zscore, entryZ, exitZ):
      # generate pairs signals based on z-score thresholds
      signals = pd.Series(0, index=zscore.index)
      position = 0
      
      for i in range(len(zscore)):
          currentZ = zscore.iloc[i]
          
          if pd.isna(currentZ):
              signals.iloc[i] = position
              continue
          
          # entry conditions
          if position == 0:
              if currentZ > entryZ:
                  position = -1  # short the spread (short stock1, long stock2)
              elif currentZ < -entryZ:
                  position = 1   # long the spread (long stock1, short stock2)
          
          # exit conditions
          elif position != 0:
              if abs(currentZ) < exitZ:
                  position = 0   # exit position
              # also exit if z-score reverses strongly
              elif (position == 1 and currentZ > entryZ) or \
                   (position == -1 and currentZ < -entryZ):
                  position = 0
          
          signals.iloc[i] = position
      
      return signals
  
  def _calcSignalStrength(self, zscore, entryZ, exitZ):
      # calculate signal strength based on z-score magnitude
      absZ = abs(zscore)
      strength = pd.Series(0.0, index=zscore.index)
      
      # linear scaling between exit and entry thresholds
      mask = absZ >= exitZ
      strength[mask] = np.minimum(
          (absZ[mask] - exitZ) / (entryZ - exitZ), 1.0
      )
      
      return strength
  
  def _calcMeanReversionProb(self, zscore, correlation):
      # calculate probability of mean reversion
      
      # higher correlation increases mean reversion probability
      # higher z-score magnitude increases mean reversion probability
      
      corrFactor = abs(correlation).fillna(0)
      zFactor = 1 - np.exp(-abs(zscore) / 2)  # saturating function
      
      prob = (corrFactor * zFactor).clip(0, 1)
      return prob
  
  def analyzeSpreadCharacteristics(self, signals):
      # analyze spread characteristics
      spread = signals['spread'].dropna()
      zscore = signals['zscore'].dropna()
      
      # test for stationarity (simplified approach)
      spreadChanges = spread.diff().dropna()
      
      return {
          'spread_statistics': {
              'mean': spread.mean(),
              'std': spread.std(),
              'minimum': spread.min(),
              'maximum': spread.max(),
              'skewness': spread.skew(),
              'kurtosis': spread.kurtosis()
          },
          'zscore_statistics': {
              'mean': zscore.mean(),
              'std': zscore.std(),
              'extreme_positive': (zscore > 3).sum(),
              'extreme_negative': (zscore < -3).sum(),
              'mean_reversion_rate': self._calcMeanReversionRate(zscore)
          },
          'stationarity_indicators': {
              'spread_volatility': spread.std(),
              'spread_change_volatility': spreadChanges.std(),
              'autocorrelation_lag1': spread.autocorr(lag=1)
          }
      }
  
  def _calcMeanReversionRate(self, zscore, threshold=1.0):
      # calculate rate at which extreme z-scores revert to mean
      extremeEvents = abs(zscore) > threshold
      if not extremeEvents.any():
          return 0
      
      reversionCount = 0
      totalExtreme = 0
      
      for i in range(1, len(zscore)):
          if extremeEvents.iloc[i-1]:  # previous was extreme
              totalExtreme += 1
              if abs(zscore.iloc[i]) < abs(zscore.iloc[i-1]):  # reverted
                  reversionCount += 1
      
      return reversionCount / totalExtreme if totalExtreme > 0 else 0
  
  def calcHedgeRatios(self, data1, data2, method='ols', window=30):
      # calculate hedge ratios for the pairs trade
      alignedData = pd.concat([data1['Close'], data2['Close']], axis=1, keys=['stock1', 'stock2']).dropna()
      
      hedgeRatios = []
      
      for i in range(len(alignedData)):
          if i < window:
              hedgeRatios.append(np.nan)
          else:
              windowData = alignedData.iloc[i-window:i]
              
              if method == 'ols':
                  # ordinary least squares
                  x = windowData['stock2'].values
                  y = windowData['stock1'].values
                  
                  # add constant term
                  X = np.column_stack([np.ones(len(x)), x])
                  
                  try:
                      coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
                      hedgeRatio = coefficients[1]  # slope coefficient
                  except:
                      hedgeRatio = np.nan
                      
              elif method == 'tls':
                  # total least squares
                  try:
                      corrMatrix = np.corrcoef(windowData['stock1'], windowData['stock2'])
                      hedgeRatio = corrMatrix[0, 1] * (
                          windowData['stock1'].std() / windowData['stock2'].std()
                      )
                  except:
                      hedgeRatio = np.nan
              
              hedgeRatios.append(hedgeRatio)
      
      return pd.Series(hedgeRatios, index=alignedData.index)
  
  def optimizeParams(self, data1, data2, winRange=(20, 60), entryRange=(1.5, 3.0), exitRange=(0.1, 1.0)):
      # optimize pairs trading parameters
      bestSharpe = -np.inf
      bestParams = {}
      results = []
      
      windows = range(winRange[0], winRange[1] + 1, 5)
      entryZs = np.arange(entryRange[0], entryRange[1] + 0.1, 0.1)
      exitZs = np.arange(exitRange[0], exitRange[1] + 0.1, 0.1)
      
      for pairsWin in windows:
          for entryZ in entryZs:
              for exitZ in exitZs:
                  if entryZ <= exitZ:
                      continue
                  
                  try:
                      # generate signals
                      signals = self.genSigs(
                          data1, data2, pairsWin, entryZ, exitZ
                      )
                      
                      # calculate performance (simplified pairs trading returns)
                      # in reality would need to account for both legs properly
                      returns = signals['price'].pct_change()
                      strategyReturns = returns * signals['signal'].shift(1)
                      
                      if strategyReturns.std() != 0:
                          sharpe = strategyReturns.mean() / strategyReturns.std() * np.sqrt(252)
                      else:
                          sharpe = 0
                      
                      # count signals generated
                      totalSignals = abs(signals['positions'].dropna()).sum()
                      
                      result = {
                          'pairs_window': pairsWin,
                          'entry_zscore': entryZ,
                          'exit_zscore': exitZ,
                          'sharpe_ratio': sharpe,
                          'total_return': strategyReturns.sum(),
                          'volatility': strategyReturns.std() * np.sqrt(252),
                          'total_signals': totalSignals
                      }
                      
                      results.append(result)
                      
                      if sharpe > bestSharpe and totalSignals > 5:
                          bestSharpe = sharpe
                          bestParams = {
                              'pairsWin': pairsWin,
                              'entryZ': entryZ,
                              'exitZ': exitZ,
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
          'strategy_type': 'Market Neutral/Statistical Arbitrage',
          'parameters': ['pairsWin', 'entryZ', 'exitZ'],
          'strengths': [
              'market neutral in theory',
              'profits from relative price movements',
              'lower correlation to market direction',
              'statistical edge from mean reversion'
          ],
          'weaknesses': [
              'requires strong cointegration relationship',
              'suffers during structural breaks',
              'transaction costs can be significant',
              'needs careful risk management'
          ],
          'best_market_conditions': [
              'stable correlation between asset pairs',
              'mean reverting spread behavior',
              'low transaction cost environments',
              'markets with statistical arbitrage opportunities'
          ]
      }