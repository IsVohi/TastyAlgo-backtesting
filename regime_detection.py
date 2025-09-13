"""
regime detection - tries to figure out if market is bull/bear/sideways
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class RegDet:
  
  def __init__(self):
      self.regimeColors = {'Bull': 'green', 'Bear': 'red', 'Sideways': 'yellow'}
  
  def detectStat(self, data, window=20):
      # statistical method - pretty simple but works
      returns = data['Close'].pct_change()
      rollingMean = returns.rolling(window).mean()
      rollingStd = returns.rolling(window).std()
      
      regimes = []
      for i in range(len(data)):
          if i < window:
              regimes.append("Neutral")  # not enough data yet
          else:
              meanRet = rollingMean.iloc[i]
              stdRet = rollingStd.iloc[i]
              
              # simple classification logic
              if meanRet > stdRet * 0.5:
                  regimes.append("Bull")
              elif meanRet < -stdRet * 0.5:
                  regimes.append("Bear")
              else:
                  regimes.append("Sideways")
      
      return regimes
  
  def detectKmeans(self, data, numClusters=3, window=20):
      # k-means clustering approach - more sophisticated
      
      # calculate features for clustering
      returns = data['Close'].pct_change().fillna(0)
      volatility = returns.rolling(window).std().fillna(0)
      
      # add more features to make it better
      rsiVals = self._calculateRSI(data['Close'], window)
      volumeRatio = (data['Volume'] / data['Volume'].rolling(window).mean()).fillna(1)
      
      # build feature matrix
      features = np.column_stack([returns, volatility, rsiVals, volumeRatio])
      
      # run kmeans
      kmeans = KMeans(n_clusters=numClusters, random_state=42, n_init=10)
      clusterLabels = kmeans.fit_predict(features)
      
      # map clusters to regime names based on average returns
      clusterReturns = {}
      for clusterId in range(numClusters):
          clusterMask = clusterLabels == clusterId  
          clusterReturns[clusterId] = returns[clusterMask].mean()
      
      # sort clusters by return and assign names
      sortedClusters = sorted(clusterReturns.items(), key=lambda x: x[1])
      regimeMapping = {
          sortedClusters[0][0]: "Bear",     # worst returns
          sortedClusters[1][0]: "Sideways", # middle returns  
          sortedClusters[2][0]: "Bull"      # best returns
      }
      
      regimes = [regimeMapping[cluster] for cluster in clusterLabels]
      return regimes
  
  def _calculateRSI(self, prices, window=14):
      # relative strength index calculation
      delta = prices.diff()
      gains = (delta.where(delta > 0, 0)).rolling(window=window).mean()
      losses = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
      
      rs = gains / losses
      rsi = 100 - (100 / (1 + rs))
      
      # normalize for clustering (0-1 range)
      return (rsi / 100).fillna(0.5)
  
  def analyzeTransitions(self, regimes):
      # analyze how regimes change over time
      regimeSeries = pd.Series(regimes)
      transitions = {}
      
      # count total regime changes
      totalChanges = (regimeSeries != regimeSeries.shift()).sum()
      
      # calculate average duration for each regime
      regimeDurations = {}
      currentRegime = None
      currentDuration = 0
      
      for regime in regimes:
          if regime != currentRegime:
              if currentRegime is not None:
                  if currentRegime not in regimeDurations:
                      regimeDurations[currentRegime] = []
                  regimeDurations[currentRegime].append(currentDuration)
              currentRegime = regime
              currentDuration = 1
          else:
              currentDuration += 1
      
      # don't forget the last regime
      if currentRegime is not None:
          if currentRegime not in regimeDurations:
              regimeDurations[currentRegime] = []
          regimeDurations[currentRegime].append(currentDuration)
      
      # calculate average durations
      avgDurations = {}
      for regime, durations in regimeDurations.items():
          avgDurations[regime] = np.mean(durations)
      
      return {
          'total_transitions': totalChanges,
          'average_durations': avgDurations,
          'regime_counts': regimeSeries.value_counts().to_dict()
      }
  
  def getRegimeStats(self, regimes, returns):
      # get statistics for each regime
      regimeStats = {}
      regimeSeries = pd.Series(regimes, index=returns.index)
      
      for regime in ['Bull', 'Bear', 'Sideways']:
          regimeMask = regimeSeries == regime
          regimeReturns = returns[regimeMask]
          
          if len(regimeReturns) > 0:
              regimeStats[regime] = {
                  'mean_return': regimeReturns.mean(),
                  'std_return': regimeReturns.std(),
                  'sharpe_ratio': regimeReturns.mean() / regimeReturns.std() if regimeReturns.std() != 0 else 0,
                  'skewness': regimeReturns.skew(),
                  'kurtosis': regimeReturns.kurtosis(),
                  'num_periods': len(regimeReturns)
              }
          else:
              # no data for this regime
              regimeStats[regime] = {
                  'mean_return': 0, 'std_return': 0, 'sharpe_ratio': 0,
                  'skewness': 0, 'kurtosis': 0, 'num_periods': 0
              }
      
      return regimeStats