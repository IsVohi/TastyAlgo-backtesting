"""
momentum strategy implementation
buy high sell higher... or get burned
"""

import pandas as pd
import numpy as np

class MomStrat:
  
  def __init__(self):
      self.name = "Momentum Strategy"
      self.description = """
      momentum strategy that buys strong performers and sells weak ones.
      uses n-day returns vs configurable thresholds for signal generation.
      """
  
  def genSigs(self, data, momWin=14, buyThresh=0.02, sellThresh=-0.02):
      # generate signals based on momentum
      
      # parameter validation
      if momWin <= 0:
          raise ValueError("momentum window must be positive")
      
      if buyThresh <= sellThresh:
          raise ValueError("buy threshold must be greater than sell threshold")
      
      if len(data) < momWin + 1:
          raise ValueError(f"need at least {momWin + 1} data points")
      
      # initialize signals dataframe
      signals = pd.DataFrame(index=data.index)
      signals['price'] = data['Close']
      
      # calculate momentum indicators
      signals['returns_1d'] = data['Close'].pct_change()
      signals['returns_nd'] = data['Close'].pct_change(momWin)
      signals['momentum'] = signals['returns_nd']  # primary momentum signal
      
      # calculate additional momentum indicators for analysis
      signals['rsi'] = self._calcRSI(data['Close'], momWin)
      signals['price_velocity'] = self._calcPriceVelocity(data['Close'], momWin)
      signals['momentum_ma'] = signals['momentum'].rolling(5).mean()
      
      # generate trading signals
      signals['signal'] = self._genMomentumSignals(
          signals['momentum'], buyThresh, sellThresh
      )
      
      # calculate position changes
      signals['positions'] = signals['signal'].diff()
      
      # add signal strength indicator
      signals['signal_strength'] = self._calcSignalStrength(
          signals['momentum'], buyThresh, sellThresh
      )
      
      return signals
  
  def _calcRSI(self, prices, window=14):
      # calculate relative strength index
      delta = prices.diff()
      gains = delta.where(delta > 0, 0).rolling(window=window).mean()
      losses = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
      
      rs = gains / losses
      rsi = 100 - (100 / (1 + rs))
      
      return rsi.fillna(50)  # neutral RSI for missing values
  
  def _calcPriceVelocity(self, prices, window):
      # calculate price velocity (average price change)
      returns = prices.pct_change()
      velocity = returns.rolling(window).mean()
      return velocity
  
  def _genMomentumSignals(self, momentum, buyThresh, sellThresh):
      # generate momentum-based signals
      signals = pd.Series(0, index=momentum.index)
      
      # generate signals based on momentum thresholds
      signals.loc[momentum > buyThresh] = 1   # buy signal
      signals.loc[momentum < sellThresh] = -1  # sell signal
      # everything else remains neutral (0)
      
      return signals
  
  def _calcSignalStrength(self, momentum, buyThresh, sellThresh):
      # calculate signal strength based on momentum magnitude
      strength = pd.Series(0.0, index=momentum.index)
      
      # buy signal strength
      buyMask = momentum > buyThresh
      strength[buyMask] = np.minimum(
          (momentum[buyMask] - buyThresh) / (buyThresh * 2), 1.0
      )
      
      # sell signal strength
      sellMask = momentum < sellThresh
      strength[sellMask] = np.minimum(
          abs(momentum[sellMask] - sellThresh) / abs(sellThresh * 2), 1.0
      )
      
      return strength
  
  def analyzeMomentumDistribution(self, signals):
      # analyze momentum distribution characteristics
      momentum = signals['momentum'].dropna()
      
      return {
          'mean_momentum': momentum.mean(),
          'std_momentum': momentum.std(),
          'skewness': momentum.skew(),
          'kurtosis': momentum.kurtosis(),
          'percentiles': {
              '5th': momentum.quantile(0.05),
              '25th': momentum.quantile(0.25),
              '50th': momentum.quantile(0.50),
              '75th': momentum.quantile(0.75),
              '95th': momentum.quantile(0.95)
          },
          'positive_momentum_ratio': (momentum > 0).mean(),
          'extreme_events': {
              'positive_extreme': (momentum > momentum.quantile(0.95)).sum(),
              'negative_extreme': (momentum < momentum.quantile(0.05)).sum()
          }
      }
  
  def calcMomentumPersistence(self, signals, lookAheadPeriods=5):
      # calculate momentum persistence analysis
      momentum = signals['momentum']
      persistenceData = []
      
      for i in range(len(momentum) - lookAheadPeriods):
          currentMomentum = momentum.iloc[i]
          if abs(currentMomentum) > 0.01:  # only analyze significant momentum
              futureMomentum = momentum.iloc[i+1:i+lookAheadPeriods+1]
              sameDirection = ((currentMomentum > 0) == (futureMomentum > 0)).mean()
              persistenceData.append({
                  'initial_momentum': currentMomentum,
                  'persistence_rate': sameDirection,
                  'average_future_momentum': futureMomentum.mean()
              })
      
      if not persistenceData:
          return {'overall_persistence': 0, 'analysis': 'insufficient momentum data'}
      
      persistenceDf = pd.DataFrame(persistenceData)
      
      return {
          'overall_persistence': persistenceDf['persistence_rate'].mean(),
          'strong_momentum_persistence': persistenceDf[
              abs(persistenceDf['initial_momentum']) > 0.05
          ]['persistence_rate'].mean(),
          'weak_momentum_persistence': persistenceDf[
              abs(persistenceDf['initial_momentum']) <= 0.02
          ]['persistence_rate'].mean(),
          'momentum_decay_rate': 1 - persistenceDf['persistence_rate'].mean()
      }
  
  def optimizeThresholds(self, data, momWin=14, threshRange=(-0.1, 0.1), step=0.005):
      # optimize buy and sell thresholds
      bestSharpe = -np.inf
      bestParams = {}
      results = []
      
      thresholds = np.arange(threshRange[0], threshRange[1] + step, step)
      
      for sellThresh in thresholds:
          for buyThresh in thresholds:
              if buyThresh <= sellThresh:
                  continue
              
              try:
                  # generate signals
                  signals = self.genSigs(data, momWin, buyThresh, sellThresh)
                  
                  # calculate performance
                  returns = signals['price'].pct_change()
                  strategyReturns = returns * signals['signal'].shift(1)
                  
                  if strategyReturns.std() != 0:
                      sharpe = strategyReturns.mean() / strategyReturns.std() * np.sqrt(252)
                  else:
                      sharpe = 0
                  
                  # count signals generated
                  totalSignals = abs(signals['positions'].dropna()).sum()
                  
                  result = {
                      'buy_threshold': buyThresh,
                      'sell_threshold': sellThresh,
                      'sharpe_ratio': sharpe,
                      'total_return': strategyReturns.sum(),
                      'volatility': strategyReturns.std() * np.sqrt(252),
                      'total_signals': totalSignals,
                      'signal_frequency': totalSignals / len(signals) * 100
                  }
                  
                  results.append(result)
                  
                  if sharpe > bestSharpe and totalSignals > 5:  # minimum signal requirement
                      bestSharpe = sharpe
                      bestParams = {
                          'momWin': momWin,
                          'buyThresh': buyThresh,
                          'sellThresh': sellThresh,
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
          'strategy_type': 'Momentum/Trend Following',
          'parameters': ['momWin', 'buyThresh', 'sellThresh'],
          'strengths': [
              'captures strong price movements effectively',
              'responsive to market changes',
              'works well in trending markets',
              'configurable sensitivity levels'
          ],
          'weaknesses': [
              'generates false signals in choppy markets',
              'may miss trend reversals',
              'requires careful threshold calibration',
              'performance depends on market volatility'
          ],
          'best_market_conditions': [
              'trending markets with clear direction',
              'high volatility environments',
              'markets with momentum persistence',
              'crisis or euphoria periods'
          ]
      }