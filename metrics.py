"""
performance metrics calculation
handles all the math for evaluating trading performance
"""

import pandas as pd
import numpy as np
from scipy import stats

class MetCalc:
  
  def __init__(self):
      self.tradingDaysPerYear = 252
      self.riskFreeRate = 0.02  # assuming 2% risk-free rate
  
  def calcOverall(self, btData, trades):
      # calculate comprehensive performance metrics
      portfolioReturns = btData['portfolio_returns'].dropna()
      
      if len(portfolioReturns) == 0:
          return self._getEmptyMetrics()
      
      # basic return calculations
      totalReturn = (btData['total'].iloc[-1] / btData['total'].iloc[0] - 1) * 100
      annualizedReturn = ((btData['total'].iloc[-1] / btData['total'].iloc[0]) ** 
                         (self.tradingDaysPerYear / len(btData)) - 1) * 100
      
      # risk-adjusted metrics
      volatility = portfolioReturns.std() * np.sqrt(self.tradingDaysPerYear) * 100
      sharpeRatio = self._calculateSharpeRatio(portfolioReturns)
      sortinoRatio = self._calculateSortinoRatio(portfolioReturns)
      
      # drawdown analysis
      maxDrawdown = self._calculateMaxDrawdown(btData['total'])
      calmarRatio = annualizedReturn / abs(maxDrawdown) if maxDrawdown != 0 else 0
      
      # trade-based metrics
      tradeMetrics = self._calculateTradeMetrics(trades)
      
      # additional risk measures
      var95 = self._calculateVaR(portfolioReturns, 0.05)
      cvar95 = self._calculateCVaR(portfolioReturns, 0.05)
      
      # benchmark comparison
      benchmarkMetrics = self._calculateBenchmarkComparison(btData)
      
      return {
          'Total Return (%)': totalReturn,
          'Annualized Return (%)': annualizedReturn,
          'Volatility (%)': volatility,
          'Sharpe Ratio': sharpeRatio,
          'Sortino Ratio': sortinoRatio,
          'Max Drawdown (%)': maxDrawdown,
          'Calmar Ratio': calmarRatio,
          'Win Rate (%)': tradeMetrics['win_rate'],
          'Number of Trades': tradeMetrics['total_trades'],
          'VaR 95% (%)': var95 * 100,
          'CVaR 95% (%)': cvar95 * 100,
          'Information Ratio': benchmarkMetrics['information_ratio'],
          'Beta': benchmarkMetrics['beta'],
          'Alpha (%)': benchmarkMetrics['alpha'] * 100
      }
  
  def calcRegime(self, btData, trades):
      # performance metrics split by market regime
      regimes = ['Bull', 'Bear', 'Sideways']
      regimeMetrics = {}
      
      for regime in regimes:
          regimeData = btData[btData['regime'] == regime]
          regimeTrades = [t for t in trades if t.get('regime') == regime]
          
          if len(regimeData) > 0:
              regimeReturns = regimeData['portfolio_returns'].dropna()
              
              if len(regimeReturns) > 0:
                  # calculate regime-specific metrics
                  totalReturn = regimeReturns.sum() * 100
                  volatility = regimeReturns.std() * np.sqrt(self.tradingDaysPerYear) * 100
                  sharpe = self._calculateSharpeRatio(regimeReturns)
                  
                  # drawdown for this specific regime
                  regimePortfolioValues = regimeData['total']
                  if len(regimePortfolioValues) > 1:
                      regimeMaxDrawdown = self._calculateMaxDrawdown(regimePortfolioValues)
                  else:
                      regimeMaxDrawdown = 0
                  
                  # trade metrics for this regime
                  tradeMetrics = self._calculateTradeMetrics(regimeTrades)
                  
                  regimeMetrics[regime] = {
                      'Total Return (%)': totalReturn,
                      'Sharpe Ratio': sharpe,
                      'Volatility (%)': volatility,
                      'Max Drawdown (%)': regimeMaxDrawdown,
                      'Number of Trades': tradeMetrics['total_trades'],
                      'Win Rate (%)': tradeMetrics['win_rate'],
                      'Days': len(regimeData)
                  }
              else:
                  regimeMetrics[regime] = self._getEmptyRegimeMetrics(len(regimeData))
          else:
              regimeMetrics[regime] = self._getEmptyRegimeMetrics(0)
      
      return regimeMetrics
  
  def _calculateSharpeRatio(self, returns):
      # standard sharpe ratio calculation
      if returns.std() == 0:
          return 0
      
      excessReturns = returns - (self.riskFreeRate / self.tradingDaysPerYear)
      return excessReturns.mean() / returns.std() * np.sqrt(self.tradingDaysPerYear)
  
  def _calculateSortinoRatio(self, returns):
      # sortino ratio using downside deviation
      downsideReturns = returns[returns < 0]
      
      if len(downsideReturns) == 0 or downsideReturns.std() == 0:
          return 0
      
      excessReturns = returns - (self.riskFreeRate / self.tradingDaysPerYear)
      return excessReturns.mean() / downsideReturns.std() * np.sqrt(self.tradingDaysPerYear)
  
  def _calculateMaxDrawdown(self, portfolioValues):
      # calculate maximum drawdown percentage
      runningMax = portfolioValues.expanding(min_periods=1).max()
      drawdown = (portfolioValues - runningMax) / runningMax * 100
      return drawdown.min()
  
  def _calculateVaR(self, returns, confidenceLevel):
      # value at risk calculation
      return returns.quantile(confidenceLevel)
  
  def _calculateCVaR(self, returns, confidenceLevel):
      # conditional value at risk (expected shortfall)
      var = self._calculateVaR(returns, confidenceLevel)
      return returns[returns <= var].mean()
  
  def _calculateTradeMetrics(self, trades):
      # calculate trade-based performance metrics
      if not trades:
          return {'total_trades': 0, 'win_rate': 0}
      
      tradeDf = pd.DataFrame(trades)
      
      # basic trade counting
      totalTrades = len(trades)
      
      # for win rate, need to pair buy/sell trades
      buyTrades = tradeDf[tradeDf['action'] == 'BUY'].reset_index(drop=True)
      sellTrades = tradeDf[tradeDf['action'] == 'SELL'].reset_index(drop=True)
      
      if len(buyTrades) > 0 and len(sellTrades) > 0:
          minTrades = min(len(buyTrades), len(sellTrades))
          profitLoss = []
          
          for i in range(minTrades):
              profit = (sellTrades.iloc[i]['price'] - buyTrades.iloc[i]['price']) * buyTrades.iloc[i]['shares']
              profitLoss.append(profit)
          
          winRate = (np.array(profitLoss) > 0).mean() * 100 if profitLoss else 0
      else:
          winRate = 0
      
      return {
          'total_trades': totalTrades,
          'win_rate': winRate
      }
  
  def _calculateBenchmarkComparison(self, btData):
      # compare strategy performance vs benchmark
      if 'benchmark_returns' not in btData.columns:
          return {'information_ratio': 0, 'beta': 0, 'alpha': 0}
      
      portfolioReturns = btData['portfolio_returns'].dropna()
      benchmarkReturns = btData['benchmark_returns'].dropna()
      
      # align the return series
      alignedData = pd.concat([portfolioReturns, benchmarkReturns], axis=1).dropna()
      
      if len(alignedData) < 2:
          return {'information_ratio': 0, 'beta': 0, 'alpha': 0}
      
      portRet = alignedData.iloc[:, 0]
      benchRet = alignedData.iloc[:, 1]
      
      # excess returns for information ratio
      excessReturns = portRet - benchRet
      
      # information ratio calculation
      if excessReturns.std() != 0:
          informationRatio = excessReturns.mean() / excessReturns.std() * np.sqrt(self.tradingDaysPerYear)
      else:
          informationRatio = 0
      
      # beta calculation using covariance
      if benchRet.std() != 0:
          covariance = np.cov(portRet, benchRet)[0, 1]
          beta = covariance / benchRet.var()
      else:
          beta = 0
      
      # alpha calculation using CAPM
      riskFreeDailyRate = self.riskFreeRate / self.tradingDaysPerYear
      alpha = portRet.mean() - (riskFreeDailyRate + beta * (benchRet.mean() - riskFreeDailyRate))
      
      return {
          'information_ratio': informationRatio,
          'beta': beta,
          'alpha': alpha
      }
  
  def _getEmptyMetrics(self):
      # return empty metrics when no data available
      return {
          'Total Return (%)': 0,
          'Annualized Return (%)': 0,
          'Volatility (%)': 0,
          'Sharpe Ratio': 0,
          'Sortino Ratio': 0,
          'Max Drawdown (%)': 0,
          'Calmar Ratio': 0,
          'Win Rate (%)': 0,
          'Number of Trades': 0,
          'VaR 95% (%)': 0,
          'CVaR 95% (%)': 0,
          'Information Ratio': 0,
          'Beta': 0,
          'Alpha (%)': 0
      }
  
  def _getEmptyRegimeMetrics(self, days):
      # empty regime metrics template
      return {
          'Total Return (%)': 0,
          'Sharpe Ratio': 0,
          'Volatility (%)': 0,
          'Max Drawdown (%)': 0,
          'Number of Trades': 0,
          'Win Rate (%)': 0,
          'Days': days
      }