"""
backtesting engine - does the heavy lifting for trade simulation
probably has some bugs but seems to work
"""

import pandas as pd
import numpy as np

class BtEngine:
  
  def __init__(self, initialCapital=100000, commission=0.001):
      # setup bt engine with some defaults
      self.initialCapital = initialCapital
      self.commission = commission
  
  def runBt(self, signals, regimeData, capital=None):
      # main backtesting function - this is where the magic happens
      if capital is None:
        capital = self.initialCapital
      
      # prepare the data for backtesting
      btData = signals.copy()
      btData['regime'] = regimeData
      
      # initialize portfolio tracking variables
      btData = self._initPortfolioTracking(btData, capital)
      
      # execute trades and track the portfolio - TODO: optimize this loop
      tradeHistory = []
      currentPosition = 0
      availableCash = capital
      
      for i in range(1, len(btData)):
          # get current market state
          currentPrice = btData['price'].iloc[i]
          currentRegime = btData['regime'].iloc[i]
          
          # check for position changes
          if 'positions' in btData.columns:
              positionChange = btData['positions'].iloc[i]
              
              if not pd.isna(positionChange) and positionChange != 0:
                  # execute the trade
                  tradeResult = self._executeTrade(
                      positionChange, currentPrice, currentPosition, availableCash, 
                      btData.index[i], currentRegime
                  )
                  
                  if tradeResult:
                      currentPosition = tradeResult['new_position']
                      availableCash = tradeResult['new_cash']
                      tradeHistory.append(tradeResult['trade_record'])
          
          # update portfolio values
          holdingsValue = currentPosition * currentPrice
          totalValue = availableCash + holdingsValue
          
          btData.loc[btData.index[i], 'holdings'] = holdingsValue
          btData.loc[btData.index[i], 'cash'] = availableCash
          btData.loc[btData.index[i], 'total'] = totalValue
      
      # calculate returns and performance metrics
      btData = self._calculatePortfolioReturns(btData)
      
      return btData, tradeHistory
  
  def _initPortfolioTracking(self, data, initialCapital):
      # setup tracking columns for portfolio
      data['holdings'] = 0.0
      data['cash'] = initialCapital
      data['total'] = initialCapital
      data['returns'] = 0.0
      data['portfolio_returns'] = 0.0
      data['cumulative_returns'] = 1.0
      
      return data
  
  def _executeTrade(self, positionChange, price, currentPos, currentCash, timestamp, regime):
      # execute a single trade with commission costs
      try:
          if positionChange > 0:  # buying
              if currentPos <= 0:  # not currently long
                  # calculate how many shares we can afford
                  maxShares = currentCash / (price * (1 + self.commission))
                  sharesToBuy = int(maxShares)
                  
                  if sharesToBuy > 0:
                      tradeCost = sharesToBuy * price * (1 + self.commission)
                      newCash = currentCash - tradeCost
                      newPosition = currentPos + sharesToBuy
                      
                      return {
                          'new_position': newPosition,
                          'new_cash': newCash,
                          'trade_record': {
                              'date': timestamp,
                              'action': 'BUY',
                              'price': price,
                              'shares': sharesToBuy,
                              'commission': sharesToBuy * price * self.commission,
                              'regime': regime,
                              'portfolio_value': newCash + newPosition * price
                          }
                      }
          
          elif positionChange < 0:  # selling
              if currentPos > 0:  # currently holding shares
                  # sell all shares
                  saleProceeds = currentPos * price * (1 - self.commission)
                  newCash = currentCash + saleProceeds
                  
                  return {
                      'new_position': 0,
                      'new_cash': newCash,
                      'trade_record': {
                          'date': timestamp,
                          'action': 'SELL',
                          'price': price,
                          'shares': currentPos,
                          'commission': currentPos * price * self.commission,
                          'regime': regime,
                          'portfolio_value': newCash
                      }
                  }
          
          return None
          
      except Exception as e:
          print(f"error executing trade: {e}")
          return None
  
  def _calculatePortfolioReturns(self, btData):
      # calculate portfolio returns and benchmark comparison
      
      # portfolio returns
      btData['portfolio_returns'] = btData['total'].pct_change()
      btData['cumulative_returns'] = (1 + btData['portfolio_returns']).cumprod()
      
      # benchmark returns (simple buy and hold)
      btData['benchmark_returns'] = btData['price'].pct_change()
      btData['benchmark_cumulative'] = (1 + btData['benchmark_returns']).cumprod()
      
      # excess returns over benchmark
      btData['excess_returns'] = (
          btData['portfolio_returns'] - btData['benchmark_returns']
      )
      
      return btData
  
  def calculateTradeStatistics(self, trades):
      # calculate detailed statistics about the trades
      if not trades:
          return {'total_trades': 0, 'message': 'no trades executed'}
      
      tradeDf = pd.DataFrame(trades)
      
      # separate buy and sell trades for analysis
      buyTrades = tradeDf[tradeDf['action'] == 'BUY'].reset_index(drop=True)
      sellTrades = tradeDf[tradeDf['action'] == 'SELL'].reset_index(drop=True)
      
      # calculate round-trip trade results
      roundTripTrades = []
      minTradeCount = min(len(buyTrades), len(sellTrades))
      
      for i in range(minTradeCount):
          entryPrice = buyTrades.iloc[i]['price']
          exitPrice = sellTrades.iloc[i]['price']
          shareCount = buyTrades.iloc[i]['shares']
          
          entryCommission = buyTrades.iloc[i]['commission']
          exitCommission = sellTrades.iloc[i]['commission']
          
          pnl = (exitPrice - entryPrice) * shareCount - entryCommission - exitCommission
          returnPct = pnl / (entryPrice * shareCount)
          
          holdingPeriod = (sellTrades.iloc[i]['date'] - buyTrades.iloc[i]['date']).days
          
          roundTripTrades.append({
              'entry_date': buyTrades.iloc[i]['date'],
              'exit_date': sellTrades.iloc[i]['date'],
              'entry_price': entryPrice,
              'exit_price': exitPrice,
              'shares': shareCount,
              'pnl': pnl,
              'return_percent': returnPct,
              'holding_period_days': holdingPeriod,
              'entry_regime': buyTrades.iloc[i]['regime'],
              'exit_regime': sellTrades.iloc[i]['regime']
          })
      
      if roundTripTrades:
          rtDf = pd.DataFrame(roundTripTrades)
          
          # calculate key statistics
          winningTrades = rtDf[rtDf['pnl'] > 0]
          losingTrades = rtDf[rtDf['pnl'] <= 0]
          
          stats = {
              'total_round_trips': len(roundTripTrades),
              'winning_trades': len(winningTrades),
              'losing_trades': len(losingTrades),
              'win_rate': len(winningTrades) / len(roundTripTrades) * 100,
              'total_pnl': rtDf['pnl'].sum(),
              'average_pnl_per_trade': rtDf['pnl'].mean(),
              'average_return_per_trade': rtDf['return_percent'].mean() * 100,
              'best_trade': rtDf['pnl'].max(),
              'worst_trade': rtDf['pnl'].min(),
              'average_holding_period': rtDf['holding_period_days'].mean(),
              'total_commissions_paid': tradeDf['commission'].sum(),
              'profit_factor': abs(winningTrades['pnl'].sum() / losingTrades['pnl'].sum()) if len(losingTrades) > 0 else np.inf
          }
          
          # regime-specific performance
          regimeStats = {}
          for regime in rtDf['entry_regime'].unique():
              regimeTrades = rtDf[rtDf['entry_regime'] == regime]
              if len(regimeTrades) > 0:
                  regimeStats[regime] = {
                      'trade_count': len(regimeTrades),
                      'win_rate': (regimeTrades['pnl'] > 0).mean() * 100,
                      'average_pnl': regimeTrades['pnl'].mean(),
                      'average_return': regimeTrades['return_percent'].mean() * 100
                  }
          
          stats['regime_breakdown'] = regimeStats
          
      else:
          stats = {
              'total_round_trips': 0,
              'message': 'no completed round-trip trades found'
          }
      
      return stats