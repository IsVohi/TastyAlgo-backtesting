"""
data fetcher for yahoo finance - works most of the time
"""

import yfinance as yf
import streamlit as st
import pandas as pd

class DataGet:
  
  def __init__(self):
    self.cacheTime = 3600  # 1hr should be enough
  
  @st.cache_data(ttl=3600)
  def getData(_self, ticker, startDate, endDate):
    # get data from yahoo - sometimes fails but whatever
    try:
        tickerObj = yf.Ticker(ticker)
        stockData = tickerObj.history(start=startDate, end=endDate)
        
        if stockData.empty:
          raise ValueError(f"no data found for {ticker}")
            
        # check if enough data points
        if len(stockData) < 50:
          st.warning(f"only got {len(stockData)} days for {ticker} - might not be enough")
        
        # handle missing data points
        missingCount = stockData.isnull().sum().sum()
        if missingCount > 0:
          st.info(f"found {missingCount} missing data points for {ticker} - filling forward")
          stockData = stockData.fillna(method='ffill')
        
        return stockData
        
    except Exception as e:
        st.error(f"failed to get data for {ticker}: {str(e)}")
        return None
  
  def validateData(self, data, ticker):
    # basic validation - could be more thorough
    if data is None or data.empty:
        st.error(f"no data available for {ticker}")
        return False
    
    requiredCols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missingCols = [col for col in requiredCols if col not in data.columns]
    
    if missingCols:
        st.error(f"missing required columns for {ticker}: {missingCols}")
        return False
    
    # check for reasonable prices
    if (data['Close'] <= 0).any():
        st.error(f"found invalid price data for {ticker}")
        return False
    
    return True
  
  def getDataSummary(self, data):
    # return summary info about the data
    if data is None or data.empty:
        return {}
    
    startDate = data.index[0].strftime('%Y-%m-%d')
    endDate = data.index[-1].strftime('%Y-%m-%d')
    minPrice = data['Close'].min()
    maxPrice = data['Close'].max()
    avgVolume = data['Volume'].mean()
    totalMissing = data.isnull().sum().sum()
    
    return {
        'total_days': len(data),
        'date_range': f"{startDate} to {endDate}",
        'price_range': f"${minPrice:.2f} - ${maxPrice:.2f}",
        'avg_volume': f"{avgVolume:,.0f}",
        'missing_data_points': totalMissing
    }