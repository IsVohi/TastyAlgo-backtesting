"""Used plotly for chart generation."""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ChartGen:
  def __init__(self):
      self.regimeColors = {'Bull': 'green', 'Bear': 'red', 'Sideways': 'yellow'}
      self.chartTemplate = 'plotly_white'
  
  def createMain(self, signals, btResults, strategy, stratParams):
      # create the main multi-panel chart - this is the big one
      
      # setup the subplot structure
      fig = make_subplots(
          rows=3, cols=1,
          subplot_titles=['Price & Trading Signals', 'Market Regimes', 'Portfolio Value'],
          vertical_spacing=0.05,
          row_heights=[0.5, 0.2, 0.3],
          shared_xaxes=True
      )
      
      # add the main price line
      fig.add_trace(
          go.Scatter(
              x=signals.index,
              y=signals['price'],
              mode='lines',
              name='Price',
              line=dict(color='blue', width=2),
              hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
          ),
          row=1, col=1
      )
      
      # add strategy-specific indicators
      self._addStrategyIndicators(fig, signals, strategy, stratParams)
      
      # add buy/sell signal markers
      self._addTradeSignals(fig, signals)
      
      # add regime visualization
      self._addRegimeVisualization(fig, btResults)
      
      # add portfolio performance line
      fig.add_trace(
          go.Scatter(
              x=btResults.index,
              y=btResults['total'],
              mode='lines',
              name='Portfolio',
              line=dict(color='purple', width=2),
              hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
          ),
          row=3, col=1
      )
      
      # add benchmark comparison if data is available
      if 'benchmark_cumulative' in btResults.columns:
          initialValue = btResults['total'].iloc[0]
          benchmarkValue = btResults['benchmark_cumulative'] * initialValue
          
          fig.add_trace(
              go.Scatter(
                  x=btResults.index,
                  y=benchmarkValue,
                  mode='lines',
                  name='Buy & Hold',
                  line=dict(color='gray', width=1, dash='dash'),
                  hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
              ),
              row=3, col=1
          )
      
      # update the layout
      tickerName = stratParams.get('ticker', 'Unknown')
      if strategy == "Pairs Trading":
          tickerName = f"{stratParams.get('tick1', 'Stock1')} & {stratParams.get('tick2', 'Stock2')}"
      
      fig.update_layout(
          height=800,
          title_text=f"{strategy} - {tickerName}",
          showlegend=True,
          template=self.chartTemplate,
          hovermode='x unified'
      )
      
      # customize axes
      fig.update_xaxes(title_text="Date", row=3, col=1)
      fig.update_yaxes(title_text="Price ($)", row=1, col=1)
      fig.update_yaxes(title_text="Regime", row=2, col=1, showticklabels=False)
      fig.update_yaxes(title_text="Portfolio Value ($)", row=3, col=1)
      
      return fig
  
  def _addStrategyIndicators(self, fig, signals, strategy, stratParams):
      # add strategy-specific indicators to the price chart
      
      if strategy == "MA Crossover":
          # add moving average lines
          if 'short_ma' in signals.columns:
              fig.add_trace(
                  go.Scatter(
                      x=signals.index,
                      y=signals['short_ma'],
                      mode='lines',
                      name=f"MA{stratParams.get('shortWin', 20)}",
                      line=dict(color='orange', width=1),
                      hovertemplate='Date: %{x}<br>MA: $%{y:.2f}<extra></extra>'
                  ),
                  row=1, col=1
              )
          
          if 'long_ma' in signals.columns:
              fig.add_trace(
                  go.Scatter(
                      x=signals.index,
                      y=signals['long_ma'],
                      mode='lines',
                      name=f"MA{stratParams.get('longWin', 50)}",
                      line=dict(color='red', width=1),
                      hovertemplate='Date: %{x}<br>MA: $%{y:.2f}<extra></extra>'
                  ),
                  row=1, col=1
              )
      
      elif strategy == "Vol Breakout":
          # add volatility threshold visualization if available
          if 'vol_threshold' in signals.columns:
              # need to normalize volatility for display purposes
              priceMean = signals['price'].mean()
              volScaled = signals['vol_threshold'] * priceMean * 10  # arbitrary scaling factor
              
              fig.add_trace(
                  go.Scatter(
                      x=signals.index,
                      y=signals['price'] + volScaled,
                      mode='lines',
                      name='Vol Threshold',
                      line=dict(color='cyan', width=1, dash='dot'),
                      opacity=0.7,
                      hovertemplate='Date: %{x}<br>Vol Threshold<extra></extra>'
                  ),
                  row=1, col=1
              )
      
      elif strategy == "Pairs Trading":
          # for pairs trading, might want to show the spread
          # but that would require additional data preparation
          pass
  
  def _addTradeSignals(self, fig, signals):
      # add buy and sell signal markers to the price chart
      
      if 'positions' not in signals.columns:
          return
      
      # buy signals (position changes > 0)
      buySignals = signals[signals.get('positions', 0) > 0]
      if not buySignals.empty:
          fig.add_trace(
              go.Scatter(
                  x=buySignals.index,
                  y=buySignals['price'],
                  mode='markers',
                  name='Buy Signal',
                  marker=dict(
                      color='green',
                      size=10,
                      symbol='triangle-up',
                      line=dict(width=2, color='darkgreen')
                  ),
                  hovertemplate='Date: %{x}<br>Buy: $%{y:.2f}<extra></extra>'
              ),
              row=1, col=1
          )
      
      # sell signals (position changes < 0)
      sellSignals = signals[signals.get('positions', 0) < 0]
      if not sellSignals.empty:
          fig.add_trace(
              go.Scatter(
                  x=sellSignals.index,
                  y=sellSignals['price'],
                  mode='markers',
                  name='Sell Signal',
                  marker=dict(
                      color='red',
                      size=10,
                      symbol='triangle-down',
                      line=dict(width=2, color='darkred')
                  ),
                  hovertemplate='Date: %{x}<br>Sell: $%{y:.2f}<extra></extra>'
              ),
              row=1, col=1
          )
  
  def _addRegimeVisualization(self, fig, btResults):
      # add market regime visualization to the chart
      
      for regime in ['Bull', 'Bear', 'Sideways']:
          regimePeriods = btResults[btResults['regime'] == regime]
          if not regimePeriods.empty:
              fig.add_trace(
                  go.Scatter(
                      x=regimePeriods.index,
                      y=[1] * len(regimePeriods),
                      mode='markers',
                      name=f'{regime} Market',
                      marker=dict(
                          color=self.regimeColors[regime],
                          size=4,
                          opacity=0.8
                      ),
                      showlegend=True,
                      hovertemplate=f'Date: %{{x}}<br>Regime: {regime}<extra></extra>'
                  ),
                  row=2, col=1
              )
  
  def createRegDist(self, regimes):
      # create pie chart showing regime distribution
      regimeCounts = pd.Series(regimes).value_counts()
      
      fig = px.pie(
          values=regimeCounts.values,
          names=regimeCounts.index,
          title="Market Regime Distribution",
          color_discrete_map=self.regimeColors,
          template=self.chartTemplate
      )
      
      fig.update_traces(
          textposition='inside',
          textinfo='percent+label',
          hovertemplate='Regime: %{label}<br>Days: %{value}<br>Percentage: %{percent}<extra></extra>'
      )
      
      return fig
  
  def createDdChart(self, btResults):
      # create drawdown chart showing portfolio decline periods
      
      # calculate drawdown series
      runningMax = btResults['total'].expanding(min_periods=1).max()
      drawdown = (btResults['total'] - runningMax) / runningMax * 100
      
      fig = go.Figure()
      
      fig.add_trace(
          go.Scatter(
              x=btResults.index,
              y=drawdown,
              mode='lines',
              fill='tonexty',
              name='Drawdown',
              line=dict(color='red', width=1),
              fillcolor='rgba(255, 0, 0, 0.3)',
              hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
          )
      )
      
      # add zero reference line
      fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
      
      fig.update_layout(
          title='Portfolio Drawdown Over Time',
          xaxis_title='Date',
          yaxis_title='Drawdown (%)',
          template=self.chartTemplate,
          height=400,
          showlegend=False
      )
      
      return fig
  
  def createPerformanceComparison(self, btResults):
      # create chart comparing strategy performance vs benchmark
      fig = go.Figure()
      
      # normalize to percentage returns
      initialValue = btResults['total'].iloc[0]
      strategyReturns = (btResults['total'] / initialValue - 1) * 100
      
      fig.add_trace(
          go.Scatter(
              x=btResults.index,
              y=strategyReturns,
              mode='lines',
              name='Strategy',
              line=dict(color='blue', width=2),
              hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
          )
      )
      
      # add benchmark comparison if available
      if 'benchmark_cumulative' in btResults.columns:
          benchmarkReturns = (btResults['benchmark_cumulative'] - 1) * 100
          
          fig.add_trace(
              go.Scatter(
                  x=btResults.index,
                  y=benchmarkReturns,
                  mode='lines',
                  name='Buy & Hold',
                  line=dict(color='gray', width=2, dash='dash'),
                  hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
              )
          )
      
      fig.update_layout(
          title='Cumulative Returns Comparison',
          xaxis_title='Date',
          yaxis_title='Cumulative Return (%)',
          template=self.chartTemplate,
          height=400,
          hovermode='x unified'
      )
      
      return fig