"""utility functions and helpers. Collection of random useful stuff"""
import pandas as pd
import numpy as np
import io
from datetime import datetime

def getStratDescs():
    # strategy descriptions for the dashboard
    return {
        "MA Crossover": """
        **MA Crossover Strategy**  
        classic trend following approach. buy when short ma crosses above long ma, 
        sell when it crosses below. works great in trending markets but gets chopped up sideways.
        """,
        "Momentum": """
        **Momentum Strategy**  
        momentum trading - buy the strong stuff, sell the weak stuff. uses n-day returns 
        compared to thresholds. can work really well in trending environments but watch out for reversals.
        """,
        "Vol Breakout": """
        **Volatility Breakout Strategy**  
        trades when volatility spikes above historical norms. idea is that high vol often 
        precedes big moves. catches some good moves but also plenty of false signals.
        """,
        "Pairs Trading": """
        **Pairs Trading Strategy**  
        market neutral approach that exploits price divergences between correlated stocks. 
        uses statistical methods to identify when spreads are out of whack, then bets on mean reversion.
        """
    }

def getAbout():
    # about text for the app
    return """
    ### TastyAlgo Dashboard
    
    built this to experiment with different trading strategies and see how they perform 
    across different market conditions. turns out regime matters a lot.
    
    **regime detection**
    - statistical method uses rolling returns and volatility
    - k-means clustering groups similar market conditions
    - helps understand when strategies work and when they dont
    
    **trading strategies**
    - **ma crossover**: classic trend following with moving averages
    - **momentum**: rides price continuation patterns  
    - **vol breakout**: trades volatility spikes for big moves
    - **pairs trading**: market neutral mean reversion plays
    
    **performance metrics**
    - sharpe ratio for risk-adjusted returns
    - max drawdown shows worst peak-to-trough decline
    - win rate shows percentage of profitable trades
    - everything broken down by market regime
    
    data comes from yahoo finance. this is for educational purposes only - 
    dont use this for real trading without doing your own research.
    """

def genCsvReport(strategy, tickers, startDate, endDate, metrics, regimeDf, trades, btResults):
    # generate csv report for download
    output = io.StringIO()
    # header section
    output.write("TASTYALGO PERFORMANCE REPORT\n")
    output.write(f"Strategy: {strategy}\n")
    output.write(f"Assets: {tickers}\n")
    output.write(f"Period: {startDate} to {endDate}\n")
    output.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    # overall performance metrics
    output.write("OVERALL PERFORMANCE\n")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            output.write(f"{key}: {value:.4f}\n")
        else:
            output.write(f"{key}: {value}\n")
    output.write("\n")
    # regime performance breakdown
    output.write("REGIME PERFORMANCE\n")
    regimeDf.to_csv(output)
    output.write("\n")
    
    # trade history
    if trades:
        output.write("TRADE HISTORY\n")
        tradeDf = pd.DataFrame(trades)
        tradeDf.to_csv(output, index=False)
        output.write("\n")
    
    # portfolio timeseries (last 100 days sample)
    output.write("PORTFOLIO TIMESERIES (Sample)\n")
    sampleData = btResults[['total', 'portfolio_returns', 'regime']].tail(100)
    sampleData.to_csv(output)
    
    return output.getvalue()

def formatNumber(value, formatType='currency'):
    # number formatting for display
    if pd.isna(value) or value is None:
        return "N/A"
    
    if formatType == 'currency':
        return f"${value:,.2f}"
    elif formatType == 'percentage':
        return f"{value:.2f}%"
    elif formatType == 'ratio':
        return f"{value:.3f}"
    else:
        return f"{value:.2f}"

def calculateCorrelationMatrix(dataDict):
    # calc correlation matrix for multiple assets
    if len(dataDict) < 2:
        return pd.DataFrame()
    
    # combine all the price series
    combinedData = pd.DataFrame(dataDict)
    
    # calculate returns
    returnsData = combinedData.pct_change().dropna()
    
    # correlation matrix
    corrMatrix = returnsData.corr()
    
    return corrMatrix

def detectOutliers(series, method='iqr', threshold=1.5):
    # outlier detection using iqr or z-score
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lowerBound = Q1 - threshold * IQR
        upperBound = Q3 + threshold * IQR
        
        outliers = (series < lowerBound) | (series > upperBound)
        
    elif method == 'zscore':
        zScores = np.abs((series - series.mean()) / series.std())
        outliers = zScores > threshold
    
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")
    
    return outliers

def calcInfoCoefficient(predictions, actuals):
    # information coefficient (rank correlation)
    alignedData = pd.concat([predictions, actuals], axis=1).dropna()
    
    if len(alignedData) < 2:
        return 0
    
    # rank correlation calculation
    ic = alignedData.iloc[:, 0].corr(alignedData.iloc[:, 1], method='spearman')
    
    return ic if not pd.isna(ic) else 0

def createPerformanceSummary(metricsDict):
    # create formatted performance summary
    summary = []
    summary.append("PERFORMANCE SUMMARY")
    summary.append("=" * 40)
    
    # key metrics with proper formatting
    keyMetrics = [
        ('Total Return', 'Total Return (%)', 'percentage'),
        ('Sharpe Ratio', 'Sharpe Ratio', 'ratio'),
        ('Max Drawdown', 'Max Drawdown (%)', 'percentage'),
        ('Win Rate', 'Win Rate (%)', 'percentage'),
        ('Trades', 'Number of Trades', 'number')
    ]
    
    for displayName, key, formatType in keyMetrics:
        if key in metricsDict:
            value = metricsDict[key]
            if formatType == 'percentage':
                formattedValue = f"{value:.2f}%"
            elif formatType == 'ratio':
                formattedValue = f"{value:.3f}"
            else:
                formattedValue = str(value)
            
            summary.append(f"{displayName:<18}: {formattedValue}")
    
    return "\n".join(summary)

def validateStrategyParams(strategy, params):
    # validate strategy parameters before running backtest
    if strategy == "MA Crossover":
        shortWin = params.get('shortWin', 20)
        longWin = params.get('longWin', 50)
        
        if shortWin >= longWin:
            return False, "short MA period must be less than long MA period"
        
        if shortWin < 1 or longWin < 1:
            return False, "MA periods must be positive integers"
    
    elif strategy == "Momentum":
        momWin = params.get('momWin', 14)
        buyThresh = params.get('buyThresh', 0.02)
        sellThresh = params.get('sellThresh', -0.02)
        
        if momWin < 1:
            return False, "momentum window must be positive"
        
        if buyThresh <= sellThresh:
            return False, "buy threshold must be greater than sell threshold"
    
    elif strategy == "Vol Breakout":
        volWin = params.get('volWin', 20)
        volMult = params.get('volMult', 2.0)
        
        if volWin < 1:
            return False, "volatility window must be positive"
        
        if volMult <= 0:
            return False, "volatility multiplier must be positive"
    
    elif strategy == "Pairs Trading":
        pairsWin = params.get('pairsWin', 30)
        entryZ = params.get('entryZ', 2.0)
        exitZ = params.get('exitZ', 0.5)
        
        if pairsWin < 1:
            return False, "pairs window must be positive"
        
        if entryZ <= exitZ:
            return False, "entry z-score must be greater than exit z-score"
    
    return True, ""

def calculateStrategyComplexity(strategy, params):
    # calculate complexity score for the strategy configuration
    baseComplexity = {
        "MA Crossover": 2.0,
        "Momentum": 4.0,
        "Vol Breakout": 6.0,
        "Pairs Trading": 8.0
    }
    
    complexity = baseComplexity.get(strategy, 5.0)
    if strategy == "MA Crossover":
        shortWin = params.get('shortWin', 20)
        longWin = params.get('longWin', 50)
        
        if shortWin < 10:
            complexity += 1.0
        if longWin < 30:
            complexity += 0.5
    
    elif strategy == "Pairs Trading":
        entryZ = params.get('entryZ', 2.0)
        # lower z-scores mean higher trading frequency
        if entryZ < 1.5:
            complexity += 1.0
    
    return min(complexity, 10.0)

def getMarketHoursInfo():
    return {
        'NYSE': {'open': '09:30', 'close': '16:00', 'timezone': 'EST'},
        'NASDAQ': {'open': '09:30', 'close': '16:00', 'timezone': 'EST'},
        'LSE': {'open': '08:00', 'close': '16:30', 'timezone': 'GMT'},
        'TSE': {'open': '09:00', 'close': '15:00', 'timezone': 'JST'},
        'SSE': {'open': '09:30', 'close': '15:00', 'timezone': 'CST'}
    }

def createRiskWarning():
    return """
    ⚠️ **Risk Warning**: this dashboard is for educational purposes only. 
    past performance does not guarantee future results. trading involves significant risk of loss. 
    always do your own research and consult with professionals before making investment decisions.

    📊 **Data Disclaimer**: market data provided by yahoo finance and may contain delays or errors. 
    this tool should not be used for actual trading decisions without verification.
    """