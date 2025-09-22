"""
TastyAlgo - regime aware backtesting dashboard
messy but works
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from data_fetcher import DataGet
from regime_detection import RegDet
from strategies.moving_average import MAcross
from strategies.momentum import MomStrat  
from strategies.volatility_breakout import VolBreak
from strategies.pairs_trading import PairsTrade
from backtesting import BtEngine
from metrics import MetCalc
from visualizations import ChartGen
import utils

warnings.filterwarnings('ignore')
if 'about_expanded' not in st.session_state:
    st.session_state['about_expanded'] = True


st.set_page_config(
    page_title="TastyAlgo Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# initing all the components
dataGet = DataGet()
regDet = RegDet()
btEng = BtEngine()
metCalc = MetCalc()
chartGen = ChartGen()

st.title("📈 TastyAlgo")
st.markdown("### Algo trading backtesting with regime detection")

st.sidebar.header("🎮 Controls")

strat = st.sidebar.selectbox(
    "Strategy",
    ["MA Crossover", "Momentum", "Vol Breakout", "Pairs Trading"],
    help="pick your poison"
)

if strat == "Pairs Trading":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        tick1 = st.selectbox("Stock 1", ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ", "AMZN"], key="tick1")
    with col2:
        tick2 = st.selectbox("Stock 2", ["MSFT", "GOOGL", "TSLA", "SPY", "QQQ", "AMZN", "AAPL"], key="tick2", index=1)
    if tick1 == tick2:
        st.sidebar.error("need different stocks")
        st.stop()
else:
    ticker = st.sidebar.selectbox(
        "Stock Ticker",
        ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ", "AMZN", "NVDA", "META", "NFLX"],
        help="what to trade"
    )

st.sidebar.subheader("📅 Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    startDate = st.date_input("Start", datetime.now() - timedelta(days=730))
with col2:
    endDate = st.date_input("End", datetime.now())

st.sidebar.subheader("⚙️ Parameters")
stratParams = {}

if strat == "MA Crossover":
    stratParams['shortWin'] = st.sidebar.slider("Short MA", 5, 50, 20, help="fast ma period")
    stratParams['longWin'] = st.sidebar.slider("Long MA", 20, 200, 50, help="slow ma period")
elif strat == "Momentum":
    stratParams['momWin'] = st.sidebar.slider("Momentum Period", 5, 30, 14, help="lookback days")
    stratParams['buyThresh'] = st.sidebar.slider("Buy Thresh (%)", 0.0, 10.0, 2.0, 0.1, help="min return to buy") / 100
    stratParams['sellThresh'] = st.sidebar.slider("Sell Thresh (%)", -10.0, 0.0, -2.0, 0.1, help="max return to sell") / 100
elif strat == "Vol Breakout":
    stratParams['volWin'] = st.sidebar.slider("Vol Window", 10, 50, 20, help="vol calculation period")
    stratParams['volMult'] = st.sidebar.slider("Vol Multiplier", 1.0, 5.0, 2.0, 0.1, help="vol threshold multiplier")
elif strat == "Pairs Trading":
    stratParams['pairsWin'] = st.sidebar.slider("Lookback", 20, 100, 30, help="cointegration window")
    stratParams['entryZ'] = st.sidebar.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1, help="z score to enter trade")
    stratParams['exitZ'] = st.sidebar.slider("Exit Z-Score", 0.0, 1.0, 0.5, 0.1, help="z score to exit trade")

st.sidebar.subheader("🎯 Regime Detection")
regMethod = st.sidebar.radio(
    "Method",
    ["Statistical", "K-Means"],
    help="how to detect market regimes"
)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)", min_value=1000, step=1000, value=10000
)

if initial_capital >= 1000:
    run_bt = st.sidebar.button("Run Backtest", type="primary")
else:
    st.sidebar.warning("Enter at least $1000 as initial capital to enable backtest.")
    run_bt = False
if run_bt:
    with st.spinner("fetching market data..."):
        if strat == "Pairs Trading":
            data1 = dataGet.getData(tick1, startDate, endDate)
            data2 = dataGet.getData(tick2, startDate, endDate)
            if data1 is None or data2 is None:
                st.stop()
        else:
            data = dataGet.getData(ticker, startDate, endDate)
            if data is None:
                st.stop()
    
    st.success("✅ data loaded successfully!")
    
    with st.spinner("detecting market regimes..."):
        if strat == "Pairs Trading":
            # used first stock for regime detection (*ni)
            if regMethod == "Statistical":
                regimes = regDet.detectStat(data1)
            else:
                regimes = regDet.detectKmeans(data1)  
            regimeData = pd.Series(regimes, index=data1.index)
        else:
            if regMethod == "Statistical":
                regimes = regDet.detectStat(data)
            else:
                regimes = regDet.detectKmeans(data)
            regimeData = pd.Series(regimes, index=data.index)
    
    with st.spinner("generating signals..."):
        if strat == "MA Crossover":
            stratObj = MAcross()
            signals = stratObj.genSigs(data, **stratParams)
        elif strat == "Momentum":  
            stratObj = MomStrat()
            signals = stratObj.genSigs(data, **stratParams)
        elif strat == "Vol Breakout":
            stratObj = VolBreak()
            signals = stratObj.genSigs(data, **stratParams)
        elif strat == "Pairs Trading":
            stratObj = PairsTrade()
            signals = stratObj.genSigs(data1, data2, **stratParams)
    
    with st.spinner("running backtest..."):
        btResults, trades = btEng.runBt(signals, regimeData, capital=initial_capital)
        metrics = metCalc.calcOverall(btResults, trades)
        regimeMetrics = metCalc.calcRegime(btResults, trades)
    
    st.success("🎉 backtest complete!")
    
    st.markdown("---")
    stratDescs = utils.getStratDescs()
    st.markdown(f'<div class="strategy-card">{stratDescs[strat]}</div>', unsafe_allow_html=True)
    
    st.subheader("📊 Performance Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Return", f"{metrics['Total Return (%)']:.2f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")  
    with col3:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")
    with col4:
        st.metric("Win Rate", f"{metrics['Win Rate (%)']:.1f}%")
    with col5:
        st.metric("Volatility", f"{metrics['Volatility (%)']:.2f}%")
    with col6:
        st.metric("# Trades", f"{metrics['Number of Trades']}")
    
    st.subheader("🎯 Regime Performance")
    regimeDf = pd.DataFrame(regimeMetrics).T
    # Format trades/days columns as integer, then rest rounded for display
    for col in ['Number of Trades', 'Days']:
        if col in regimeDf.columns:
            regimeDf[col] = regimeDf[col].astype(int)
    regimeDf = regimeDf.round(2)
    st.dataframe(
        regimeDf.style.background_gradient(cmap='RdYlGn', subset=['Total Return (%)', 'Sharpe Ratio'])
            .format({'Total Return (%)': '{:.2f}%', 
                     'Sharpe Ratio': '{:.2f}', 
                     'Volatility (%)': '{:.2f}%'}),
        use_container_width=True
    )
    
    st.subheader("📈 Charts & Analysis")
    mainChart = chartGen.createMain(signals, btResults, strat, stratParams)
    st.plotly_chart(mainChart, use_container_width=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        regChart = chartGen.createRegDist(regimes)
        st.plotly_chart(regChart, use_container_width=True)
    with col2:
        ddChart = chartGen.createDdChart(btResults)
        st.plotly_chart(ddChart, use_container_width=True)
    
    if trades:
        st.subheader("📋 Trade History")
        tradeDf = pd.DataFrame(trades)
        tradeDf['date'] = pd.to_datetime(tradeDf['date']).dt.date
        def colorRegime(regime):
            colors = {
                'Bull': 'background-color: #286140; color: #fff;',
                'Bear': 'background-color: #912a2a; color: #fff;',
                'Sideways': 'background-color: #ad8802; color: #fff;'
            }
            return colors.get(regime, '')
        styledTrades = tradeDf.style.applymap(colorRegime, subset=['regime'])
        st.dataframe(styledTrades, use_container_width=True)
    
    if trades and btResults is not None:
        st.subheader("💾 Export Results")
        csvData = utils.genCsvReport(
            strat, 
            ticker if strat != 'Pairs Trading' else f'{tick1}, {tick2}',
            startDate, endDate, metrics, regimeDf, trades, btResults
        )
        st.download_button(
            label="📊 Download Full Report",
            data=csvData,
            file_name=f"TastyAlgo_{strat}_{ticker if strat != 'Pairs Trading' else f'{tick1}_{tick2}'}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    st.session_state['about_expanded'] = False
with st.expander("ℹ️ About TastyAlgo", expanded=st.session_state['about_expanded']):
    st.markdown(utils.getAbout())

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 14px;'>
        TastyAlgo v1.0 | built for quant trading | data via yahoo finance
    </div>
    """, 
    unsafe_allow_html=True
)
