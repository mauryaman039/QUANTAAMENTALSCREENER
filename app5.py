# To run this app:
# 1. Make sure you have Python installed.
# 2. Install the necessary libraries: pip install streamlit pandas yfinance numpy plotly
# 3. Save this code as a Python file (e.g., app.py).
# 4. Open your terminal or command prompt, navigate to the file's directory, and run: streamlit run app.py

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time  # Import the time library for delays

# --- Page Configuration ---
st.set_page_config(
    page_title="GARP Quantamental Screener",
    page_icon="🎯",
    layout="wide"
)

# --- GLOBAL CONSTANTS & CONFIGURATION ---
LARGE_CAP_THRESHOLD = 50000 * 1e7  # 50,000 Crores INR

# --- SECTOR P/E BENCHMARKS ---
SECTOR_PE_BENCHMARKS = {
    'Financial Services': 25, 'Technology': 35, 'Healthcare': 40,
    'Consumer Cyclical': 35, 'Industrials': 30, 'Basic Materials': 20,
    'Energy': 15, 'Consumer Defensive': 40, 'Utilities': 20,
    'Communication Services': 28, 'Real Estate': 30, 'Default': 25
}

# --- Stock Universes (Hardcoded for reliability) ---
STOCK_UNIVERSES = {
    "Nifty 50": [
        'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
        'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS',
        'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
        'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS',
        'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LTIM.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS',
        'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS',
        'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS',
        'WIPRO.NS'
    ],
    "Nifty Next 50": [
        "ABB.NS", "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS", "BAJAJHLDNG.NS",
        "BANKBARODA.NS", "BOSCHLTD.NS", "CANBK.NS", "CGPOWER.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS",
        "DLF.NS", "DMART.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "HAL.NS", "HDFCAMC.NS", "ICICIGI.NS",
        "ICICIPRULI.NS", "IOC.NS", "INDIGO.NS", "NAUKRI.NS", "JINDALSTEL.NS", "JSWENERGY.NS", "LICI.NS",
        "MARICO.NS", "MOTHERSON.NS", "PIDILITIND.NS", "PFC.NS", "PNB.NS", "RECLTD.NS", "SHREECEM.NS",
        "SIEMENS.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TVSMOTOR.NS", "MCDOWELL-N.NS", "VBL.NS", "VEDL.NS",
        "ZYDUSLIFE.NS", "ZOMATO.NS"
    ],
    "Nifty Small Cap": [
        'AADHARHFC.NS', 'AARTIIND.NS', 'ACE.NS', 'AEGISLOG.NS', 'AFFLE.NS', 'AMARAJAELE.NS', 'AMBER.NS',
        'ANANTRAJ.NS', 'ANGELONE.NS', 'ASTERDM.NS', 'ATUL.NS', 'BATAINDIA.NS', 'BEML.NS', 'BSOFT.NS',
        'BLS.NS', 'BRIGADE.NS', 'CASTROLIND.NS', 'CESC.NS', 'CHAMBLFERT.NS', 'CAMS.NS', 'CREDITACC.NS',
        'CROMPTON.NS', 'CYIENT.NS', 'DATAPATTNS.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'LALPATHLAB.NS',
        'FSL.NS', 'FIVESTAR.NS', 'GRSE.NS', 'GODIGIT.NS', 'GODFRYPHLP.NS', 'GESHIP.NS', 'GSPL.NS',
        'HBLPOWER.NS', 'HFCL.NS', 'HSCL.NS', 'HINDCOPPER.NS', 'IDBI.NS', 'IFCI.NS', 'IIFL.NS',
        'INDIAMART.NS', 'IEX.NS', 'INOXWIND.NS', 'IRCON.NS', 'ITI.NS', 'JBMA.NS', 'JWL.NS', 'KPIL.NS',
        'KARURVYSYA.NS', 'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'LAURUSLABS.NS', 'MGL.NS', 'MANAPPURAM.NS',
        'MCX.NS', 'NH.NS', 'NATCOPHARM.NS', 'NAVINFLUOR.NS', 'NBCC.NS', 'NCC.NS', 'NEULANDLAB.NS', 'NEWGEN.NS',
        'NUVAMA.NS', 'PCBL.NS', 'PGEL.NS', 'PEL.NS', 'PPLPHARMA.NS', 'PNBHOUSING.NS', 'POONAWALLA.NS',
        'PVRINOX.NS', 'RADICO.NS', 'RAILTEL.NS', 'RKFORGE.NS', 'REDINGTON.NS', 'RPOWER.NS', 'RITES.NS',
        'SHYAMMETL.NS', 'SIGNATURE.NS', 'SONATSOFTW.NS', 'SWANENERGY.NS', 'TATACHEM.NS', 'TTML.NS',
        'TEJASNET.NS', 'RAMCOCEM.NS', 'TITAGARH.NS', 'TRIDENT.NS', 'TRITURBINE.NS', 'WELCORP.NS',
        'WELSPUNLIV.NS', 'ZENTEC.NS', 'ZENSARTECH.NS'
    ]
}


# --- Technical Indicator & Charting Functions ---
def calculate_rsi(data, period=14):
    """Calculates the Relative Strength Index (RSI) with robustness for NaN values."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Handle the case where loss is zero to avoid division by zero
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan) # Replace infinities
    
    rsi = 100 - (100 / (1 + rs))
    
    # Fill initial NaNs with 50 (neutral) to avoid propagation
    rsi = rsi.fillna(50) 
    return rsi


def plot_technical_chart(hist):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist['bollinger_upper'], name='Upper Band', line=dict(color='lightgray', width=1)))
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist['bollinger_lower'], name='Lower Band', line=dict(color='lightgray', width=1),
                   fill='tonexty', fillcolor='rgba(211,211,211,0.2)'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['ema_50'], name='50-Day EMA',
                             line=dict(color='purple', width=1.5, dash='dash')))
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist['ema_200'], name='200-Day EMA', line=dict(color='red', width=1.5, dash='dash')))
    fig.update_layout(title='Technical Price Chart with Bollinger Bands', yaxis_title='Price (INR)',
                      legend_title='Indicators', template='plotly_white')
    return fig


def plot_fundamental_chart(financials):
    if 'Basic EPS' in financials.index:
        eps = financials.loc['Basic EPS'].dropna().sort_index()
        # Ensure there's data to plot
        if not eps.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=eps.index.year, y=eps.values, name='Basic EPS', marker_color='green'))
            fig.update_layout(title='Annual Earnings Per Share (EPS)', yaxis_title='Amount (INR)', xaxis_title='Year',
                              template='plotly_white')
            return fig
    return None


# --- Caching & Data Fetching ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol, period="2y"):
    """
    Fetches, processes, and calculates all data for a single stock.
    Returns a dictionary of data or an error dict.
    """
    try:
        if not ticker_symbol.endswith(".NS"):
            return {"error": f"Invalid ticker. Must end with .NS (e.g., RELIANCE.NS)"}

        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Check for valid info
        if not info or 'shortName' not in info:
            return {"error": "Could not fetch stock info. The ticker might be delisted."}

        hist = stock.history(period=period)
        financials = stock.financials
        
        # Check for sufficient history
        if hist.empty or len(hist) < 200:
            return {"error": "Insufficient trading history (less than 200 days)."}

        # --- Fundamental Calculations ---
        market_cap = info.get('marketCap')
        trailing_pe = info.get('trailingPE')
        eps_cagr_3y = None
        peg_ratio = None

        if not financials.empty and 'Basic EPS' in financials.index and len(financials.columns) >= 4:
            eps_series = financials.loc['Basic EPS'].dropna()
            if len(eps_series) >= 4:
                end_eps = eps_series.iloc[0]
                start_eps = eps_series.iloc[3]
                if start_eps and start_eps > 0 and end_eps > 0:
                    eps_cagr_3y = ((end_eps / start_eps) ** (1 / 3)) - 1
        
        if trailing_pe and trailing_pe > 0 and eps_cagr_3y and eps_cagr_3y > 0:
            peg_ratio = trailing_pe / (eps_cagr_3y * 100)

        eps_growth_1y = None
        if not financials.empty and 'Basic EPS' in financials.index and len(financials.columns) >= 2:
            eps_series_1y = financials.loc['Basic EPS'].dropna()
            if len(eps_series_1y) >= 2:
                eps_this_year = eps_series_1y.iloc[0]
                eps_last_year = eps_series_1y.iloc[1]
                if eps_last_year and eps_last_year > 0:
                    eps_growth_1y = (eps_this_year - eps_last_year) / eps_last_year

        momentum_6m = hist['Close'].pct_change(periods=126).iloc[-1] if len(hist) > 126 else None

        # --- Technical Calculations ---
        hist['ema_50'] = hist['Close'].ewm(span=50, adjust=False).mean()
        hist['ema_200'] = hist['Close'].ewm(span=200, adjust=False).mean()
        hist['sma_20'] = hist['Close'].rolling(window=20).mean()
        hist['std_20'] = hist['Close'].rolling(window=20).std()
        hist['bollinger_upper'] = hist['sma_20'] + (hist['std_20'] * 2)
        hist['bollinger_lower'] = hist['sma_20'] - (hist['std_20'] * 2)
        hist['bollinger_width'] = (hist['bollinger_upper'] - hist['bollinger_lower']) / hist['sma_20']

        # Handle potential division by zero if sma_20 is 0
        hist['bollinger_width'] = hist['bollinger_width'].replace([np.inf, -np.inf], np.nan)

        # Check for NaN in critical calculations
        if hist[['ema_50', 'ema_200', 'bollinger_width']].isnull().all().any():
             return {"error": "Failed to calculate technical indicators."}

        technicals = {
            "current_price": hist['Close'].iloc[-1],
            "yearly_high": hist['High'].rolling(window=252).max().iloc[-1],
            "ema_50": hist['ema_50'].iloc[-1],
            "ema_200": hist['ema_200'].iloc[-1],
            "rsi_14": calculate_rsi(hist['Close']).iloc[-1],
            "avg_volume_20d": hist['Volume'].rolling(window=20).mean().iloc[-1],
            "last_volume": hist['Volume'].iloc[-1],
            "bollinger_width": hist['bollinger_width'].iloc[-1],
            "is_squeezing": hist['bollinger_width'].iloc[-1] < hist['bollinger_width'].rolling(window=126).quantile(0.10).iloc[-1]
        }
        
        # Check for NaN values in technicals
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in technicals.values()):
            return {"error": f"NaN value encountered in technical data for {ticker_symbol}."}

        return {
            "ticker": ticker_symbol, "info": info,
            "fundamentals": {
                "market_cap": market_cap, "eps_growth_1y": eps_growth_1y, "eps_cagr_3y": eps_cagr_3y,
                "trailing_pe": trailing_pe, "peg_ratio": peg_ratio, "momentum_6m": momentum_6m,
                "sector": info.get('sector', 'Default'), "company_name": info.get('shortName', ticker_symbol)
            }, "technicals": technicals, "raw_hist": hist, "raw_financials": financials
        }
    except Exception as e:
        # Catch specific rate-limiting errors
        if "Too Many Requests" in str(e) or "Rate limited" in str(e):
            return {"error": "Too Many Requests. Rate limited. Try after a while."}
        return {"error": f"An unexpected error occurred: {e}"}


# --- UNIFIED SCORING LOGIC ---
def calculate_fundamental_score(fund_data):
    scores = {}
    growth = fund_data.get('eps_growth_1y')
    if growth is not None:
        if growth > 0.25:
            scores['growth'] = 100
        elif growth > 0.15:
            scores['growth'] = 80
        elif growth > 0.05:
            scores['growth'] = 60
        else:
            scores['growth'] = 20
    else:
        scores['growth'] = 0
    pe = fund_data.get('trailing_pe')
    sector = fund_data.get('sector', 'Default')
    benchmark_pe = SECTOR_PE_BENCHMARKS.get(sector, SECTOR_PE_BENCHMARKS['Default'])
    if pe is not None and pe > 0:
        if pe < 0.8 * benchmark_pe:
            scores['value'] = 100
        elif pe < benchmark_pe:
            scores['value'] = 80
        elif pe < 1.2 * benchmark_pe:
            scores['value'] = 60
        else:
            scores['value'] = 20
    else:
        scores['value'] = 0
    peg = fund_data.get('peg_ratio')
    if peg is not None:
        if peg < 1.0:
            scores['garp'] = 100
        elif peg < 1.5:
            scores['garp'] = 80
        elif peg < 2.0:
            scores['garp'] = 60
        else:
            scores['garp'] = 20
    else:
        scores['garp'] = 0
    momentum = fund_data.get('momentum_6m')
    if momentum is not None:
        if momentum > 0.30:
            scores['momentum'] = 100
        elif momentum > 0.15:
            scores['momentum'] = 80
        elif momentum > 0:
            scores['momentum'] = 60
        else:
            scores['momentum'] = 20
    else:
        scores['momentum'] = 0
    market_cap = fund_data.get('market_cap', 0)
    if market_cap and market_cap > LARGE_CAP_THRESHOLD:
        final_score = (scores.get('garp', 0) * 0.40) + (scores.get('value', 0) * 0.30) + (
                    scores.get('growth', 0) * 0.15) + (scores.get('momentum', 0) * 0.15)
    else:
        final_score = (scores.get('growth', 0) * 0.40) + (scores.get('momentum', 0) * 0.30) + (
                    scores.get('value', 0) * 0.15) + (scores.get('garp', 0) * 0.15)
    return int(final_score)


def calculate_technical_score(tech_data):
    scores = {}
    price = tech_data.get('current_price')
    ema_50 = tech_data.get('ema_50')
    ema_200 = tech_data.get('ema_200')

    if price is None or ema_50 is None or ema_200 is None:
        return 0

    is_uptrend = price > ema_50 and ema_50 > ema_200
    if is_uptrend:
        scores['trend'] = 100
    elif price > ema_200:
        scores['trend'] = 60
    else:
        scores['trend'] = 10

    pullback_proximity = abs(price - ema_50) / price if price > 0 else 1
    if pullback_proximity < 0.02:
        pullback_score = 100
    elif pullback_proximity < 0.05:
        pullback_score = 80
    else:
        pullback_score = 20

    squeeze_score = 100 if tech_data.get('is_squeezing', False) else 20
    scores['setup'] = max(pullback_score, squeeze_score)

    volume_ratio = 0
    avg_vol = tech_data.get('avg_volume_20d')
    last_vol = tech_data.get('last_volume')
    if avg_vol and last_vol and avg_vol > 0:
        volume_ratio = last_vol / avg_vol

    if volume_ratio > 1.5:
        volume_score = 100
    elif volume_ratio > 1.0:
        volume_score = 80
    else:
        volume_score = 40

    rsi = tech_data.get('rsi_14')
    if rsi is not None and 45 < rsi < 70:
        rsi_score = 100
    elif rsi is not None and 40 < rsi < 75:
        rsi_score = 70
    else:
        rsi_score = 20

    scores['confirmation'] = (volume_score + rsi_score) / 2
    final_score = (scores.get('trend', 0) * 0.40) + (scores.get('setup', 0) * 0.40) + (
                scores.get('confirmation', 0) * 0.20)
    return int(final_score)


# --- UI Rendering Functions ---
def display_stock_analysis(stock_data):
    st.markdown("---")
    
    # Check for warnings
    if stock_data.get("warning"):
        st.warning(f"**Data Warning:** {stock_data['warning']}")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(f"**{stock_data['fundamentals']['company_name']} ({stock_data['ticker']})**",
                  f"₹{stock_data['technicals']['current_price']:.2f}")
        st.markdown(f"**Fundamental: {stock_data['fundamental_score']} / 100**")
        st.markdown(f"**Technical: {stock_data['technical_score']} / 100**")
        if stock_data.get('is_high_conviction'): st.markdown(
            "<h5><span style='color:green;'>🎯 High Conviction</span></h5>", unsafe_allow_html=True)
    with col2:
        fund_score = stock_data['fundamental_score']
        tech_score = stock_data['technical_score']
        fund_summary = f"**Fundamental Analysis (Score: {fund_score}):** "
        if fund_score > 80:
            fund_summary += "Shows an **elite GARP** profile, with strong EPS growth at a reasonable price."
        elif fund_score >= 65:
            fund_summary += "Shows a **strong GARP** profile."
        else:
            fund_summary += "The fundamental GARP profile is currently sub-optimal."
        tech_summary = f"\n\n**Technical Analysis (Score: {tech_score}):** "
        if tech_score > 80:
            tech_summary += "The stock is showing an **excellent technical setup**, indicating a potential entry point in a healthy uptrend."
        elif tech_score >= 65:
            tech_summary += "The stock is showing a **favorable technical setup**, such as a pullback to support or a volatility contraction."
        else:
            tech_summary += "The technical picture does not currently present a clear entry point."
        verdict = "\n\n**Verdict:** "
        if fund_score >= 65 and tech_score >= 65:
            verdict += "🎯 This is a **high-conviction GARP candidate** with a favorable entry setup."
        elif fund_score >= 65:
            verdict += "👀 A **fundamentally strong GARP company** to watch for a better technical entry."
        else:
            verdict += "❌ This stock **does not meet the combined GARP criteria**."
        st.info(fund_summary + tech_summary + verdict)

    with st.expander("View Detailed Metrics & Charts"):
        fund, tech = stock_data['fundamentals'], stock_data['technicals']
        fund_col, tech_col = st.columns(2)
        with fund_col:
            st.markdown("**Fundamental Data**")
            st.text(f"Market Cap: ₹{(fund.get('market_cap') or 0) / 1e7:,.0f} Cr",
                    help="The total market value of a company's outstanding shares. Market Cap = Current Share Price × Total Number of Shares.")
            st.text(f"Trailing P/E: {fund.get('trailing_pe'):.2f}" if fund.get(
                'trailing_pe') is not None else "Trailing P/E: N/A",
                    help="Price-to-Earnings ratio. A common metric for valuation. It is calculated by dividing the stock's current price by its earnings per share (EPS) over the last 12 months.")
            st.text(f"1Y EPS Growth: {fund.get('eps_growth_1y', 0) * 100:.2f}%" if fund.get(
                'eps_growth_1y') is not None else "1Y EPS Growth: N/A",
                    help="The percentage increase in a company's Earnings Per Share over the last year. A key indicator of short-term growth.")
            st.text(f"3Y EPS CAGR: {fund.get('eps_cagr_3y', 0) * 100:.2f}%" if fund.get(
                'eps_cagr_3y') is not None else "3Y EPS CAGR: N/A",
                    help="Compound Annual Growth Rate of EPS over 3 years. This measures the smoothed, long-term earnings growth.")
            st.text(f"PEG Ratio (3Y CAGR): {fund.get('peg_ratio'):.2f}" if fund.get(
                'peg_ratio') is not None else "PEG Ratio (3Y CAGR): N/A",
                    help="Price/Earnings to Growth ratio. Compares the P/E ratio to the 3-year earnings growth rate. A value under 1.0 is often considered favorable, suggesting the stock price is reasonable relative to its growth.")
            st.text(f"6M Momentum: {fund.get('momentum_6m', 0) * 100:.2f}%" if fund.get(
                'momentum_6m') is not None else "6M Momentum: N/A",
                    help="The stock's price change over the last 6 months (approximately 126 trading days).")
        with tech_col:
            st.markdown("**Technical Data**")
            price = tech.get('current_price', 0)
            ema_50 = tech.get('ema_50', 0)
            ema_200 = tech.get('ema_200', 0)
            is_uptrend = price > ema_50 and ema_50 > ema_200
            is_long_term_ok = price > ema_200
            trend_status = "Healthy Uptrend" if is_uptrend else "Long-Term Positive" if is_long_term_ok else "Downtrend"
            st.text(f"Trend Status: {trend_status}",
                    help="Indicates the health of the current trend based on the alignment of the price, 50-day EMA, and 200-day EMA.")
            st.text(f"Price vs 50D EMA: {((price / ema_50) - 1) * 100:.2f}%" if ema_50 > 0 else "N/A",
                    help="How far the current price is from the 50-day Exponential Moving Average. A value near 0% indicates a pullback to a key support/resistance level.")
            st.text(f"Bollinger Width: {tech.get('bollinger_width', 0):.3f}",
                    help="Measures the volatility of the stock. A lower value indicates the bands are tightening, which can signal a 'squeeze' before a significant price move.")
            volume_ratio = (tech['last_volume'] / tech['avg_volume_20d']) if tech.get('avg_volume_20d') and tech.get(
                'last_volume') and tech['avg_volume_20d'] > 0 else 0
            st.text(f"Volume vs 20D Avg: {volume_ratio:.2f}x",
                    help="Compares the last day's trading volume to the 20-day average volume. A value > 1.0 indicates higher-than-average interest.")
            st.text(f"RSI (14-day): {tech.get('rsi_14'):.2f}",
                    help="Relative Strength Index. A momentum indicator measuring the speed and change of price movements. Values > 70 are considered overbought, and < 30 are oversold. This model favors values between 45-70.")

        st.plotly_chart(plot_technical_chart(stock_data['raw_hist']), use_container_width=True)
        if 'raw_financials' in stock_data and not stock_data['raw_financials'].empty:
            chart = plot_fundamental_chart(stock_data['raw_financials'])
            if chart: st.plotly_chart(chart, use_container_width=True)


# --- Main Application ---
st.title("🎯 GARP Quantamental Screener")
st.markdown("A disciplined model to find high-quality stocks with strong technical setups.")
with st.expander("⚠️ Important Disclaimer & Data Information", expanded=True):
    st.warning("""
    **This is an educational tool, not financial advice.**
    - All analysis is based on a predefined quantitative model and publicly available data. It does not constitute a recommendation to buy or sell any security.
    - The data is sourced from Yahoo Finance and may have inaccuracies or delays. Always verify data from multiple sources.
    - **Always do your own research** and consult with a qualified financial advisor before making any investment decisions.
    """)

# --- UI Layout with Tabs ---
about_tab, screener_tab, analyzer_tab = st.tabs(["About / How to Use", "Screener", "Single Stock Analyzer"])

with about_tab:
    st.header("Welcome to the GARP Quantamental Screener!")
    st.markdown("""
    This tool is designed for investors who follow the **Growth at a Reasonable Price (GARP)** strategy. It combines quantitative fundamental analysis with technical analysis to identify potentially strong investment opportunities.

    ### What is Quantamental Analysis?
    It's a hybrid approach that uses quantitative models (the **Quant** part) to analyze fundamental financial data (the **amental** part). Our model scores stocks on both their fundamental quality and their technical setup.

    ### How the Scoring Works
    Every stock is graded on two distinct models, each out of 100:

    **1. The Fundamental Score (The "What to Buy"):**
    This score measures the quality and value of the underlying business. It's based on:
    - **Growth:** Recent (1-Year) earnings per share (EPS) growth.
    - **Value:** The stock's P/E ratio compared to its industry peers.
    - **GARP Quality:** The PEG ratio, which balances the P/E ratio against long-term (3-Year) earnings growth. A low PEG is highly desirable.
    - **Momentum:** The stock's price performance over the last 6 months.

    **2. The Technical Score (The "When to Buy"):**
    This score evaluates the current price chart to find a favorable entry point. It prioritizes:
    - **Trend:** Is the stock in a healthy, established uptrend?
    - **Setup:** Is there a low-risk entry opportunity right now? The model favors pullbacks to the 50-day moving average or periods of low volatility.
    - **Confirmation:** Is there supporting evidence, like above-average volume or healthy momentum (RSI)?

    ### How to Use This Tool
    1.  **Start with the Screener Tab:** Select a stock universe (e.g., Nifty Small Cap) and run the screener. This will give you a ranked list of all stocks, with the highest-scoring "High Conviction" candidates at the top.
    2.  **Analyze Promising Stocks:** When you find an interesting stock in the screener, go to the **Single Stock Analyzer** tab.
    3.  **Dive Deeper:** Enter the stock's ticker to get a detailed report, including all the specific metrics, charts, and a plain-English verdict to help with your own research process.
    """)

with screener_tab:
    st.header("Screen a Stock Universe")
    available_indices = ["Nifty 50", "Nifty Next 50", "Nifty Small Cap"]
    selected_universe_name = st.selectbox("Select Stock Universe", options=available_indices, key="screener_universe")
    if st.button("🚀 Run GARP Screener", type="primary"):
        tickers_to_scan = STOCK_UNIVERSES.get(selected_universe_name, [])
        with st.spinner(f"Analyzing {len(tickers_to_scan)} stocks..."):
            all_results = []
            progress_bar = st.progress(0, "Analyzing...")
            
            for i, ticker in enumerate(tickers_to_scan):
                data = get_stock_data(ticker)
                
                if data and not data.get("error"):
                    data['fundamental_score'] = calculate_fundamental_score(data['fundamentals'])
                    data['technical_score'] = calculate_technical_score(data['technicals'])
                    data['is_high_conviction'] = data['fundamental_score'] >= 65 and data['technical_score'] >= 65
                    all_results.append(data)
                elif data and data.get("error"):
                    # Silently skip stocks with errors (e.g., delisted, insufficient data)
                    pass
                
                # We add a delay to be "polite" to the yfinance API and avoid rate-limiting
                # Increase this if you still get rate-limiting errors.
                time.sleep(1.0) 

                progress_bar.progress((i + 1) / len(tickers_to_scan), f"Analyzing {ticker}")
            
            st.session_state.screener_results = sorted(all_results, key=lambda x: (
            x.get('is_high_conviction', False), x.get('fundamental_score', 0) + x.get('technical_score', 0)),
                                                       reverse=True)
            st.session_state.screener_run_complete = True


    # --- Display Screener Results (MODIFIED LOGIC) ---
    if 'screener_run_complete' in st.session_state:
        st.markdown("---")
        st.subheader("GARP Screener Results")
        
        if 'screener_results' in st.session_state and st.session_state.screener_results:
            results = st.session_state.screener_results
            high_conviction_count = sum(1 for r in results if r.get('is_high_conviction'))
            
            st.success(f"Analysis complete! Found **{high_conviction_count}** high-conviction GARP candidates out of {len(results)} analyzed stocks.")

            num_to_display = st.number_input("Number of stocks to display", min_value=1, max_value=len(results),
                                             value=min(10, len(results)), step=1, key="screener_display_num")
            
            # Display the top N stocks from the sorted list
            for stock in results[:num_to_display]:
                display_stock_analysis(stock)
        else:
            # This message now ONLY shows if no stocks could be analyzed at all
            st.warning("""
                **Analysis complete, but no stocks were successfully analyzed.**
                
                This is often caused by the Yahoo Finance API rate-limiting your connection.
                
                Please **wait for 5-10 minutes** and try running the screener again.
            """)
        
        # Clear the flag so it doesn't re-show on a simple page refresh
        del st.session_state.screener_run_complete


with analyzer_tab:
    st.header("Analyze a Single Stock")
    user_ticker = st.text_input("Enter a stock ticker (e.g., RELIANCE.NS)", key="single_ticker").upper()
    
    # Automatically add .NS if the user forgets
    if user_ticker and not user_ticker.endswith(".NS"):
        user_ticker += ".NS"

    if st.button("🔍 Analyze Stock", key='analyze_single'):
        if user_ticker:
            with st.spinner(f"Analyzing {user_ticker}..."):
                data = get_stock_data(user_ticker)
            
            if data and not data.get("error"):
                data['fundamental_score'] = calculate_fundamental_score(data['fundamentals'])
                data['technical_score'] = calculate_technical_score(data['technicals'])
                data['is_high_conviction'] = data['fundamental_score'] >= 65 and data['technical_score'] >= 65
                st.session_state.single_stock_result = data
                if 'single_stock_error' in st.session_state:
                     del st.session_state.single_stock_error
            elif data and data.get("error"):
                 st.session_state.single_stock_error = f"Could not retrieve data for {user_ticker}. Reason: {data['error']}"
                 if 'single_stock_result' in st.session_state:
                    del st.session_state.single_stock_result
            else:
                st.session_state.single_stock_error = f"An unknown error occurred for {user_ticker}."
                if 'single_stock_result' in st.session_state:
                    del st.session_state.single_stock_result
        else:
            st.warning("Please enter a ticker to analyze.")

    # --- Display Single Stock Analyzer Results ---
    if 'single_stock_error' in st.session_state:
        st.error(st.session_state.single_stock_error)

    if 'single_stock_result' in st.session_state:
        st.markdown("---")
        st.subheader(f"Single Stock Analysis: {st.session_state.single_stock_result['ticker']}")
        display_stock_analysis(st.session_state.single_stock_result)
