# main.py
# To run this app:
# 1. Make sure you have Python installed.
# 2. Install the necessary libraries: pip install streamlit pandas yfinance numpy
# 3. Save this code as a Python file (e.g., app.py).
# 4. Open your terminal or command prompt, navigate to the file's directory, and run: streamlit run app.py

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Quantamental GARP Screener",
    page_icon="📈",
    layout="wide"
)

# --- GLOBAL CONSTANTS ---
LARGE_CAP_THRESHOLD = 50000 * 1e7  # 50,000 Crores INR

# --- NEW: Sector P/E Benchmarks for Single Stock Analyzer ---
# These are general benchmarks. A true analysis would use more dynamic data.
SECTOR_PE_BENCHMARKS = {
    'Financial Services': 25,
    'Technology': 35,
    'Healthcare': 40,
    'Consumer Cyclical': 35,
    'Industrials': 30,
    'Basic Materials': 20,
    'Energy': 15,
    'Consumer Defensive': 40,
    'Utilities': 20,
    'Communication Services': 28,
    'Real Estate': 30,
    'Default': 25  # A fallback for sectors not listed
}

# --- Stock Universes ---
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
    "Nifty Next 50 ": [
        "ABB.NS",
        "ADANIENSOL.NS",
        "ADANIGREEN.NS",
        "ADANIPOWER.NS",
        "AMBUJACEM.NS",
        "BAJAJHLDNG.NS",
        "BAJAJHFL.NS",
        "BANKBARODA.NS",
        "BPCL.NS",
        "BRITANNIA.NS",
        "BOSCHLTD.NS",
        "CANBK.NS",
        "CGPOWER.NS",
        "CHOLAFIN.NS",
        "DABUR.NS",
        "DIVISLAB.NS",
        "DLF.NS",
        "DMART.NS",
        "GAIL.NS",
        "GODREJCP.NS",
        "HAVELLS.NS",
        "HAL.NS",
        "ICICIGI.NS",
        "ICICIPRULI.NS",
        "INDHOTEL.NS",
        "IOC.NS",
        "INDIGO.NS",
        "NAUKRI.NS",
        "IRFC.NS",
        "JINDALSTEL.NS",
        "JSWENERGY.NS",
        "LICI.NS",
        "LODHA.NS",
        "LTIM.NS",
        "PIDILITIND.NS",
        "PFC.NS",
        "PNB.NS",
        "RECLTD.NS",
        "MOTHERSON.NS",
        "SHREECEM.NS",
        "SIEMENS.NS",
        "TATAPOWER.NS",
        "TORNTPHARM.NS",
        "TVSMOTOR.NS",
        "MCDOWELL-N.NS",  # United Spirits
        "VBL.NS",
        "VEDL.NS",
        "ZYDUSLIFE.NS"
    ]
}


# --- Custom Technical Indicator Functions ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=window - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=window - 1, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_adx(data, window=14):
    df = data.copy()
    df['tr'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['plus_dm'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                             np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['minus_dm'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                              np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    df['atr'] = df['tr'].ewm(com=window - 1, adjust=False).mean()
    df['plus_di'] = 100 * (df['plus_dm'].ewm(com=window - 1, adjust=False).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].ewm(com=window - 1, adjust=False).mean() / df['atr'])
    df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 0.0001))
    adx = df['dx'].ewm(com=window - 1, adjust=False).mean()
    return adx


# --- Caching Data ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(ticker_symbol, is_single_stock=False):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        hist = stock.history(period="2y")
        financials = stock.financials

        if hist.empty or len(hist) < 252: return None

        # --- Fundamental Data ---
        market_cap = info.get('marketCap')
        trailing_pe = info.get('trailingPE')
        profit_margins = info.get('profitMargins')
        sector = info.get('sector', 'N/A')  # Get sector for dynamic P/E
        net_income_growth = None
        if not financials.empty and 'Net Income' in financials.index and len(financials.columns) >= 2:
            net_income_last_year = financials.loc['Net Income'].iloc[1]
            net_income_this_year = financials.loc['Net Income'].iloc[0]
            if net_income_last_year and net_income_last_year > 0:
                net_income_growth = (net_income_this_year - net_income_last_year) / net_income_last_year
        momentum_6m = hist['Close'].pct_change(periods=126).iloc[-1] if len(hist) > 126 else None

        # --- Technical Data ---
        current_price = hist['Close'].iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        current_rsi = calculate_rsi(hist).iloc[-1]
        current_adx = calculate_adx(hist).iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        avg_volume_50d = hist['Volume'].rolling(window=50).mean().iloc[-1]

        required_metrics = [market_cap, trailing_pe, profit_margins, net_income_growth, momentum_6m]
        if not is_single_stock and any(v is None for v in required_metrics):
            return None

        return {
            "ticker": ticker_symbol, "info": info,
            "fundamentals": {
                "market_cap": market_cap, "net_income_growth": net_income_growth,
                "trailing_pe": trailing_pe, "profit_margins": profit_margins, "momentum_6m": momentum_6m,
                "sector": sector
            },
            "technicals": {
                "current_price": current_price, "sma_200": sma_200, "sma_50": sma_50,
                "rsi_14": current_rsi, "adx_14": current_adx,
                "current_volume": current_volume, "avg_volume_50d": avg_volume_50d
            }
        }
    except Exception:
        return None


# --- Scoring Logic ---
def calculate_screener_scores(all_data):
    df = pd.DataFrame([d['fundamentals'] for d in all_data])
    df['ticker'] = [d['ticker'] for d in all_data]

    df['market_cap'].fillna(0, inplace=True)
    df['net_income_growth'].fillna(0, inplace=True)
    df['trailing_pe'].fillna(999, inplace=True)
    df['profit_margins'].fillna(0, inplace=True)
    df['momentum_6m'].fillna(0, inplace=True)

    # Calculate percentile ranks for each of the four factors
    df['growth_rank'] = df['net_income_growth'].rank(pct=True)
    df['value_rank'] = df['trailing_pe'].rank(pct=True, ascending=False)
    df['quality_rank'] = df['profit_margins'].rank(pct=True)
    df['momentum_rank'] = df['momentum_6m'].rank(pct=True)

    def calculate_dynamic_score(row):
        if row['market_cap'] > LARGE_CAP_THRESHOLD:  # Large Cap Weights
            return (row['quality_rank'] * 40) + (row['value_rank'] * 30) + \
                (row['growth_rank'] * 15) + (row['momentum_rank'] * 15)
        else:  # Mid & Small Cap Weights
            return (row['growth_rank'] * 40) + (row['momentum_rank'] * 30) + \
                (row['value_rank'] * 15) + (row['quality_rank'] * 15)

    df['fundamental_score'] = df.apply(calculate_dynamic_score, axis=1)

    scored_results = []
    for data in all_data:
        ticker_scores = df[df['ticker'] == data['ticker']].iloc[0]
        scored_results.append({
            "ticker": data['ticker'], "company_name": data['info'].get('shortName', 'N/A'),
            "fundamental_score": round(ticker_scores['fundamental_score']),
            **data['fundamentals'], **data['technicals']
        })
    return scored_results


def calculate_technical_score(tech_data):
    score = 0
    if tech_data['current_price'] > tech_data['sma_200']: score += 30
    if tech_data['current_price'] > tech_data['sma_50']: score += 10
    if 55 <= tech_data['rsi_14'] <= 75:
        score += 30
    elif 40 <= tech_data['rsi_14'] < 55:
        score += 15
    if tech_data['adx_14'] > 25: score += 20
    if tech_data['current_volume'] > tech_data['avg_volume_50d']: score += 10
    return score


def generate_description(stock):
    fund_score, tech_score = stock['fundamental_score'], stock['technical_score']
    fund_summary = f"**Fundamental Analysis:** With a score of **{fund_score}**, {stock['company_name']} shows "
    if fund_score > 80:
        fund_summary += "an **elite profile**, ranking in the top tier of its peers."
    elif fund_score >= 65:
        fund_summary += "a **strong profile**, indicating a solid blend of factors."
    else:
        fund_summary += "an **average profile**, not meeting the high standards of the model."
    tech_summary = f"\n\n**Technical Analysis (Score: {tech_score}):** "
    if stock['current_price'] > stock['sma_50'] and stock['current_price'] > stock['sma_200']:
        tech_summary += "The stock is in a **healthy short-term and long-term uptrend**. "
    elif stock['current_price'] > stock['sma_200']:
        tech_summary += "The stock is in a **long-term uptrend but facing short-term weakness**. "
    else:
        tech_summary += "The stock is currently in a **downtrend**. "
    if stock['rsi_14'] > 55:
        tech_summary += f"Momentum is **strong** (RSI: {stock['rsi_14']:.2f}). "
    else:
        tech_summary += f"Momentum is **fading or neutral** (RSI: {stock['rsi_14']:.2f}). "
    if stock['adx_14'] > 25:
        tech_summary += f"The trend has **strong conviction** (ADX: {stock['adx_14']:.2f})."
    else:
        tech_summary += f"However, the trend **lacks strong conviction** (ADX: {stock['adx_14']:.2f})."
    verdict = "\n\n**Verdict:** "
    if fund_score >= 65 and tech_score >= 65:
        verdict += "This is a **high-conviction candidate**, showing strong fundamentals and technicals."
    elif fund_score >= 65:
        verdict += "This is a **fundamentally strong company** on a watchlist for technical improvement."
    else:
        verdict += "This stock **does not meet the combined criteria** for immediate consideration."
    return fund_summary + tech_summary + verdict


# --- UI & Main Logic ---
st.title("📈 The Quantamental GARP Screener")
st.markdown("A tool to find high-quality growth stocks at a reasonable price (GARP) with strong momentum.")

# --- Initialize session_state ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = None

with st.expander("⚠️ Disclaimer"):
    st.warning(
        "This tool is for educational purposes only and is not financial advice. Data is from Yahoo Finance and may have inaccuracies.")

with st.expander("Scoring Guide"):
    st.markdown("""
    **Fundamental Score (0-100): How good is the company?**
    This score is dynamically weighted based on the company's size:
    -   **Large Caps (> ₹50,000 Cr):** Focuses on stability.
        -   *Quality (40%):* Profit Margin
        -   *Value (30%):* Trailing P/E Ratio
        -   *Growth (15%):* Net Income Growth
        -   *Momentum (15%):* 6-Month Price Momentum
    -   **Mid & Small Caps (< ₹50,000 Cr):** Focuses on upside potential.
        -   *Growth (40%):* Net Income Growth
        -   *Momentum (30%):* 6-Month Price Momentum
        -   *Value (15%):* Trailing P/E Ratio
        -   *Quality (15%):* Profit Margin

    **Technical Score (0-100): Is now a good time to buy?**
    This score is based on trend, momentum, and volume indicators.
    """)

# --- Screener Section ---
st.header("1. Stock Screener")
selected_universe = st.selectbox("Select Stock Universe", options=list(STOCK_UNIVERSES.keys()))

if st.button("🚀 Run Screener", type="primary"):
    with st.spinner(f"Analyzing {len(STOCK_UNIVERSES[selected_universe])} stocks..."):
        all_stock_data = [d for d in (get_stock_data(t) for t in STOCK_UNIVERSES[selected_universe]) if d]
    if not all_stock_data:
        st.error("Could not fetch sufficient data for the selected stocks.")
        st.session_state.watchlist = None  # Clear previous results
    else:
        results = calculate_screener_scores(all_stock_data)
        for r in results:
            r['technical_score'] = calculate_technical_score(r)
            r['combined_score'] = r['fundamental_score'] + r['technical_score']
            r['is_high_conviction'] = r['fundamental_score'] >= 65 and r['technical_score'] >= 65

        all_results_sorted = sorted(results, key=lambda x: (x['is_high_conviction'], x['combined_score']), reverse=True)

        st.session_state.watchlist = all_results_sorted

# --- Display Results if they exist in session_state ---
if st.session_state.watchlist is not None:
    watchlist_sorted = st.session_state.watchlist
    st.success(f"Analysis complete! Showing top stocks from the {len(watchlist_sorted)} analyzed stocks.")

    if not watchlist_sorted:
        st.warning("No stocks were found in the selected universe.")
    else:
        num_to_display = st.number_input(
            label="Select how many top stocks to display",
            min_value=1,
            max_value=len(watchlist_sorted),
            value=min(15, len(watchlist_sorted)),
            step=1,
            help=f"You can display up to {len(watchlist_sorted)} qualifying stocks.",
            key='stock_display_count'
        )

        for stock in watchlist_sorted[:num_to_display]:
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric(f"**{stock['company_name']} ({stock['ticker']})**", f"₹{stock['current_price']:.2f}")
                st.markdown(f"**Fundamental Score: {stock['fundamental_score']} / 100**")
                st.markdown(f"**Technical Score: {stock['technical_score']} / 100**")
                if stock['is_high_conviction']:
                    st.markdown("🎯 **High Conviction**")
            with col2:
                st.markdown("**Model Summary & Interpretation:**")
                st.info(generate_description(stock))
                with st.expander("View Detailed Metrics"):
                    fund_col, tech_col = st.columns(2)
                    with fund_col:
                        st.markdown("**Fundamental Data**")
                        st.text(f"Market Cap: ₹{(stock['market_cap'] or 0) / 1e7:,.0f} Cr")
                        st.text(f"Net Income Growth (YoY): {stock['net_income_growth'] * 100:.2f}%" if stock[
                                                                                                           'net_income_growth'] is not None else "N/A")
                        st.text(f"Trailing P/E Ratio: {stock['trailing_pe']:.2f}" if stock[
                                                                                         'trailing_pe'] is not None else "N/A")
                        st.text(f"Profit Margins: {stock['profit_margins'] * 100:.2f}%" if stock[
                                                                                               'profit_margins'] is not None else "N/A")
                        st.text(f"6-Month Momentum: {stock['momentum_6m'] * 100:.2f}%" if stock[
                                                                                              'momentum_6m'] is not None else "N/A")
                    with tech_col:
                        st.markdown("**Technical Data**")
                        st.text(f"Price vs 50D SMA: {'Above' if stock['current_price'] > stock['sma_50'] else 'Below'}")
                        st.text(
                            f"Price vs 200D SMA: {'Above' if stock['current_price'] > stock['sma_200'] else 'Below'}")
                        st.text(f"14D RSI: {stock['rsi_14']:.2f}")
                        st.text(f"14D ADX: {stock['adx_14']:.2f}")
                        st.text(
                            f"Volume vs 50D Avg: {'High' if stock['current_volume'] > stock['avg_volume_50d'] else 'Low'}")

# --- Single Stock Analyzer Section ---
st.header("2. Single Stock Analyzer")
user_ticker = st.text_input("Enter a single stock ticker (e.g., TATAMOTORS.NS)").upper()

if st.button("🔍 Analyze Stock", key='analyze_single_stock'):
    if not user_ticker:
        st.warning("Please enter a stock ticker.")
    else:
        with st.spinner(f"Analyzing {user_ticker}..."):
            data = get_stock_data(user_ticker, is_single_stock=True)
        if not data:
            st.error(
                f"Could not fetch data for {user_ticker}. Please check the ticker symbol. For Indian stocks, ensure it ends with '.NS'. The data source may also be temporarily unavailable.")
        else:
            # --- MODIFIED: Dynamic P/E logic starts here ---
            g, v, q, m = 0, 0, 0, 0  # growth, value, quality, momentum points

            # Get the sector and its benchmark P/E
            sector = data['fundamentals'].get('sector')
            benchmark_pe = SECTOR_PE_BENCHMARKS.get(sector, SECTOR_PE_BENCHMARKS['Default'])

            st.info(f"Analyzing based on the '{sector}' sector benchmark P/E of {benchmark_pe}.")

            # Value check against dynamic benchmark
            if data['fundamentals']['trailing_pe'] is not None and data['fundamentals']['trailing_pe'] < benchmark_pe:
                v = 1

            # Other checks remain the same
            if data['fundamentals']['net_income_growth'] is not None and data['fundamentals'][
                'net_income_growth'] > 0.15: g = 1
            if data['fundamentals']['profit_margins'] is not None and data['fundamentals'][
                'profit_margins'] > 0.10: q = 1
            if data['fundamentals']['momentum_6m'] is not None and data['fundamentals']['momentum_6m'] > 0.20: m = 1

            fund_score = 0
            market_cap = data['fundamentals']['market_cap']
            if market_cap and market_cap > LARGE_CAP_THRESHOLD:  # Large Cap
                fund_score = (q * 40) + (v * 30) + (g * 15) + (m * 15)
            else:  # Mid & Small Cap
                fund_score = (g * 40) + (m * 30) + (v * 15) + (q * 15)

            tech_score = calculate_technical_score(data['technicals'])
            stock_info = {"fundamental_score": fund_score, "technical_score": tech_score, **data['fundamentals'],
                          **data['technicals'], **data['info']}
            stock_info['company_name'] = data['info'].get('shortName', user_ticker)

            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric(f"**{stock_info['company_name']} ({data['ticker']})**", f"₹{stock_info['current_price']:.2f}")
                st.markdown(f"**Fundamental Score: {stock_info['fundamental_score']} / 100**")
                st.markdown(f"**Technical Score: {stock_info['technical_score']} / 100**")
                if stock_info['fundamental_score'] >= 65 and stock_info['technical_score'] >= 65:
                    st.markdown("🎯 **High Conviction**")
            with col2:
                st.markdown("**Model Summary & Interpretation:**")
                st.info(generate_description(stock_info))
                with st.expander("View Detailed Metrics"):
                    fund_col, tech_col = st.columns(2)
                    with fund_col:
                        st.markdown("**Fundamental Data**")
                        st.text(f"Market Cap: ₹{(stock_info['market_cap'] or 0) / 1e7:,.0f} Cr")
                        st.text(f"Net Income Growth (YoY): {stock_info['net_income_growth'] * 100:.2f}%" if stock_info[
                                                                                                                'net_income_growth'] is not None else "N/A")
                        st.text(f"Trailing P/E Ratio: {stock_info['trailing_pe']:.2f}" if stock_info[
                                                                                              'trailing_pe'] is not None else "N/A")
                        st.text(f"Profit Margins: {stock_info['profit_margins'] * 100:.2f}%" if stock_info[
                                                                                                    'profit_margins'] is not None else "N/A")
                        st.text(f"6-Month Momentum: {stock_info['momentum_6m'] * 100:.2f}%" if stock_info[
                                                                                                   'momentum_6m'] is not None else "N/A")
                    with tech_col:
                        st.markdown("**Technical Data**")
                        st.text(
                            f"Price vs 50D SMA: {'Above' if stock_info['current_price'] > stock_info['sma_50'] else 'Below'}")
                        st.text(
                            f"Price vs 200D SMA: {'Above' if stock_info['current_price'] > stock_info['sma_200'] else 'Below'}")
                        st.text(f"14D RSI: {stock_info['rsi_14']:.2f}")
                        st.text(f"14D ADX: {stock_info['adx_14']:.2f}")
                        st.text(
                            f"Volume vs 50D Avg: {'High' if stock_info['current_volume'] > stock_info['avg_volume_50d'] else 'Low'}")