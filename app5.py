import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="GARP Screener- Created By Maurya Chakravartula",
    page_icon="🎯",
    layout="wide"
)

# --- PORTFOLIO STATE TRACKING ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}


def toggle_entry(ticker):
    if ticker in st.session_state.portfolio:
        del st.session_state.portfolio[ticker]
    else:
        st.session_state.portfolio[ticker] = datetime.now()


# --- HEAVY DISCLAIMERS ---
def show_disclaimers():
    st.error("⚠️ **CRITICAL LEGAL DISCLAIMER & MARKET RISK WARNING**")
    st.markdown("""
    * **RESEARCH ONLY:** This application is strictly an automated quantitative research tool. It is **NOT** a financial advisory service.
    * **NO RECOMMENDATION:** The ratings (Strong Buy, Accumulate, etc.) are purely mathematical outputs based on historical data. They do not constitute personalized investment advice.
    * **VOLATILITY WARNING:** Quantitative models can fail during "Black Swan" events. Past performance is **never** a guarantee of future returns.
    * **CONSULT PROFESSIONALS:** Investing involves a high risk of capital loss. Please consult a qualified, registered investment advisor before making any financial commitments.
    """)


# --- INDUSTRY BENCHMARKS ---
FAIR_PE_INDIA = {
    'Technology': 29.03, 'Financial Services': 22.51, 'Financial_Services_High': 42.34,
    'Consumer Cyclical': 37.93, 'Consumer Defensive': 57.92, 'Healthcare': 33.60,
    'Energy': 24.49, 'Utilities': 24.49, 'Basic Materials': 24.21,
    'Real Estate': 48.64, 'Industrials': 69.00, 'Default': 25.00
}

FAIR_PE_US = {
    'Technology': 32.00, 'Financial Services': 14.50, 'Financial_Services_High': 25.00,
    'Consumer Cyclical': 22.00, 'Consumer Defensive': 19.00, 'Healthcare': 24.00,
    'Energy': 12.00, 'Utilities': 18.00, 'Basic Materials': 16.00,
    'Real Estate': 22.00, 'Industrials': 20.00, 'Default': 20.00
}

# --- UNIVERSES (3 INDIAN + 3 AMERICAN) ---
STOCK_UNIVERSES = {
    "Nifty 50 (India)": ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
                         'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS',
                         'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS',
                         'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS',
                         'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LTIM.NS',
                         'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS',
                         'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS',
                         'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS',
                         'WIPRO.NS'],
    "Nifty Next 50 (India)": ["ABB.NS", "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS",
                              "BAJAJHLDNG.NS", "BANKBARODA.NS", "BOSCHLTD.NS", "CANBK.NS", "CGPOWER.NS", "CHOLAFIN.NS",
                              "COLPAL.NS", "DABUR.NS", "DLF.NS", "DMART.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS",
                              "HAL.NS", "HDFCAMC.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IOC.NS", "INDIGO.NS", "NAUKRI.NS",
                              "JINDALSTEL.NS", "JSWENERGY.NS", "LICI.NS", "MARICO.NS", "MOTHERSON.NS", "PIDILITIND.NS",
                              "PFC.NS", "PNB.NS", "RECLTD.NS", "SHREECEM.NS", "SIEMENS.NS", "TATAPOWER.NS",
                              "TORNTPHARM.NS", "TVSMOTOR.NS", "MCDOWELL-N.NS", "VBL.NS", "VEDL.NS", "ZYDUSLIFE.NS",
                              "ZOMATO.NS"],
    "Nifty Small Cap (India)": ['AADHARHFC.NS', 'AARTIIND.NS', 'ACE.NS', 'AEGISLOG.NS', 'AFFLE.NS', 'AMARAJAELE.NS',
                                'AMBER.NS', 'ANANTRAJ.NS', 'ANGELONE.NS', 'ASTERDM.NS', 'ATUL.NS', 'BATAINDIA.NS',
                                'BEML.NS', 'BSOFT.NS', 'BLS.NS', 'BRIGADE.NS', 'CASTROLIND.NS', 'CESC.NS',
                                'CHAMBLFERT.NS', 'CAMS.NS', 'CREDITACC.NS', 'CROMPTON.NS', 'CYIENT.NS', 'DATAPATTNS.NS',
                                'DELHIVERY.NS', 'DEVYANI.NS', 'LALPATHLAB.NS', 'FSL.NS', 'FIVESTAR.NS', 'GRSE.NS',
                                'GODIGIT.NS', 'GODFRYPHLP.NS', 'GESHIP.NS', 'GSPL.NS', 'HBLPOWER.NS', 'HFCL.NS',
                                'HSCL.NS', 'HINDCOPPER.NS', 'IDBI.NS', 'IFCI.NS', 'IIFL.NS', 'INDIAMART.NS', 'IEX.NS',
                                'INOXWIND.NS', 'IRCON.NS', 'ITI.NS', 'JBMA.NS', 'JWL.NS', 'KPIL.NS', 'KARURVYSYA.NS',
                                'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'LAURUSLABS.NS', 'MGL.NS', 'MANAPPURAM.NS',
                                'MCX.NS', 'NH.NS', 'NATCOPHARM.NS', 'NAVINFLUOR.NS', 'NBCC.NS', 'NCC.NS',
                                'NEULANDLAB.NS', 'NEWGEN.NS', 'NUVAMA.NS', 'PCBL.NS', 'PGEL.NS', 'PEL.NS',
                                'PPLPHARMA.NS', 'PNBHOUSING.NS', 'POONAWALLA.NS', 'PVRINOX.NS', 'RADICO.NS',
                                'RAILTEL.NS', 'RKFORGE.NS', 'REDINGTON.NS', 'RPOWER.NS', 'RITES.NS', 'SHYAMMETL.NS',
                                'SIGNATURE.NS', 'SONATSOFTW.NS', 'SWANENERGY.NS', 'TATACHEM.NS', 'TTML.NS',
                                'TEJASNET.NS', 'RAMCOCEM.NS', 'TITAGARH.NS', 'TRIDENT.NS', 'TRITURBINE.NS',
                                'WELCORP.NS', 'WELSPUNLIV.NS', 'ZENTEC.NS', 'ZENSARTECH.NS'],
    "S&P 500 (US)": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'BRK-B', 'V', 'JPM', 'UNH', 'LLY', 'AVGO',
                     'HD', 'PG', 'MA', 'COST', 'ABBV', 'JNJ', 'CRM', 'BAC', 'WMT', 'CVX', 'KO', 'PEP', 'MRK', 'ORCL',
                     'ADBE', 'LIN', 'CSCO', 'ACN', 'ABT', 'TMO', 'MCD', 'DIS', 'WFC', 'DHR', 'INTC', 'INTU', 'TXN',
                     'VZ', 'PM', 'AMGN', 'QCOM', 'LOW', 'IBM', 'UNP', 'HON', 'CAT', 'GE'],
    "Nasdaq 100 (US)": ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'GOOGL', 'GOOG', 'AVGO', 'TSLA', 'ADBE', 'COST', 'PEP',
                        'NFLX', 'AMD', 'CSCO', 'TMUS', 'CMCSA', 'INTU', 'AMGN', 'TXN', 'HON', 'AMAT', 'QCOM', 'BKNG',
                        'SBUX', 'ISRG', 'ADP', 'MDLZ', 'GILD', 'INTC', 'PANW', 'REGN', 'VRTX', 'ADI', 'LRCX', 'MELI',
                        'MU', 'PYPL', 'KLAC', 'CDNS', 'CSX', 'SNPS', 'ASML', 'MAR', 'ORLY', 'NXPI', 'CTAS', 'ROP',
                        'WDAY', 'PCAR'],
    "Dow Jones 30 (US)": ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS', 'HD', 'HON',
                          'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH',
                          'V', 'VZ', 'WBA', 'WMT']
}


# --- Resilient Data Helpers ---

def get_financial_row(df, keys):
    """Searches through possible financial keys to return a valid data series."""
    if df is None or df.empty: return pd.Series()
    for key in keys:
        if key in df.index:
            return df.loc[key].dropna()
    return pd.Series()


def calculate_dynamic_cagr(series, max_periods=5):
    """Calculates CAGR based on available data with zero-division protection. Returns (rate, years_used)."""
    if series.empty or len(series) < 2: return 0, 0
    series = series.sort_index(ascending=False)
    actual_periods = min(max_periods, len(series) - 1)
    start_val, end_val = series.iloc[actual_periods], series.iloc[0]
    if start_val <= 0:
        rate = (end_val - start_val) / max(1, abs(start_val)) / actual_periods
        return rate, actual_periods
    rate = ((end_val / start_val) ** (1 / actual_periods)) - 1
    return rate, actual_periods


@st.cache_data(ttl=3600)
def fetch_comprehensive_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Resilient History fetching
        hist = stock.history(period="2y")
        if hist.empty or len(hist) < 20:
            return None
        current_price = hist['Close'].iloc[-1]

        # Financials Parsing with fallbacks
        income = stock.income_stmt
        if income.empty: income = stock.financials

        balance = stock.balance_sheet
        if balance.empty: balance = stock.quarterly_balance_sheet

        is_india = ".NS" in ticker or ".BO" in ticker
        benchmarks = FAIR_PE_INDIA if is_india else FAIR_PE_US
        main_index = "^NSEI" if is_india else "^GSPC"

        # 1. GROWTH DATA (Hyper-Resilient Key Search)
        rev_keys = ['Total Revenue', 'Revenue', 'Operating Revenue']
        rev = get_financial_row(income, rev_keys)

        eps_keys = ['Basic EPS', 'Diluted EPS', 'Diluted Net Income Per Share', 'Basic Net Income Per Share']
        eps_series = get_financial_row(income, eps_keys)

        profit_keys = ['Net Income', 'Net Income Common Stockholders',
                       'Net Income From Continuing Operation Net Minority Interest']
        net_profit_series = get_financial_row(income, profit_keys)

        rev_cagr, rev_years = calculate_dynamic_cagr(rev, 5)
        eps_cagr, eps_years = calculate_dynamic_cagr(eps_series, 5)

        last_3_profits = net_profit_series.iloc[0:3].tolist() if not net_profit_series.empty else []
        profit_increasing = len(last_3_profits) >= 2 and last_3_profits[0] > last_3_profits[-1]

        # 2. QUALITY DATA (Resilient extraction)
        equity_keys = ['Stockholders Equity', 'Total Equity', 'Total Stockholders Equity']
        equity_val = get_financial_row(balance, equity_keys).iloc[0] if not get_financial_row(balance,
                                                                                              equity_keys).empty else 1

        ebit_keys = ['EBIT', 'Operating Income']
        ebit_val = get_financial_row(income, ebit_keys).iloc[0] if not get_financial_row(income, ebit_keys).empty else 0
        net_inc_val = net_profit_series.iloc[0] if not net_profit_series.empty else 0

        asset_keys = ['Total Assets']
        total_assets_val = get_financial_row(balance, asset_keys).iloc[0] if not get_financial_row(balance,
                                                                                                   asset_keys).empty else 1

        liab_keys = ['Total Current Liabilities', 'Current Liabilities']
        curr_liab_val = get_financial_row(balance, liab_keys).iloc[0] if not get_financial_row(balance,
                                                                                               liab_keys).empty else 0

        roe_manual = net_inc_val / equity_val if equity_val > 0 else 0
        roce_manual = ebit_val / (total_assets_val - curr_liab_val) if (total_assets_val - curr_liab_val) > 0 else 0
        d_e = info.get('debtToEquity', 0) / 100

        # 3. VALUATION DATA (Resilient P/E and Market Cap)
        sector = info.get('sector', 'Neutral')
        industry = info.get('industry', 'General')
        ind_pe = benchmarks['Financial_Services_High'] if (sector == 'Financial Services' and any(
            x in industry for x in ['Asset Management', 'Broker', 'Exchange'])) else benchmarks.get(sector, benchmarks[
            'Default'])

        # Robust Trailing EPS extraction with multiple providers
        ttm_eps = info.get('trailingEps')
        if ttm_eps is None or ttm_eps == 0:
            ttm_eps = eps_series.iloc[0] if not eps_series.empty else 1

        annual_eps = eps_series.iloc[0] if not eps_series.empty else 1
        if annual_eps > 0 and ttm_eps / annual_eps < 0.4:
            ttm_eps = (ttm_eps * 0.4) + (annual_eps * 0.6)

        pe_ttm = current_price / ttm_eps if ttm_eps > 0 else 0
        growth_pct = max(0.01, eps_cagr * 100)
        peg_manual = pe_ttm / growth_pct if ttm_eps > 0 else 0

        # --- MARKET CAP FALLBACK ---
        m_cap = info.get('marketCap')
        if m_cap is None:
            shares = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')
            m_cap = (shares * current_price) if shares else 50_000_000_000

        m_cap_inr = m_cap if is_india else m_cap * 83

        if m_cap_inr < 100_000_000_000:  # Below 10,000 Cr
            cap_factor, cap_type = 0.75, "Small Cap"
        elif m_cap_inr < 500_000_000_000:  # Below 50,000 Cr
            cap_factor, cap_type = 0.65, "Mid Cap"
        elif m_cap_inr < 2_000_000_000_000:  # Below 2 Lakh Cr
            cap_factor, cap_type = 0.55, "Large Cap"
        else:  # Mega
            cap_factor, cap_type = 0.45, "Mega Cap"

        adjusted_growth = eps_cagr * cap_factor
        safe_growth = max(0, adjusted_growth)

        # Intrinsic Calculation
        fair_pe_logic = min(ind_pe, (eps_cagr * 100) * 1.8)
        if fair_pe_logic < 10: fair_pe_logic = 10

        forward_eps = ttm_eps * (1 + safe_growth)
        intrinsic_val = forward_eps * fair_pe_logic
        upside = (intrinsic_val - current_price) / current_price if current_price > 0 else 0

        # 4. TECHNICALS
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else current_price
        sma_200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else current_price

        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + (gain / loss)))
        rsi = rsi_series.iloc[-1] if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else 50

        # Index relative strength check
        rel_strength = True
        try:
            idx_hist = yf.Ticker(main_index).history(period="1y")['Close']
            if len(hist) > 126 and len(idx_hist) > 126:
                stock_perf_6m = (current_price / hist['Close'].iloc[-126]) - 1
                idx_perf_6m = (idx_hist.iloc[-1] / idx_hist.iloc[-126]) - 1
                rel_strength = stock_perf_6m > idx_perf_6m
        except:
            pass

        return {
            "ticker": ticker, "name": info.get('shortName', ticker),
            "summary": info.get('longBusinessSummary', "No summary available."),
            "sector": sector, "industry": industry, "current_price": current_price,
            "curr_pfx": "₹" if is_india else "$", "is_india": is_india,
            "metrics": {
                "rev_cagr": rev_cagr, "rev_years": rev_years,
                "eps_cagr": eps_cagr, "eps_years": eps_years,
                "net_profit_increasing": profit_increasing,
                "last_3_profits": last_3_profits, "roce": roce_manual, "roe": roe_manual, "d_e": d_e,
                "peg": peg_manual, "upside": upside, "intrinsic": intrinsic_val, "ind_pe": ind_pe, "pe_ttm": pe_ttm,
                "ttm_eps": ttm_eps, "forward_eps": forward_eps, "safe_growth": safe_growth, "sma_50": sma_50,
                "sma_200": sma_200, "rsi": rsi, "rel_strength": rel_strength,
                "fair_pe_logic": fair_pe_logic, "cap_factor": cap_factor, "cap_type": cap_type
            },
            "raw": {"hist": hist}
        }
    except Exception as e:
        return None


def get_pro_score(data):
    if not data: return 0
    m = data['metrics']
    score = 0
    if m['rev_cagr'] > 0.25:
        score += 10
    elif m['rev_cagr'] > 0.15:
        score += 7
    if m['eps_cagr'] > 0.25:
        score += 12
    elif m['eps_cagr'] > 0.15:
        score += 8
    if m['net_profit_increasing']: score += 8
    avg_q = (m['roce'] + m['roe']) / 2
    if avg_q >= 0.20:
        score += 10
    elif avg_q >= 0.12:
        score += 5
    if m['d_e'] < 0.5:
        score += 10
    elif m['d_e'] < 1.0:
        score += 6
    if 0 < m['peg'] < 1.2:
        score += 15
    elif 1.2 <= m['peg'] < 1.8:
        score += 8
    if m['peg'] > 2.5: score -= 10
    if m['upside'] >= 0.30:
        score += 15
    elif m['upside'] >= 0.15:
        score += 8
    if data['current_price'] > m['sma_50'] > m['sma_200']: score += 8
    if 48 < m['rsi'] < 65: score += 6
    if m['rel_strength']: score += 6
    return max(0, min(100, int(score)))


def display_pro_card(data):
    if not data: return
    ticker = data['ticker']
    score = get_pro_score(data)
    m = data['metrics']
    p = data['curr_pfx']
    is_frozen = False
    if ticker in st.session_state.portfolio:
        entry_date = st.session_state.portfolio[ticker]
        days_held = (datetime.now() - entry_date).days
        if days_held < 30: is_frozen = True

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            st.subheader(f"{data['name']} ({ticker})")
            st.caption(f"{data['sector']} | {data['industry']}")
            if is_frozen: st.info(f"🕒 TECHNICAL FREEZE: Day {days_held}/30")
        with c2:
            st.metric("Price", f"{p}{data['current_price']:.2f}", help="Current Market Price (CMP) of the asset.")
        with c3:
            st.metric("Intrinsic", f"{p}{m['intrinsic']:.2f}", f"{m['upside'] * 100:.1f}% Upside",
                      help="Growth-Adjusted Fair Value estimate.")
        with c4:
            if score >= 85:
                st.success(f"Score: {score}/100\n\nSTRONG BUY")
            elif score >= 70:
                st.info(f"Score: {score}/100\n\nACCUMULATE")
            else:
                st.error(f"Score: {score}/100\n\nAVOID")
            st.button("Mark as Entered" if ticker not in st.session_state.portfolio else "Remove from Holding",
                      key=f"btn_{ticker}", on_click=toggle_entry, args=(ticker,),
                      help="Start the 30-day technical noise freeze.")

        with st.expander("Analysis Deep-Dive"):
            st.markdown(f'<div style="font-size: 0.85rem; color: #cfcfcf; line-height: 1.5;">{data["summary"]}</div>',
                        unsafe_allow_html=True)
            st.divider()

            # --- TRANSPARENCY BLOCK ---
            st.markdown("#### 🧮 Intrinsic Value Transparency")
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.markdown(f"**Trailing EPS (TTM):**\n{p}{m['ttm_eps']:.2f}",
                            help="Actual earnings per share over the last 12 months.")
            with t2:
                growth_val = m['safe_growth'] * 100
                st.markdown(f"**Applied Growth:**\n{growth_val:.1f}%",
                            help=f"Categorized as {m['cap_type']}. Formula: (Historical CAGR * {m['cap_factor']}). This moderated growth rate is used for forward projection.")
            with t3:
                st.markdown(f"**Projected Forward EPS:**\n{p}{m['forward_eps']:.2f}",
                            help="Calculated as: TTM EPS * (1 + Applied Growth). This is the moderated earnings power.")
            with t4:
                st.markdown(f"**Safety PE Base:**\n{m['fair_pe_logic']:.1f}x",
                            help="The multiplier used. Min(Industry PE, Hist CAGR * 1.8).")

            st.info(
                f"**Formula Check:** {p}{m['forward_eps']:.2f} (Forward EPS) × {m['fair_pe_logic']:.1f} (Multiplier) = **{p}{m['intrinsic']:.2f}**")
            st.divider()

            g1, q1 = st.columns(2)
            with g1:
                st.markdown("#### 🔹 Growth (30 pts)")
                eps_yr_info = f" (using {m['eps_years']} yrs)" if m['eps_years'] < 5 else ""
                rev_yr_info = f" (using {m['rev_years']} yrs)" if m['rev_years'] < 5 else ""

                st.markdown(f"**EPS CAGR (5Y):** {m['eps_cagr'] * 100:.1f}%{eps_yr_info}",
                            help="Unadjusted Historical CAGR of earnings per share.")
                st.markdown(f"**Rev CAGR (5Y):** {m['rev_cagr'] * 100:.1f}%{rev_yr_info}",
                            help="Confirming operational expansion.")
                st.markdown(f"**Profit Trend:** {'✅ Increasing' if m['net_profit_increasing'] else '❌ Weak'}")
            with q1:
                st.markdown("#### 🔹 Quality (20 pts)")
                st.markdown(f"**ROCE/ROE (Avg):** {((m['roce'] + m['roe']) / 2) * 100:.1f}%",
                            help="Capital efficiency benchmark.")
                st.markdown(f"**Debt to Equity:** {m['d_e']:.2f}", help="Lower is better (<0.5).")
            v1, t1 = st.columns(2)
            with v1:
                st.markdown("#### 🔹 Valuation (30 pts)")
                st.markdown(f"**Industry PE Avg:** {m['ind_pe']:.1f}x")
                st.markdown(f"**Manual PEG:** {m['peg']:.2f}", help="Price divided by Growth. Penalty if > 2.5.")
            with t1:
                st.markdown("#### 🔹 Technicals (20 pts)")
                idx_name = "Nifty 50" if data['is_india'] else "S&P 500"
                st.markdown(f"**RS Index:** {'Outperforming' if m['rel_strength'] else 'Lagging'}")
                st.markdown(f"**MA Alignment:** {'✅ Bullish' if m['sma_50'] > m['sma_200'] else '❌ Bearish'}")

            # THE CHART SECTION
            st.divider()
            fig = go.Figure()
            hist_raw = data['raw']['hist']
            fig.add_trace(
                go.Scatter(x=hist_raw.index, y=hist_raw['Close'], name='Price', line=dict(color='#1f77b4', width=2.5)))
            fig.add_trace(go.Scatter(x=hist_raw.index, y=hist_raw['Close'].rolling(50).mean(), name='50 DMA',
                                     line=dict(color='orange', width=1.5)))
            fig.add_trace(go.Scatter(x=hist_raw.index, y=hist_raw['Close'].rolling(200).mean(), name='200 DMA',
                                     line=dict(color='red', width=1.5, dash='dot')))
            fig.update_layout(height=450, template='plotly_dark', margin=dict(l=0, r=0, t=10, b=0),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)


# --- APP LAYOUT ---
st.title("🎯 Institutional GARP Screener")
show_disclaimers()
tab_about, tab_screener, tab_analyzer = st.tabs(["About Strategy", "Screener", "Single Stock Analyzer"])

with tab_about:
    st.header("The Institutional Quantamental Framework")
    st.markdown("""
    This terminal uses a proprietary **100-Point Framework** designed to identify high-quality growth stocks while filtering out overvalued "growth traps."

    ### 📊 Detailed Scoring Roadmap

    #### 1. Growth (30 Points) - Tiered Rewards
    * **EPS CAGR (Max 12 pts):** Rewards high velocity growth. >25% gets full points.
    * **Revenue CAGR (Max 10 pts):** Confirms operational growth vs financial engineering.
    * **Profit Trajectory (Max 8 pts):** Simple check if the absolute bottom line is expanding.

    #### 2. Quality (20 Points) - Efficiency Hurdles
    * **Capital Efficiency (10 pts):** Uses Average of ROCE and ROE. We seek >20% returns.
    * **Solvency (10 pts):** Debt-to-Equity is a safety filter. Low debt (<0.5) is prioritized.

    #### 3. Valuation (30 Points) - The Bubble Filter
    * **Intrinsic Value Logic:** We moderate historical growth based on **Market Cap Category** (Small: 0.75, Mid: 0.65, Large: 0.55, Mega: 0.45). Fair P/E is anchored at **1.8x the growth rate**.
    * **PEG Punishment:** PEG > 2.5 results in an automatic **10-point loss**.

    #### 4. Technicals (20 Points) - Institutional Confirmation
    * **Relative Strength (RS):** Must outperform the native index (Nifty/S&P) over 6 months.
    * **Trend Alignment:** Price and 50DMA must reside above the 200DMA.

    ---
    ### 🔄 Re-Rating & Exit Protocols

    #### 🔹 Technical Freeze (The 30-Day Rule)
    Most retail investors sell too early due to short-term volatility. Once you "Enter" a stock:
    1.  **Freeze for 30 Trading Days:** Ignore RSI, MACD, and noise.
    2.  **State Tracking:** Use the 'Mark as Entered' button below results to track the freeze period.

    #### 🔹 Hard Exit Rules
    1.  **Immediate Exit:** -15% hard stop-loss.
    2.  **Structural Breakdown:** Exit if price stays below 200DMA for 10+ consecutive days (after the 30-day freeze).
    """)

with tab_screener:
    universe = st.selectbox("Universe Selection", options=list(STOCK_UNIVERSES.keys()),
                            help="Select a market benchmark to scan.")
    if st.button("🚀 Execute Global Quant Scan", type="primary"):
        results = []
        progress = st.progress(0, "Initiating Scan...")
        tickers = STOCK_UNIVERSES[universe]
        for i, t in enumerate(tickers):
            data = fetch_comprehensive_data(t)
            if data: results.append(data)
            progress.progress((i + 1) / len(tickers), f"Processing {t}...")

        if results:
            results = sorted(results, key=lambda x: get_pro_score(x), reverse=True)
            for stock in results[:25]: display_pro_card(stock)
        else:
            st.warning("Unable to fetch data for the selected universe at this moment. Please try again.")

with tab_analyzer:
    search_ticker = st.text_input("Ticker Search (e.g. NVDA, AAPL, RELIANCE.NS)",
                                  help="Search global tickers. Use .NS for NSE India.").upper()
    if st.button("🔍 Analyze Asset"):
        if search_ticker:
            with st.spinner(f"Analyzing {search_ticker} Fundamentals..."):
                data = fetch_comprehensive_data(search_ticker)
                if data:
                    display_pro_card(data)
                else:
                    st.error(
                        f"Analysis Failed for '{search_ticker}'. This ticker may be delisted, the ticker symbol might be incorrect, or the data provider is experiencing temporary latency.")
