"""
Options Analyzer — Streamlit  (Cash-Secured Put + Covered Call)
================================================================
Web version of the desktop Options Analyzer. Pick a strategy in the
sidebar, set your parameters, and screen option-income opportunities.

  • Cash-Secured Put  — Ret % = Premium / Strike
  • Covered Call      — Ret % = Premium / Cost per share
                        (leave "Cost basis" blank to use the live price)

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import math
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────
#  Constants / theme colors
# ─────────────────────────────────────────────────────────────────────
MODE_CSP = "Cash-Secured Put"
MODE_CC  = "Covered Call"

# Annual risk-free rate used for the Black-Scholes delta estimate.
# Delta is fairly insensitive to this; adjust to taste (e.g. T-bill yield).
RISK_FREE_RATE = 0.045

COLOR_SUCCESS = "#10b981"
COLOR_DANGER  = "#ef4444"
COLOR_WARNING = "#f59e0b"
COLOR_CARD    = "#23272f"
COLOR_BORDER  = "#374151"
COLOR_TEXT    = "#e5e7eb"

st.set_page_config(
    page_title="Options Analyzer",
    page_icon="📈",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────
#  Option math — Black-Scholes / Merton (for delta)
# ─────────────────────────────────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function (no SciPy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(opt: str, S: float, K: float, T: float,
              r: float, q: float, sigma: float):
    """Black-Scholes-Merton price with continuous dividend yield q."""
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def _implied_vol(opt: str, price: float, S: float, K: float, T: float,
                 r: float, q: float):
    """Back implied vol out of an option's mid price by bisection.

    Returns None when the price sits outside the model's no-arbitrage
    bounds (common for stale / illiquid quotes) so the caller can fall back.
    """
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    lo, hi = 1e-4, 5.0
    p_lo = _bs_price(opt, S, K, T, r, q, lo)
    p_hi = _bs_price(opt, S, K, T, r, q, hi)
    if p_lo is None or p_hi is None or not (p_lo <= price <= p_hi):
        return None
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        pm = _bs_price(opt, S, K, T, r, q, mid)
        if pm is None:
            return None
        if pm > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _bs_delta(opt: str, S: float, K: float, T: float,
              r: float, q: float, sigma):
    """Black-Scholes-Merton delta. Calls in [0, 1], puts in [-1, 0]."""
    if sigma is None or sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if opt == "call":
        return math.exp(-q * T) * _norm_cdf(d1)
    return -math.exp(-q * T) * _norm_cdf(-d1)


def _trailing_dividend_yield(tk_obj, price: float) -> float:
    """Estimate a continuous dividend yield from the last 12 months of payouts."""
    try:
        divs = tk_obj.dividends
        if divs is None or divs.empty or price <= 0:
            return 0.0
        cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
        ttm = float(divs[divs.index >= cutoff].sum())
        q = ttm / price
        return q if 0.0 <= q < 0.5 else 0.0   # clamp obvious bad data
    except Exception:
        return 0.0


def _contract_delta(opt: str, mid_price: float, yahoo_iv, S: float, K: float,
                    T: float, r: float, q: float):
    """Delta for one contract: derive IV from the mid, else fall back to Yahoo IV."""
    sigma = _implied_vol(opt, mid_price, S, K, T, r, q)
    if sigma is None:
        sigma = float(yahoo_iv) if (yahoo_iv is not None and pd.notna(yahoo_iv)
                                    and yahoo_iv > 0) else None
    return _bs_delta(opt, S, K, T, r, q, sigma)


# ─────────────────────────────────────────────────────────────────────
#  Core data-fetch logic  (cached so reruns are fast)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_options(mode: str, ticker: str, strike: float, strike_range: float,
                  min_volume: int, dte_max: int, cost_basis: float | None):
    """Fetch put or call option chains.

    Returns (current_price, cost_per_share, dataframe).
    """
    tk_obj = yf.Ticker(ticker)

    info = tk_obj.fast_info
    current_price = float(info.last_price)

    cost_per_share = cost_basis if (cost_basis and cost_basis > 0) else current_price

    expirations = tk_obj.options
    if not expirations:
        raise ValueError(f"No option data found for {ticker}. Check the ticker symbol.")

    today = date.today()
    rows = []
    kind = "puts" if mode == MODE_CSP else "calls"

    # Inputs for the Black-Scholes delta estimate
    opt_type = "put" if mode == MODE_CSP else "call"
    r = RISK_FREE_RATE
    q = _trailing_dividend_yield(tk_obj, current_price)

    for exp_str in expirations:
        exp_date = date.fromisoformat(exp_str)
        days_to_exp = (exp_date - today).days

        if days_to_exp <= 0 or days_to_exp > dte_max:
            continue

        try:
            chain = tk_obj.option_chain(exp_str)
            contracts = chain.puts if mode == MODE_CSP else chain.calls
        except Exception:
            continue

        lo = strike - strike_range
        hi = strike + strike_range
        filtered = contracts[(contracts["strike"] >= lo) &
                             (contracts["strike"] <= hi)].copy()
        filtered = filtered[filtered["volume"].fillna(0) >= min_volume]

        if filtered.empty:
            continue

        for _, row in filtered.iterrows():
            mid_price = (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 else row["lastPrice"]
            strike_val = float(row["strike"])

            if mode == MODE_CSP:
                pct_return = (mid_price / strike_val) * 100 if strike_val > 0 else 0
            else:  # Covered Call — yield on capital tied up in shares
                pct_return = (mid_price / cost_per_share) * 100 if cost_per_share > 0 else 0

            ann_return = (pct_return * 365 / days_to_exp) if days_to_exp > 0 else 0

            # Black-Scholes-Merton delta (IV backed out of the mid, else Yahoo IV)
            yahoo_iv = row["impliedVolatility"] if "impliedVolatility" in row.index else None
            delta = _contract_delta(
                opt_type, float(mid_price), yahoo_iv,
                current_price, strike_val, days_to_exp / 365.0, r, q,
            )

            row_dict = {
                "Contract":   row["contractSymbol"],
                "Expiry":     exp_str,
                "DTE":        days_to_exp,
                "Strike":     round(strike_val, 2),
                "Bid":        round(float(row["bid"]), 2),
                "Ask":        round(float(row["ask"]), 2),
                "Premium":    round(float(mid_price), 2),
                "Volume":     int(row["volume"]) if pd.notna(row["volume"]) else 0,
                "OI":         int(row["openInterest"]) if pd.notna(row["openInterest"]) else 0,
                "Delta":      round(delta, 3) if delta is not None else None,
                "Ret %":      round(pct_return, 3),
                "Ann. Ret %": round(ann_return, 2),
            }

            if mode == MODE_CC:
                called_gain = ((strike_val - cost_per_share) / cost_per_share) * 100 \
                    if cost_per_share > 0 else 0
                row_dict["If-Called %"] = round(pct_return + called_gain, 2)

            rows.append(row_dict)

    if not rows:
        raise ValueError(
            f"No {kind} found for {ticker} at strike ${strike} ±${strike_range} "
            f"with volume ≥ {min_volume}. Try widening the strike range "
            f"or lowering minimum volume."
        )

    return current_price, cost_per_share, pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str) -> pd.DataFrame:
    """Fetch ~6 months of daily OHLC data (falls back to max available)."""
    tk_obj = yf.Ticker(ticker)
    hist = tk_obj.history(period="6mo", interval="1d")
    if hist.empty:
        hist = tk_obj.history(period="max", interval="1d")
    if hist.empty:
        raise ValueError(f"No price history found for {ticker}.")
    return hist[["Open", "High", "Low", "Close", "Volume"]]


# ─────────────────────────────────────────────────────────────────────
#  Chart
# ─────────────────────────────────────────────────────────────────────
def make_chart(ticker: str, hist: pd.DataFrame, strike: float) -> go.Figure:
    last_close = float(hist["Close"].iloc[-1])
    pct_from_last = (strike - last_close) / last_close * 100

    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        increasing_line_color=COLOR_SUCCESS,
        decreasing_line_color=COLOR_DANGER,
        name=ticker,
    )])

    fig.add_hline(
        y=strike, line_dash="dash", line_color=COLOR_WARNING, line_width=1.4,
        annotation_text=f"  Strike ${strike:.2f}  ({pct_from_last:+.1f}%)",
        annotation_position="top left",
        annotation_font_color=COLOR_WARNING,
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"{ticker} — Daily, 6 Months", x=0.01, font=dict(size=14)),
        height=460,
        margin=dict(l=10, r=10, t=44, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_CARD,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_BORDER)
    return fig


# ─────────────────────────────────────────────────────────────────────
#  Sidebar — strategy selector + parameters
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("Options Analyzer")

mode = st.sidebar.radio(
    "Strategy",
    [MODE_CSP, MODE_CC],
    help="CSP screens the put chain; Covered Call screens the call chain.",
)

if mode == MODE_CSP:
    st.sidebar.caption("Cash-Secured Put screener — Ret % = Premium ÷ Strike")
else:
    st.sidebar.caption("Covered call income screener — Ret % = Premium ÷ Cost per share")

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

ticker = st.sidebar.text_input("Ticker", value="KO").strip().upper()
strike = st.sidebar.number_input("Strike price ($)", value=75.0, step=1.0, min_value=0.0)
strike_range = st.sidebar.number_input("Strike range (±$)", value=2.0, step=0.5, min_value=0.0)

cost_basis: float | None = None
if mode == MODE_CC:
    cost_raw = st.sidebar.text_input(
        "Cost basis ($, blank = market)", value="",
        help="Your cost per share. Leave blank to use the current market price.",
    ).strip()
    if cost_raw:
        try:
            cost_basis = float(cost_raw)
        except ValueError:
            st.sidebar.error("Cost basis must be a number.")
else:
    st.sidebar.text_input("Cost basis ($)", value="", disabled=True,
                          help="Covered Call only.")

min_volume = st.sidebar.number_input("Min volume", value=0, step=1, min_value=0)
top_n = st.sidebar.number_input("Top N", value=5, step=1, min_value=1)
dte_max = st.sidebar.number_input("DTE max (days)", value=50, step=5, min_value=1)

run = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Premiums use bid/ask mid. Data via Yahoo Finance.")


# ─────────────────────────────────────────────────────────────────────
#  Main panel
# ─────────────────────────────────────────────────────────────────────
st.title(f"{mode} Analyzer")

if not run:
    st.info("Set your parameters in the sidebar and click **Run Analysis**.")
    st.stop()

if not ticker:
    st.error("Ticker is required.")
    st.stop()

with st.spinner(f"Fetching option chain for {ticker}…"):
    try:
        price, cost_per_share, df = fetch_options(
            mode, ticker, strike, strike_range, int(min_volume),
            int(dte_max), cost_basis,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    try:
        hist = fetch_history(ticker)
    except Exception:
        hist = None

# ── Summary metrics ──────────────────────────────────────────────────
best_ann = df["Ann. Ret %"].max()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker", ticker)
c2.metric("Price", f"${price:.2f}")
c3.metric("Best Ann. Ret.", f"{best_ann:.2f}%")
c4.metric("Contracts Found", str(len(df)))

if mode == MODE_CC and abs(cost_per_share - price) > 1e-6:
    st.caption(f"Using cost basis **${cost_per_share:.2f}** for return calculations.")

# ── Chart ────────────────────────────────────────────────────────────
st.subheader("Price chart — daily candles, 6 months")
if hist is not None and not hist.empty:
    st.plotly_chart(make_chart(ticker, hist, strike), use_container_width=True)
else:
    st.warning("Price history unavailable for this ticker.")

# ── Table ────────────────────────────────────────────────────────────
st.subheader("Top contracts by annualized return")

df_top = (df.sort_values("Ann. Ret %", ascending=False)
            .head(int(top_n))
            .reset_index(drop=True))

display_cols = ["Expiry", "DTE", "Strike", "Bid", "Ask", "Premium",
                "Volume", "OI", "Delta", "Ret %", "Ann. Ret %"]
if mode == MODE_CC:
    display_cols.append("If-Called %")

styler = (df_top[display_cols].style
          .format({
              "Strike": "{:.2f}", "Bid": "{:.2f}", "Ask": "{:.2f}",
              "Premium": "{:.2f}", "Volume": "{:,}", "OI": "{:,}",
              "Delta": lambda v: f"{v:.3f}" if pd.notna(v) else "—",
              "Ret %": "{:.3f}", "Ann. Ret %": "{:.2f}",
              **({"If-Called %": "{:.2f}"} if mode == MODE_CC else {}),
          })
          .apply(lambda r: ['background-color: #0f2e22; color: #10b981'
                            if r.name == 0 else '' for _ in r], axis=1))

st.dataframe(styler, use_container_width=True, hide_index=True)

st.download_button(
    "Download full results (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker}_{'csp' if mode == MODE_CSP else 'covered_call'}.csv",
    mime="text/csv",
)

# ── Notes ────────────────────────────────────────────────────────────
delta_note = (
    "Delta is a Black-Scholes-Merton estimate (vol implied from the mid, "
    "trailing dividend yield included); |delta| ≈ rough probability of finishing "
    "in the money. Treat it as approximate, not a broker-grade figure."
)

if mode == MODE_CSP:
    st.caption(
        "Premium = mid of bid/ask • Ret % = Premium ÷ Strike • "
        "Ann. Ret % = Ret % × (365 ÷ DTE). " + delta_note + " Mid-prices shown; "
        "actual fills may differ. Short puts carry downside risk if the stock "
        "drops below the strike."
    )
else:
    st.caption(
        "Premium = mid of bid/ask • Ret % = Premium ÷ Cost per share • "
        "Ann. Ret % = Ret % × (365 ÷ DTE) • If-Called % = premium yield + capital "
        "gain to the strike if shares are assigned. " + delta_note + " Mid-prices "
        "shown; actual fills may differ. A covered call caps your upside at the "
        "strike and your shares may be called away."
    )
