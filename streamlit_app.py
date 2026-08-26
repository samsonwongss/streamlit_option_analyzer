"""
Options Analyzer — Streamlit
============================
Screen option-income and credit-spread opportunities, then fine-tune the
premium / net credit on any candidate to see how the fill price changes
the return and annualized return rate.

Strategies:
  • Cash-Secured Put  — Ret % = Premium / Strike
  • Covered Call      — Ret % = Premium / Cost per share
                        (leave "Cost basis" blank to use the live price)
  • Bull Put Spread   — Sell higher-strike put, buy lower-strike put (bullish)
  • Bear Call Spread  — Sell lower-strike call, buy higher-strike call (bearish)

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
MODE_BPS = "Bull Put Spread"
MODE_BCS = "Bear Call Spread"

SPREAD_MODES = {MODE_BPS, MODE_BCS}
SINGLE_MODES = {MODE_CSP, MODE_CC}

RISK_FREE_RATE = 0.045

COLOR_SUCCESS = "#10b981"
COLOR_DANGER  = "#ef4444"
COLOR_WARNING = "#f59e0b"
COLOR_ACCENT  = "#60a5fa"
COLOR_CARD    = "#23272f"
COLOR_BORDER  = "#374151"

st.set_page_config(
    page_title="Options Analyzer",
    page_icon="📈",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────
#  Black-Scholes delta (scipy-free — uses math.erf)
# ─────────────────────────────────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(S: float, K: float, T: float, sigma, option_type: str):
    if sigma is None or T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    try:
        d1 = (math.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * T) \
             / (sigma * math.sqrt(T))
        return _norm_cdf(d1) if option_type == "call" else _norm_cdf(d1) - 1.0
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────
def _mid(row) -> float:
    bid = float(row["bid"])
    return (bid + float(row["ask"])) / 2 if bid > 0 else float(row["lastPrice"])


# ─────────────────────────────────────────────────────────────────────
#  Single-leg fetch  (CSP / Covered Call)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_options(mode: str, ticker: str, strike: float, strike_range: float,
                  min_volume: int, dte_max: int, cost_basis: float | None):
    tk_obj = yf.Ticker(ticker)
    current_price  = float(tk_obj.fast_info.last_price)
    cost_per_share = cost_basis if (cost_basis and cost_basis > 0) else current_price

    expirations = tk_obj.options
    if not expirations:
        raise ValueError(f"No option data found for {ticker}.")

    today = date.today()
    kind  = "puts" if mode == MODE_CSP else "calls"
    rows  = []

    for exp_str in expirations:
        exp_date    = date.fromisoformat(exp_str)
        days_to_exp = (exp_date - today).days
        if days_to_exp <= 0 or days_to_exp > dte_max:
            continue
        try:
            chain     = tk_obj.option_chain(exp_str)
            contracts = chain.puts if mode == MODE_CSP else chain.calls
        except Exception:
            continue

        lo       = strike - strike_range
        hi       = strike + strike_range
        filtered = contracts[(contracts["strike"] >= lo) &
                             (contracts["strike"] <= hi)].copy()
        filtered = filtered[filtered["volume"].fillna(0) >= min_volume]
        if filtered.empty:
            continue

        T        = days_to_exp / 365.0
        opt_type = "put" if mode == MODE_CSP else "call"

        for _, row in filtered.iterrows():
            mid_price  = (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 else row["lastPrice"]
            strike_val = float(row["strike"])

            if mode == MODE_CSP:
                pct_return = (mid_price / strike_val) * 100 if strike_val > 0 else 0
            else:
                pct_return = (mid_price / cost_per_share) * 100 if cost_per_share > 0 else 0

            ann_return = pct_return * 365 / days_to_exp if days_to_exp > 0 else 0

            iv = float(row["impliedVolatility"]) \
                if "impliedVolatility" in row.index and pd.notna(row["impliedVolatility"]) else None
            delta_val = bs_delta(current_price, strike_val, T, iv, opt_type)

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
                "Delta":      round(delta_val, 3) if delta_val is not None else None,
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
            f"with volume ≥ {min_volume}. Try widening the strike range or "
            "lowering minimum volume."
        )
    return current_price, cost_per_share, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  Credit-spread fetch  (Bull Put Spread / Bear Call Spread)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_spreads(mode: str, ticker: str, strike: float, strike_range: float,
                  max_spread: float, min_pl_ratio: float,
                  min_volume: int, dte_max: int):
    tk_obj = yf.Ticker(ticker)
    current_price = float(tk_obj.fast_info.last_price)

    expirations = tk_obj.options
    if not expirations:
        raise ValueError(f"No option data found for {ticker}.")

    today = date.today()
    rows  = []

    for exp_str in expirations:
        exp_date    = date.fromisoformat(exp_str)
        days_to_exp = (exp_date - today).days
        if days_to_exp <= 0 or days_to_exp > dte_max:
            continue
        try:
            chain = tk_obj.option_chain(exp_str)
        except Exception:
            continue

        T         = days_to_exp / 365.0
        is_put    = (mode == MODE_BPS)
        contracts = chain.puts if is_put else chain.calls
        opt_type  = "put" if is_put else "call"

        lo     = strike - strike_range
        hi     = strike + strike_range
        shorts = contracts[(contracts["strike"] >= lo) &
                           (contracts["strike"] <= hi)].copy()
        shorts = shorts[shorts["volume"].fillna(0) >= min_volume]
        if shorts.empty:
            continue

        for _, sr in shorts.iterrows():
            short_strike = float(sr["strike"])
            short_mid    = _mid(sr)
            short_iv     = float(sr["impliedVolatility"]) \
                if "impliedVolatility" in sr.index and pd.notna(sr["impliedVolatility"]) else None
            short_delta  = bs_delta(current_price, short_strike, T, short_iv, opt_type)

            if is_put:
                longs = contracts[
                    (contracts["strike"] < short_strike) &
                    (contracts["strike"] >= short_strike - max_spread)
                ]
            else:
                longs = contracts[
                    (contracts["strike"] > short_strike) &
                    (contracts["strike"] <= short_strike + max_spread)
                ]

            # Apply the same min-volume filter to the long leg — both legs
            # must meet the volume threshold to keep the pair.
            longs = longs[longs["volume"].fillna(0) >= min_volume]
            if longs.empty:
                continue

            short_vol = int(sr["volume"]) if pd.notna(sr["volume"]) else 0

            for _, lr in longs.iterrows():
                long_strike  = float(lr["strike"])
                long_mid     = _mid(lr)
                spread_width = abs(short_strike - long_strike)
                net_credit   = short_mid - long_mid

                if net_credit <= 0 or spread_width <= 0:
                    continue

                max_risk = spread_width - net_credit
                if max_risk <= 0:
                    continue

                pl_ratio = net_credit / max_risk
                if pl_ratio < min_pl_ratio:
                    continue

                ret_pct  = pl_ratio * 100
                ann_ret  = ret_pct * 365 / days_to_exp
                long_vol = int(lr["volume"]) if pd.notna(lr["volume"]) else 0

                rows.append({
                    "Expiry":     exp_str,
                    "DTE":        days_to_exp,
                    "Short K":    round(short_strike, 2),
                    "Short Vol":  short_vol,
                    "Long K":     round(long_strike,  2),
                    "Long Vol":   long_vol,
                    "Width":      round(spread_width, 2),
                    "Net Cr.":    round(net_credit,   2),
                    "Max Risk":   round(max_risk,     2),
                    "P/L":        round(pl_ratio,     3),
                    "Short Δ":    round(short_delta,  3) if short_delta is not None else None,
                    "Ret %":      round(ret_pct,      3),
                    "Ann. Ret %": round(ann_ret,      2),
                })

    if not rows:
        kind = "bull put spread" if mode == MODE_BPS else "bear call spread"
        raise ValueError(
            f"No {kind} pairs found for {ticker} near ${strike} ±${strike_range}, "
            f"max spread ${max_spread}, min P/L {min_pl_ratio:.2f}. "
            "Try widening the strike range, increasing max spread, or lowering the P/L filter."
        )
    return current_price, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  Price-history fetch
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str) -> pd.DataFrame:
    tk_obj = yf.Ticker(ticker)
    hist   = tk_obj.history(period="6mo", interval="1d")
    if hist.empty:
        hist = tk_obj.history(period="max", interval="1d")
    if hist.empty:
        raise ValueError(f"No price history found for {ticker}.")
    return hist[["Open", "High", "Low", "Close", "Volume"]]


# ─────────────────────────────────────────────────────────────────────
#  Chart
# ─────────────────────────────────────────────────────────────────────
def make_chart(ticker: str, hist: pd.DataFrame, short_strike: float,
               long_strike: float | None = None) -> go.Figure:
    last_close = float(hist["Close"].iloc[-1])

    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"],
        low=hist["Low"],   close=hist["Close"],
        increasing_line_color=COLOR_SUCCESS,
        decreasing_line_color=COLOR_DANGER,
        name=ticker,
    )])

    def _line(y, color, label):
        pct = (y - last_close) / last_close * 100
        fig.add_hline(
            y=y, line_dash="dash", line_color=color, line_width=1.4,
            annotation_text=f"  {label} ${y:.2f}  ({pct:+.1f}%)",
            annotation_position="top left",
            annotation_font_color=color,
        )

    _line(short_strike, COLOR_WARNING, "Short K" if long_strike is not None else "Strike")
    if long_strike is not None and long_strike != short_strike:
        _line(long_strike, COLOR_ACCENT, "Long K")

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
    [MODE_CSP, MODE_CC, MODE_BPS, MODE_BCS],
    help="Single-leg income strategies (CSP, Covered Call) or two-leg credit spreads.",
)

_subtitles = {
    MODE_CSP: "Cash-secured put screener",
    MODE_CC:  "Covered call income screener",
    MODE_BPS: "Bull put credit spread screener (bullish / neutral)",
    MODE_BCS: "Bear call credit spread screener (bearish / neutral)",
}
st.sidebar.caption(_subtitles[mode])

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

ticker_input = st.sidebar.text_input("Ticker", value="KO").strip().upper()
strike_input = st.sidebar.number_input("Strike price ($)", value=75.0, step=1.0, min_value=0.0,
                                       help="For spreads, this is the target short-leg strike.")
strike_range_input = st.sidebar.number_input("Strike range (±$)", value=2.0, step=0.5, min_value=0.0)

cost_basis_input: float | None = None
if mode == MODE_CC:
    cost_raw = st.sidebar.text_input(
        "Cost basis ($, blank = market)", value="",
        help="Your cost per share. Leave blank to use the current market price.",
    ).strip()
    if cost_raw:
        try:
            cost_basis_input = float(cost_raw)
        except ValueError:
            st.sidebar.error("Cost basis must be a number.")

max_spread_input: float | None = None
pl_ratio_n: int | None = None
if mode in SPREAD_MODES:
    max_spread_input = st.sidebar.number_input(
        "Max spread width ($)", value=10.0, step=1.0, min_value=0.5,
        help="Max distance between the short and long strikes.",
    )
    pl_ratio_n = st.sidebar.slider(
        "Target P/L ratio", min_value=2, max_value=10, value=3, step=1,
        format="1:%d",
        help=("Keep spreads where Net Credit / Max Risk ≥ 1 / N. "
              "Left (1:2) is stricter — requires ratio ≥ 0.5. "
              "Right (1:10) is looser — requires ratio ≥ 0.1."),
    )

min_volume_input = st.sidebar.number_input("Min volume", value=0, step=1, min_value=0)
top_n_input      = st.sidebar.number_input("Top N", value=5, step=1, min_value=1)
dte_max_input    = st.sidebar.number_input("DTE max (days)", value=50, step=5, min_value=1)

run = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Premiums use bid/ask mid. Data via Yahoo Finance.")


# ─────────────────────────────────────────────────────────────────────
#  Session state — persist results across reruns triggered by fine-tune
# ─────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# Handle a Run click: fetch fresh data and stash it in session_state.
if run:
    if not ticker_input:
        st.error("Ticker is required.")
    else:
        try:
            if mode in SPREAD_MODES:
                pl_ratio = 1.0 / pl_ratio_n
                with st.spinner(f"Fetching option chain for {ticker_input}…"):
                    price, df = fetch_spreads(
                        mode, ticker_input, float(strike_input),
                        float(strike_range_input), float(max_spread_input),
                        float(pl_ratio), int(min_volume_input), int(dte_max_input),
                    )
                    try:
                        hist = fetch_history(ticker_input)
                    except Exception:
                        hist = None
                st.session_state.results = {
                    "mode": mode, "ticker": ticker_input, "price": price,
                    "cost_per_share": None, "df": df, "hist": hist,
                    "strike": float(strike_input), "top_n": int(top_n_input),
                }
            else:
                with st.spinner(f"Fetching option chain for {ticker_input}…"):
                    price, cost_per_share, df = fetch_options(
                        mode, ticker_input, float(strike_input),
                        float(strike_range_input), int(min_volume_input),
                        int(dte_max_input), cost_basis_input,
                    )
                    try:
                        hist = fetch_history(ticker_input)
                    except Exception:
                        hist = None
                st.session_state.results = {
                    "mode": mode, "ticker": ticker_input, "price": price,
                    "cost_per_share": cost_per_share, "df": df, "hist": hist,
                    "strike": float(strike_input), "top_n": int(top_n_input),
                }
        except Exception as e:
            st.error(str(e))


# ─────────────────────────────────────────────────────────────────────
#  Main panel — render from session_state so fine-tune widgets don't wipe it
# ─────────────────────────────────────────────────────────────────────
results = st.session_state.results

if results is None:
    st.title("Options Analyzer")
    st.info("Set your parameters in the sidebar and click **Run Analysis**.")
    st.stop()

active_mode    = results["mode"]
ticker         = results["ticker"]
price          = results["price"]
cost_per_share = results["cost_per_share"]
df             = results["df"]
hist           = results["hist"]
target_strike  = results["strike"]
top_n          = results["top_n"]

st.title(f"{active_mode} Analyzer")

# ── Summary metrics ──────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker", ticker)
c2.metric("Price",  f"${price:.2f}")
if active_mode in SPREAD_MODES:
    c3.metric("Best P/L Ratio", f"{df['P/L'].max():.3f}")
    c4.metric("Pairs Found",    str(len(df)))
else:
    c3.metric("Best Ann. Ret.",  f"{df['Ann. Ret %'].max():.2f}%")
    c4.metric("Contracts Found", str(len(df)))

if active_mode == MODE_CC and cost_per_share and abs(cost_per_share - price) > 1e-6:
    st.caption(f"Using cost basis **${cost_per_share:.2f}** for return calculations.")

# ── Chart ────────────────────────────────────────────────────────────
st.subheader("Price chart — daily candles, 6 months")
if hist is not None and not hist.empty:
    if active_mode in SPREAD_MODES:
        best_row = df.sort_values(["P/L", "Ann. Ret %"], ascending=False).iloc[0]
        chart_fig = make_chart(ticker, hist,
                               float(best_row["Short K"]),
                               float(best_row["Long K"]))
    else:
        chart_fig = make_chart(ticker, hist, float(target_strike))
    st.plotly_chart(chart_fig, use_container_width=True)
else:
    st.warning("Price history unavailable for this ticker.")

# ── Top-N table ──────────────────────────────────────────────────────
if active_mode in SPREAD_MODES:
    st.subheader("Top spread pairs by P/L ratio")
    df_top = (df.sort_values(["P/L", "Ann. Ret %"], ascending=False)
                .head(int(top_n)).reset_index(drop=True))
    display_cols = ["Expiry", "DTE", "Short K", "Short Vol", "Long K", "Long Vol",
                    "Width", "Net Cr.", "Max Risk", "P/L", "Short Δ",
                    "Ret %", "Ann. Ret %"]
    fmt = {
        "Short K": "{:.2f}", "Long K": "{:.2f}",
        "Short Vol": "{:,}", "Long Vol": "{:,}",
        "Width": "{:.2f}", "Net Cr.": "{:.2f}", "Max Risk": "{:.2f}",
        "P/L": "{:.3f}", "Short Δ": "{:+.3f}",
        "Ret %": "{:.3f}", "Ann. Ret %": "{:.2f}",
    }
else:
    st.subheader("Top contracts by annualized return")
    df_top = (df.sort_values("Ann. Ret %", ascending=False)
                .head(int(top_n)).reset_index(drop=True))
    display_cols = ["Expiry", "DTE", "Strike", "Bid", "Ask", "Premium",
                    "Volume", "OI", "Delta", "Ret %", "Ann. Ret %"]
    if active_mode == MODE_CC:
        display_cols.append("If-Called %")
    fmt = {
        "Strike": "{:.2f}", "Bid": "{:.2f}", "Ask": "{:.2f}",
        "Premium": "{:.2f}", "Volume": "{:,}", "OI": "{:,}",
        "Delta": "{:+.3f}", "Ret %": "{:.3f}", "Ann. Ret %": "{:.2f}",
    }
    if active_mode == MODE_CC:
        fmt["If-Called %"] = "{:.2f}"

styler = (df_top[display_cols].style
          .format(fmt, na_rep="—")
          .apply(lambda r: ['background-color: #0f2e22; color: #10b981'
                            if r.name == 0 else '' for _ in r], axis=1))
st.dataframe(styler, use_container_width=True, hide_index=True)

csv_kind = {MODE_CSP: "csp", MODE_CC: "covered_call",
            MODE_BPS: "bull_put_spread", MODE_BCS: "bear_call_spread"}[active_mode]
st.download_button(
    "Download full results (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker}_{csv_kind}.csv",
    mime="text/csv",
)

# ─────────────────────────────────────────────────────────────────────
#  Fine-tune section
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Fine-tune a candidate")
st.caption(
    "Pick a contract or spread from the top results and adjust the fill price "
    "to see how the return and annualized return change."
)

if active_mode in SPREAD_MODES:
    # ── Spread fine-tune ─────────────────────────────────────────────
    labels = [
        f"{i+1}. {r['Expiry']}  •  Short ${r['Short K']:.2f} / Long ${r['Long K']:.2f}  "
        f"•  Width ${r['Width']:.2f}  •  Net Cr. ${r['Net Cr.']:.2f}"
        for i, r in df_top.iterrows()
    ]
    sel = st.selectbox(
        "Spread pair",
        options=list(range(len(df_top))),
        format_func=lambda i: labels[i],
        key="ft_spread_sel",
    )
    row = df_top.iloc[sel]

    orig_credit  = float(row["Net Cr."])
    orig_risk    = float(row["Max Risk"])
    orig_pl      = float(row["P/L"])
    orig_ret     = float(row["Ret %"])
    orig_ann     = float(row["Ann. Ret %"])
    width        = float(row["Width"])
    dte          = int(row["DTE"])

    # Net credit must be in (0, width) for the math to make sense.
    max_credit = max(width - 0.01, 0.01)
    safe_default = min(max(orig_credit, 0.01), max_credit)

    new_credit = st.number_input(
        "Net credit ($)",
        min_value=0.01,
        max_value=float(max_credit),
        value=float(safe_default),
        step=0.05,
        format="%.2f",
        key=f"ft_credit_{ticker}_{sel}_{row['Expiry']}_{row['Short K']}_{row['Long K']}",
        help=f"Spread width is ${width:.2f}; net credit must be below this.",
    )

    new_risk = width - new_credit
    new_pl   = new_credit / new_risk if new_risk > 0 else float("inf")
    new_ret  = new_pl * 100
    new_ann  = new_ret * 365 / dte if dte > 0 else 0

    # Static: strike distances from current price (positive = above price)
    short_k = float(row["Short K"])
    long_k  = float(row["Long K"])
    short_vs_price = (short_k - price) / price * 100 if price > 0 else 0
    long_vs_price  = (long_k  - price) / price * 100 if price > 0 else 0

    s1, s2 = st.columns(2)
    s1.metric(
        "Short K vs Price", f"{short_vs_price:+.2f}%",
        help=f"Short strike ${short_k:.2f} vs current price ${price:.2f}.",
    )
    s2.metric(
        "Long K vs Price",  f"{long_vs_price:+.2f}%",
        help=f"Long strike ${long_k:.2f} vs current price ${price:.2f}.",
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net credit", f"${new_credit:.2f}",
              f"{new_credit - orig_credit:+.2f}")
    m2.metric("Max risk",   f"${new_risk:.2f}",
              f"{new_risk - orig_risk:+.2f}", delta_color="inverse")
    m3.metric("P/L ratio",  f"{new_pl:.3f}",
              f"{new_pl - orig_pl:+.3f}")
    m4.metric("Ann. Ret %", f"{new_ann:.2f}%",
              f"{new_ann - orig_ann:+.2f}%")

    st.caption(
        f"Original: Net Cr. ${orig_credit:.2f} • Max Risk ${orig_risk:.2f} • "
        f"P/L {orig_pl:.3f} • Ret % {orig_ret:.3f} • Ann. Ret % {orig_ann:.2f}%"
    )

else:
    # ── Single-leg fine-tune ─────────────────────────────────────────
    labels = [
        f"{i+1}. {r['Expiry']}  •  Strike ${r['Strike']:.2f}  •  Premium ${r['Premium']:.2f}"
        for i, r in df_top.iterrows()
    ]
    sel = st.selectbox(
        "Contract",
        options=list(range(len(df_top))),
        format_func=lambda i: labels[i],
        key="ft_single_sel",
    )
    row = df_top.iloc[sel]

    orig_premium = float(row["Premium"])
    orig_ret     = float(row["Ret %"])
    orig_ann     = float(row["Ann. Ret %"])
    strike_val   = float(row["Strike"])
    dte          = int(row["DTE"])

    # Return basis depends on strategy
    if active_mode == MODE_CSP:
        basis = strike_val
        basis_label = f"Strike ${strike_val:.2f}"
    else:  # Covered Call — use stored cost-per-share, fall back to live price
        basis = float(cost_per_share) if cost_per_share else float(price)
        basis_label = f"Cost basis ${basis:.2f}"

    new_premium = st.number_input(
        "Premium ($)",
        min_value=0.01,
        value=float(orig_premium),
        step=0.05,
        format="%.2f",
        key=f"ft_prem_{ticker}_{sel}_{row['Expiry']}_{row['Strike']}",
        help=f"Return is computed against {basis_label}.",
    )

    new_ret = (new_premium / basis) * 100 if basis > 0 else 0
    new_ann = new_ret * 365 / dte if dte > 0 else 0

    is_cc = active_mode == MODE_CC
    n_cols = 5 if is_cc else 4
    cols = st.columns(n_cols)
    cols[0].metric("Premium",    f"${new_premium:.2f}",
                   f"{new_premium - orig_premium:+.2f}")
    cols[1].metric("Ret %",      f"{new_ret:.3f}%",
                   f"{new_ret - orig_ret:+.3f}%")
    cols[2].metric("Ann. Ret %", f"{new_ann:.2f}%",
                   f"{new_ann - orig_ann:+.2f}%")

    if is_cc:
        orig_if_called = float(row["If-Called %"]) if "If-Called %" in row.index else None
        called_gain    = (strike_val - basis) / basis * 100 if basis > 0 else 0
        new_if_called  = new_ret + called_gain
        delta_if_called = (f"{new_if_called - orig_if_called:+.2f}%"
                           if orig_if_called is not None else None)
        cols[3].metric("If-Called %", f"{new_if_called:.2f}%", delta_if_called)

    # Static: strike distance from current price (positive = above price)
    strike_vs_price = (strike_val - price) / price * 100 if price > 0 else 0
    cols[-1].metric(
        "Strike vs Price", f"{strike_vs_price:+.2f}%",
        help=f"Strike ${strike_val:.2f} vs current price ${price:.2f}.",
    )

    if is_cc:
        st.caption(
            f"Original: Premium ${orig_premium:.2f} • Ret % {orig_ret:.3f} • "
            f"Ann. Ret % {orig_ann:.2f}% • If-Called % "
            f"{float(row['If-Called %']):.2f}%"
        )
    else:
        st.caption(
            f"Original: Premium ${orig_premium:.2f} • Ret % {orig_ret:.3f} • "
            f"Ann. Ret % {orig_ann:.2f}%"
        )

# ── Notes ────────────────────────────────────────────────────────────
_notes = {
    MODE_CSP: (
        "Ret % = Premium ÷ Strike • Ann. Ret % = Ret % × (365 ÷ DTE). "
        "Mid-prices shown; actual fills may differ. Short puts carry downside "
        "risk below the strike."
    ),
    MODE_CC: (
        "Ret % = Premium ÷ Cost per share • Ann. Ret % = Ret % × (365 ÷ DTE) • "
        "If-Called % adds the capital gain to strike if shares are assigned. "
        "A covered call caps upside at the strike; shares may be called away."
    ),
    MODE_BPS: (
        "Sell higher-strike put (Short K), buy lower-strike put (Long K) for net credit. "
        "Max Profit = Net Credit • Max Risk = Width − Net Credit • "
        "P/L = Net Credit ÷ Max Risk • Ret % = P/L × 100 • "
        "Ann. Ret % = Ret % × (365 ÷ DTE). Bullish/neutral: full profit if both "
        "puts expire worthless."
    ),
    MODE_BCS: (
        "Sell lower-strike call (Short K), buy higher-strike call (Long K) for net credit. "
        "Max Profit = Net Credit • Max Risk = Width − Net Credit • "
        "P/L = Net Credit ÷ Max Risk • Ret % = P/L × 100 • "
        "Ann. Ret % = Ret % × (365 ÷ DTE). Bearish/neutral: full profit if both "
        "calls expire worthless."
    ),
}
st.caption(_notes[active_mode])
