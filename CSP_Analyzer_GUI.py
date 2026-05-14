"""
CSP (Cash-Secured Put) Options Analyzer — GUI
==============================================
Modern graphical interface for the CSP analyzer.

USAGE:
    pip install customtkinter yfinance pandas mplfinance matplotlib
    python CSP_Analyzer_GUI.py
"""

import threading
import queue
from datetime import date
from tkinter import ttk
import tkinter as tk

import customtkinter as ctk
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ─────────────────────────────────────────────────────────────────────
#  Theme
# ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Color palette
COLOR_BG          = "#1a1d24"
COLOR_CARD        = "#23272f"
COLOR_CARD_LIGHT  = "#2d323c"
COLOR_ACCENT      = "#3b82f6"      # blue
COLOR_ACCENT_HI   = "#60a5fa"
COLOR_SUCCESS     = "#10b981"      # green
COLOR_WARNING     = "#f59e0b"      # amber
COLOR_DANGER      = "#ef4444"      # red
COLOR_TEXT        = "#e5e7eb"
COLOR_TEXT_DIM    = "#9ca3af"
COLOR_BORDER      = "#374151"


# ─────────────────────────────────────────────────────────────────────
#  Core data-fetch logic (adapted from the original script)
# ─────────────────────────────────────────────────────────────────────
def fetch_puts(ticker: str, strike: float, strike_range: float,
               min_volume: int, dte_max: int):
    """Fetch put option chains and return (current_price, dataframe)."""
    tk_obj = yf.Ticker(ticker)

    info = tk_obj.fast_info
    current_price = info.last_price

    expirations = tk_obj.options
    if not expirations:
        raise ValueError(f"No option data found for {ticker}. Check the ticker symbol.")

    today = date.today()
    rows = []

    for exp_str in expirations:
        exp_date = date.fromisoformat(exp_str)
        days_to_exp = (exp_date - today).days

        if days_to_exp <= 0:
            continue
        if days_to_exp > dte_max:
            continue

        try:
            chain = tk_obj.option_chain(exp_str)
            puts = chain.puts
        except Exception:
            continue

        lo = strike - strike_range
        hi = strike + strike_range
        filtered = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)].copy()
        filtered = filtered[filtered["volume"].fillna(0) >= min_volume]

        if filtered.empty:
            continue

        for _, row in filtered.iterrows():
            mid_price = (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 else row["lastPrice"]
            pct_return = (mid_price / row["strike"]) * 100 if row["strike"] > 0 else 0
            ann_return = (pct_return * 365 / days_to_exp) if days_to_exp > 0 else 0

            rows.append({
                "Contract":      row["contractSymbol"],
                "Expiry":        exp_str,
                "DTE":           days_to_exp,
                "Strike":        round(float(row["strike"]), 2),
                "Bid":           round(float(row["bid"]), 2),
                "Ask":           round(float(row["ask"]), 2),
                "Premium":       round(float(mid_price), 2),
                "Volume":        int(row["volume"]) if pd.notna(row["volume"]) else 0,
                "OI":            int(row["openInterest"]) if pd.notna(row["openInterest"]) else 0,
                "Ret %":         round(pct_return, 3),
                "Ann. Ret %":    round(ann_return, 2),
            })

    if not rows:
        raise ValueError(
            f"No puts found for {ticker} at strike ${strike} ±${strike_range} "
            f"with volume ≥ {min_volume}. Try widening the strike range "
            f"or lowering minimum volume."
        )

    return current_price, pd.DataFrame(rows)


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
#  GUI
# ─────────────────────────────────────────────────────────────────────
class CSPAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CSP Options Analyzer")
        self.geometry("1340x920")
        self.minsize(1120, 800)
        self.configure(fg_color=COLOR_BG)

        # Communication queue between worker thread and GUI
        self.result_queue: queue.Queue = queue.Queue()
        self.current_df: pd.DataFrame | None = None
        self.sort_state: dict[str, bool] = {}  # column -> ascending?
        self.chart_canvas: FigureCanvasTkAgg | None = None
        self.chart_fig = None

        self._build_layout()
        self.after(100, self._poll_queue)

    # ─────────────────────────────  layout  ───────────────────────────
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=320)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_panel()

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color=COLOR_CARD, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        # ── Title ────────────────────────────────────────────────
        title = ctk.CTkLabel(
            sidebar, text="CSP Analyzer",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLOR_TEXT,
        )
        title.grid(row=0, column=0, padx=24, pady=(28, 2), sticky="w")

        subtitle = ctk.CTkLabel(
            sidebar, text="Cash-Secured Put screener",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_DIM,
        )
        subtitle.grid(row=1, column=0, padx=24, pady=(0, 24), sticky="w")

        # Separator
        sep = ctk.CTkFrame(sidebar, height=1, fg_color=COLOR_BORDER)
        sep.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 18))

        # ── Inputs ────────────────────────────────────────────────
        inputs_label = ctk.CTkLabel(
            sidebar, text="PARAMETERS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_TEXT_DIM,
        )
        inputs_label.grid(row=3, column=0, padx=24, pady=(0, 10), sticky="w")

        # Input frame
        form = ctk.CTkFrame(sidebar, fg_color="transparent")
        form.grid(row=4, column=0, sticky="ew", padx=20)
        form.grid_columnconfigure(0, weight=1)

        self.entry_ticker  = self._add_input(form, 0, "Ticker", "KO")
        self.entry_strike  = self._add_input(form, 1, "Strike Price ($)", "75")
        self.entry_range   = self._add_input(form, 2, "Strike Range (±$)", "2.0")
        self.entry_volume  = self._add_input(form, 3, "Min Volume", "0")
        self.entry_top_n   = self._add_input(form, 4, "Top N", "5")
        self.entry_dte     = self._add_input(form, 5, "DTE Max (days)", "50")

        # ── Run button ────────────────────────────────────────────
        self.run_btn = ctk.CTkButton(
            sidebar, text="Run Analysis",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=44,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HI,
            command=self._on_run,
        )
        self.run_btn.grid(row=5, column=0, padx=24, pady=(22, 10), sticky="ew")

        # ── Status pill ───────────────────────────────────────────
        self.status_label = ctk.CTkLabel(
            sidebar, text="Ready",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_DIM,
        )
        self.status_label.grid(row=6, column=0, padx=24, pady=(0, 6), sticky="w")

        self.progress = ctk.CTkProgressBar(
            sidebar, mode="indeterminate",
            progress_color=COLOR_ACCENT,
            fg_color=COLOR_CARD_LIGHT,
            height=4,
        )
        self.progress.grid(row=7, column=0, padx=24, pady=(0, 24), sticky="ew")
        self.progress.set(0)

        # ── Footer info ───────────────────────────────────────────
        footer = ctk.CTkLabel(
            sidebar,
            text="Premiums use bid/ask mid.\nData via Yahoo Finance.",
            font=ctk.CTkFont(size=10),
            text_color=COLOR_TEXT_DIM,
            justify="left",
        )
        footer.grid(row=99, column=0, padx=24, pady=18, sticky="sw")
        sidebar.grid_rowconfigure(98, weight=1)

    def _add_input(self, parent, row, label_text, default):
        label = ctk.CTkLabel(
            parent, text=label_text,
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_DIM,
        )
        label.grid(row=row * 2, column=0, sticky="w", pady=(8, 2), padx=4)

        entry = ctk.CTkEntry(
            parent,
            height=36,
            fg_color=COLOR_CARD_LIGHT,
            border_color=COLOR_BORDER,
            border_width=1,
            text_color=COLOR_TEXT,
            font=ctk.CTkFont(size=13),
        )
        entry.insert(0, default)
        entry.grid(row=row * 2 + 1, column=0, sticky="ew", padx=4)
        entry.bind("<Return>", lambda e: self._on_run())
        return entry

    # ─────────────────────────────  main panel  ───────────────────────
    def _build_main_panel(self):
        main = ctk.CTkFrame(self, fg_color=COLOR_BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew", padx=24, pady=24)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=4, minsize=380)   # chart
        main.grid_rowconfigure(4, weight=1, minsize=150)   # table

        # ── Header / summary cards ────────────────────────────────
        header = ctk.CTkFrame(main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        for i in range(4):
            header.grid_columnconfigure(i, weight=1, uniform="card")

        self.card_ticker     = self._make_card(header, 0, "TICKER",   "—",   COLOR_TEXT)
        self.card_price      = self._make_card(header, 1, "PRICE",    "—",   COLOR_ACCENT_HI)
        self.card_best       = self._make_card(header, 2, "BEST ANN. RET.", "—", COLOR_SUCCESS)
        self.card_contracts  = self._make_card(header, 3, "CONTRACTS FOUND", "—", COLOR_WARNING)

        # ── Chart section ─────────────────────────────────────────
        chart_lbl = ctk.CTkLabel(
            main, text="PRICE CHART  —  DAILY CANDLES, 6 MONTHS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_DIM,
        )
        chart_lbl.grid(row=1, column=0, sticky="w", pady=(4, 8))

        chart_card = ctk.CTkFrame(main, fg_color=COLOR_CARD, corner_radius=10)
        chart_card.grid(row=2, column=0, sticky="nsew")
        chart_card.grid_columnconfigure(0, weight=1)
        chart_card.grid_rowconfigure(0, weight=1)

        self.chart_container = tk.Frame(chart_card, bg=COLOR_CARD, bd=0,
                                        highlightthickness=0)
        self.chart_container.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._chart_placeholder()

        # ── Table section ─────────────────────────────────────────
        section_lbl = ctk.CTkLabel(
            main, text="TOP CONTRACTS BY ANNUALIZED RETURN",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_DIM,
        )
        section_lbl.grid(row=3, column=0, sticky="w", pady=(16, 8))

        # ── Table ─────────────────────────────────────────────────
        table_frame = ctk.CTkFrame(main, fg_color=COLOR_CARD, corner_radius=10)
        table_frame.grid(row=4, column=0, sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        self._build_table(table_frame)

        # ── Notes ─────────────────────────────────────────────────
        notes = ctk.CTkLabel(
            main,
            text=("Premium = mid of bid/ask    •    Ret % = Premium ÷ Strike "
                  "   •    Ann. Ret % = Ret % × (365 ÷ DTE)\n"
                  "Mid-prices shown; actual fills may differ. Short puts carry "
                  "downside risk if the stock drops below the strike."),
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_DIM,
            justify="left",
        )
        notes.grid(row=5, column=0, sticky="w", pady=(12, 0))

    # ─────────────────────────────  chart  ────────────────────────────
    def _chart_placeholder(self, msg="Run an analysis to load the price chart."):
        if self.chart_fig is not None:
            plt.close(self.chart_fig)
            self.chart_fig = None
        self.chart_canvas = None
        for w in self.chart_container.winfo_children():
            w.destroy()
        lbl = ctk.CTkLabel(
            self.chart_container, text=msg,
            font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_DIM,
        )
        lbl.pack(expand=True)

    def _render_chart(self, ticker, hist, strike):
        if self.chart_fig is not None:
            plt.close(self.chart_fig)
            self.chart_fig = None
        for w in self.chart_container.winfo_children():
            w.destroy()

        if hist is None or hist.empty:
            self._chart_placeholder("Price history unavailable for this ticker.")
            return

        mc = mpf.make_marketcolors(
            up=COLOR_SUCCESS, down=COLOR_DANGER,
            edge="inherit", wick="inherit",
        )
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            facecolor=COLOR_CARD,
            figcolor=COLOR_CARD,
            edgecolor=COLOR_BORDER,
            gridcolor=COLOR_BORDER,
            gridstyle=":",
            rc={
                "axes.labelcolor": COLOR_TEXT_DIM,
                "xtick.color": COLOR_TEXT_DIM,
                "ytick.color": COLOR_TEXT_DIM,
                "axes.edgecolor": COLOR_BORDER,
                "text.color": COLOR_TEXT,
            },
        )

        fig, axlist = mpf.plot(
            hist, type="candle", style=style, returnfig=True,
            volume=False, figsize=(9, 5.0), ylabel="",
            hlines=dict(hlines=[strike], colors=[COLOR_WARNING],
                        linestyle="--", linewidths=1.2),
            datetime_format="%b %d", xrotation=0,
        )
        ax = axlist[0]
        ax.set_title(
            f"{ticker}  —  Daily, 6 Months",
            color=COLOR_TEXT, fontsize=11, loc="left", pad=10,
        )
        # Label the strike line at the right edge
        ax.text(
            1.01, strike, f"${strike:.2f}\n{(strike - hist['Close'].iloc[-1])/hist['Close'].iloc[-1]*100:+.1f}%",
            color="#1a1d24", fontsize=8.5, fontweight="bold",
            va="center", ha="left",
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.25", fc=COLOR_WARNING, ec="none", alpha=1),
        )

        self.chart_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)
        self.chart_canvas = canvas

    def _make_card(self, parent, col, title, value, value_color):
        card = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=10, height=86)
        card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0))
        card.grid_propagate(False)

        title_lbl = ctk.CTkLabel(
            card, text=title,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLOR_TEXT_DIM,
        )
        title_lbl.place(relx=0.5, y=18, anchor="center")

        value_lbl = ctk.CTkLabel(
            card, text=value,
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=value_color,
        )
        value_lbl.place(relx=0.5, rely=0.5, y=8, anchor="center")
        return value_lbl

    def _build_table(self, parent):
        # Style the ttk.Treeview to match the dark theme
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "CSP.Treeview",
            background=COLOR_CARD,
            foreground=COLOR_TEXT,
            fieldbackground=COLOR_CARD,
            rowheight=30,
            borderwidth=0,
            font=("Helvetica", 12),
        )
        style.configure(
            "CSP.Treeview.Heading",
            background=COLOR_CARD_LIGHT,
            foreground=COLOR_TEXT,
            relief="flat",
            font=("Helvetica", 11, "bold"),
            padding=(10, 8),
        )
        style.map(
            "CSP.Treeview.Heading",
            background=[("active", COLOR_BORDER)],
        )
        style.map(
            "CSP.Treeview",
            background=[("selected", COLOR_ACCENT)],
            foreground=[("selected", "#ffffff")],
        )
        style.layout("CSP.Treeview", [
            ("CSP.Treeview.treearea", {"sticky": "nswe"}),
        ])

        columns = ("Expiry", "DTE", "Strike", "Bid", "Ask",
                   "Premium", "Volume", "OI", "Ret %", "Ann. Ret %")
        widths  = (96, 56, 78, 70, 70, 88, 80, 80, 80, 110)

        container = tk.Frame(parent, bg=COLOR_CARD, bd=0)
        container.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            container,
            columns=columns,
            show="headings",
            style="CSP.Treeview",
            selectmode="browse",
        )
        for col, w in zip(columns, widths):
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._sort_by(c))
            anchor = "w" if col == "Expiry" else "e"
            self.tree.column(col, width=w, anchor=anchor, stretch=True)

        # Row colors (zebra striping + highlight)
        self.tree.tag_configure("odd",        background=COLOR_CARD)
        self.tree.tag_configure("even",       background=COLOR_CARD_LIGHT)
        self.tree.tag_configure("best",       background="#0f2e22",
                                 foreground=COLOR_SUCCESS)

        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Placeholder text
        self._show_placeholder("Set your parameters on the left and click Run Analysis.")

    # ─────────────────────────────  table helpers  ────────────────────
    def _show_placeholder(self, msg):
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", "end",
                         values=(msg, "", "", "", "", "", "", "", "", ""),
                         tags=("odd",))

    def _populate_table(self, df: pd.DataFrame, top_n: int):
        self.tree.delete(*self.tree.get_children())
        df_sorted = df.sort_values("Ann. Ret %", ascending=False).head(top_n)
        self.current_df = df_sorted.reset_index(drop=True)

        for i, row in self.current_df.iterrows():
            values = (
                row["Expiry"], row["DTE"], f"{row['Strike']:.2f}",
                f"{row['Bid']:.2f}", f"{row['Ask']:.2f}",
                f"{row['Premium']:.2f}", f"{row['Volume']:,}",
                f"{row['OI']:,}", f"{row['Ret %']:.3f}",
                f"{row['Ann. Ret %']:.2f}",
            )
            if i == 0:
                tag = "best"
            else:
                tag = "even" if i % 2 == 0 else "odd"
            self.tree.insert("", "end", values=values, tags=(tag,))

    def _sort_by(self, col):
        if self.current_df is None or self.current_df.empty:
            return
        col_map = {
            "Expiry": "Expiry", "DTE": "DTE", "Strike": "Strike",
            "Bid": "Bid", "Ask": "Ask", "Premium": "Premium",
            "Volume": "Volume", "OI": "OI", "Ret %": "Ret %",
            "Ann. Ret %": "Ann. Ret %",
        }
        key = col_map[col]
        asc = not self.sort_state.get(col, False)
        self.sort_state[col] = asc
        df = self.current_df.sort_values(key, ascending=asc).reset_index(drop=True)
        self.current_df = df

        self.tree.delete(*self.tree.get_children())
        for i, row in df.iterrows():
            values = (
                row["Expiry"], row["DTE"], f"{row['Strike']:.2f}",
                f"{row['Bid']:.2f}", f"{row['Ask']:.2f}",
                f"{row['Premium']:.2f}", f"{row['Volume']:,}",
                f"{row['OI']:,}", f"{row['Ret %']:.3f}",
                f"{row['Ann. Ret %']:.2f}",
            )
            tag = "even" if i % 2 == 0 else "odd"
            self.tree.insert("", "end", values=values, tags=(tag,))

    # ─────────────────────────────  run handler  ──────────────────────
    def _on_run(self):
        try:
            ticker  = self.entry_ticker.get().strip().upper()
            strike  = float(self.entry_strike.get())
            srange  = float(self.entry_range.get())
            vol     = int(self.entry_volume.get())
            top_n   = int(self.entry_top_n.get())
            dte_max = int(self.entry_dte.get())
            if not ticker:
                raise ValueError("Ticker is required.")
        except ValueError as e:
            self._set_status(f"Invalid input: {e}", COLOR_DANGER)
            return

        self.run_btn.configure(state="disabled", text="Running…")
        self._set_status(f"Fetching options for {ticker}…", COLOR_ACCENT_HI)
        self.progress.start()
        self._show_placeholder(f"Fetching put chain for {ticker}…")

        params = (ticker, strike, srange, vol, top_n, dte_max)
        threading.Thread(target=self._worker, args=params, daemon=True).start()

    def _worker(self, ticker, strike, srange, vol, top_n, dte_max):
        try:
            price, df = fetch_puts(ticker, strike, srange, vol, dte_max)
            try:
                hist = fetch_history(ticker)
            except Exception:
                hist = None
            self.result_queue.put(("ok", ticker, price, df, top_n, strike, hist))
        except Exception as e:
            self.result_queue.put(("err", str(e)))

    def _poll_queue(self):
        try:
            while True:
                msg = self.result_queue.get_nowait()
                if msg[0] == "ok":
                    _, ticker, price, df, top_n, strike, hist = msg
                    self._on_success(ticker, price, df, top_n, strike, hist)
                else:
                    self._on_error(msg[1])
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _on_success(self, ticker, price, df, top_n, strike, hist):
        self.progress.stop()
        self.progress.set(0)
        self.run_btn.configure(state="normal", text="Run Analysis")

        self.card_ticker.configure(text=ticker)
        self.card_price.configure(text=f"${price:.2f}")
        best = df["Ann. Ret %"].max()
        self.card_best.configure(text=f"{best:.2f}%")
        self.card_contracts.configure(text=str(len(df)))

        self._render_chart(ticker, hist, strike)
        self._populate_table(df, top_n)
        self._set_status(
            f"Found {len(df)} contracts • showing top {min(top_n, len(df))}",
            COLOR_SUCCESS,
        )

    def _on_error(self, msg):
        self.progress.stop()
        self.progress.set(0)
        self.run_btn.configure(state="normal", text="Run Analysis")
        self._set_status(msg, COLOR_DANGER)
        self._show_placeholder("No results — adjust parameters and try again.")
        self._chart_placeholder("No chart — adjust parameters and try again.")
        self.card_ticker.configure(text="—")
        self.card_price.configure(text="—")
        self.card_best.configure(text="—")
        self.card_contracts.configure(text="—")

    def _set_status(self, text, color):
        # Truncate long messages
        display = text if len(text) <= 60 else text[:57] + "…"
        self.status_label.configure(text=display, text_color=color)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = CSPAnalyzerApp()
    app.mainloop()