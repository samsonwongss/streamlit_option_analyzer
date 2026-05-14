# Options Analyzer (Streamlit)

A web version of the desktop Options Analyzer. Screens **Cash-Secured Put** and
**Covered Call** opportunities from live Yahoo Finance option chains, with an
interactive Plotly candlestick chart.

- **Cash-Secured Put** — Ret % = Premium ÷ Strike
- **Covered Call** — Ret % = Premium ÷ Cost per share (leave "Cost basis" blank to use the live price); also shows **If-Called %** = premium yield + capital gain to the strike if assigned.

## Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | The app |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Dark theme + server settings |

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Push this `streamlit_options_analyzer/` folder to a GitHub repository.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app**, pick the repo and branch.
4. Set **Main file path** to `streamlit_app.py` (or
   `streamlit_options_analyzer/streamlit_app.py` if the folder is not the repo root).
5. Click **Deploy**. Streamlit installs `requirements.txt` automatically.

## Notes

- Option-chain and price data are cached for 5 minutes (`@st.cache_data(ttl=300)`)
  to keep reruns fast and avoid hammering Yahoo Finance. Use the menu →
  *Clear cache* to force a refresh.
- Premiums use the bid/ask mid; actual fills may differ. This tool is for
  research only and is not financial advice.
