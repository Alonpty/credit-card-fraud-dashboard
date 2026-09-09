# 🕵️ Credit Card Fraud Detection Dashboard

A colorful **Streamlit** dashboard for hunting fraud in `credit_card_fraud_10k.csv`
(10,000 transactions, 151 frauds = **1.51%** base rate).

## ▶️ Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or just double-click **`run_dashboard.bat`**. The app opens at http://localhost:8501.

## 🎯 The risk score

Every transaction gets an explainable **0-100 risk score** from the signals that actually
drive fraud in this data (each rule's weight follows its measured fraud rate):

| Red flag | Points | Fraud rate when present |
|---|---|---|
| 🌍 Foreign transaction | +25 | 8.4% (5.6× base) |
| 📍 Location mismatch | +25 | 8.4% (5.6× base) |
| 📱 Device trust ≤ 25 / 26-50 / 51-75 | +25 / +15 / +3 | 6.9% / 3.6% / 0.4% |
| ⚡ Velocity 5-7 in 24h | +20 | 10.6% (7× base) |
| 🌙 Night 00:00-05:59 | +20 | 5.1% (3.3× base) |
| 💸 Amount in top 1% / top 5% | +14 / +8 | mild |

Bands: **Low** <25 · **Medium** 25-49 · **High** 50-74 · **Critical** ≥75.

**How well it works:** reviewing the top 300 of 10,000 transactions catches
**97% of all fraud** at a 49% hit rate — 32× better than random. The Low band
(7,178 transactions) contains **zero** frauds.

## 🗂️ Tabs

| Tab | What it gives you |
|---|---|
| 🔎 **Fraud Finder** | Risk-ranked review queue with the red flags spelled out per row, precision/recall scorecard, CSV export |
| 📊 **Overview** | Fraud split, fraud rate by merchant and by hour, amount distributions, age breakdown |
| 🚩 **Risk Signals** | Lift chart per red flag, hour × merchant heatmap, velocity curve, trust-vs-amount scatter |
| 🧮 **Risk Checker** | Score a single transaction — gauge, verdict, triggered flags, historical rate of similar transactions |
| 📋 **Data Explorer** | Search by ID, sort, summary stats, download the filtered set |

The sidebar filters (merchant, amount, hour, device trust, velocity, age, foreign,
mismatch, class, risk band) drive every tab at once.

## 📝 Note

Tables are rendered as custom HTML instead of `st.dataframe`, because Windows
Application Control blocks `pyarrow`'s DLL on this machine.
