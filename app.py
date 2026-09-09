"""
🕵️ Credit Card Fraud Detection Dashboard
=========================================
A colorful Streamlit dashboard over `credit_card_fraud_10k.csv`.

Run with:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# 1. PAGE SETUP + COLOR SYSTEM
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "credit_card_fraud_10k.csv"

FRAUD = "#FF2E63"      # hot pink / red   -> fraud
SAFE = "#08D9D6"       # cyan             -> legit
GOLD = "#FFD93D"
PURPLE = "#7B61FF"
GREEN = "#38E54D"
ORANGE = "#FF9F1C"

PALETTE = [FRAUD, SAFE, GOLD, PURPLE, GREEN]

BAND_COLORS = {
    "Low": GREEN,
    "Medium": GOLD,
    "High": ORANGE,
    "Critical": FRAUD,
}
BAND_ORDER = ["Low", "Medium", "High", "Critical"]

CARD_GRADIENTS = [
    "linear-gradient(135deg,#7B61FF 0%,#4A2FBD 100%)",
    "linear-gradient(135deg,#FF2E63 0%,#B3003C 100%)",
    "linear-gradient(135deg,#FF9F1C 0%,#D45500 100%)",
    "linear-gradient(135deg,#08D9D6 0%,#0089A7 100%)",
    "linear-gradient(135deg,#38E54D 0%,#0F9D58 100%)",
]

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        .hero {
            background: linear-gradient(90deg,#FF2E63 0%,#7B61FF 45%,#08D9D6 100%);
            padding: 22px 28px; border-radius: 18px; margin-bottom: 22px;
            box-shadow: 0 8px 26px rgba(123,97,255,.35);
        }
        .hero h1 {color:#fff; margin:0; font-size:2.1rem; letter-spacing:-.5px;}
        .hero p  {color:rgba(255,255,255,.92); margin:.35rem 0 0 0; font-size:1rem;}
        .kpi {
            border-radius:16px; padding:16px 18px; color:#fff; height:100%;
            box-shadow:0 6px 18px rgba(0,0,0,.35);
        }
        .kpi .label {font-size:.78rem; text-transform:uppercase; letter-spacing:1.2px;
                     opacity:.9; font-weight:600;}
        .kpi .value {font-size:1.95rem; font-weight:800; line-height:1.25; margin-top:2px;}
        .kpi .sub   {font-size:.78rem; opacity:.9;}
        .badge {
            display:inline-block; padding:4px 12px; border-radius:999px;
            font-size:.8rem; font-weight:700; color:#111; margin:3px 6px 3px 0;
        }
        .stTabs [data-baseweb="tab"] {font-size:1.02rem; font-weight:600; padding:10px 18px;}
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg,rgba(255,46,99,.22),rgba(123,97,255,.22));
            border-radius:10px 10px 0 0;
        }
        div[data-testid="stMetricValue"] {font-size:1.6rem;}
        /* --- custom HTML tables (st.dataframe needs pyarrow, blocked on this machine) --- */
        .tbl-wrap {
            overflow:auto; border-radius:14px; border:1px solid #2A2F45;
            box-shadow:0 6px 18px rgba(0,0,0,.3);
        }
        table.fx {width:100%; border-collapse:collapse; font-size:.85rem;}
        table.fx th {
            position:sticky; top:0; z-index:2; background:#1C2030; color:#FFD93D;
            padding:11px 9px; text-align:left; white-space:nowrap; font-weight:700;
            border-bottom:2px solid #FF2E63;
        }
        table.fx td {padding:7px 9px; border-top:1px solid rgba(255,255,255,.06); white-space:nowrap;}
        table.fx td.flags {white-space:normal; min-width:260px; font-size:.8rem; opacity:.95;}
        table.fx tr.fraud td {background:rgba(255,46,99,.16);}
        table.fx tbody tr:hover td {background:rgba(123,97,255,.22);}
        .bar-bg {background:rgba(255,255,255,.10); border-radius:6px; width:74px;
                 height:9px; display:inline-block; vertical-align:middle; overflow:hidden;}
        .bar-fg {height:9px; border-radius:6px; display:block;}
        .pill {padding:2px 9px; border-radius:999px; font-size:.72rem; font-weight:800; color:#111;}
    </style>
    """,
    unsafe_allow_html=True,
)

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=13),
    margin=dict(l=10, r=10, t=55, b=10),
)


# ----------------------------------------------------------------------------
# 2. DATA LAYER
# ----------------------------------------------------------------------------

HOUR_BANDS = ["🌙 Night 0-5", "🌅 Morning 6-11", "☀️ Afternoon 12-17", "🌆 Evening 18-23"]


def score_transactions(df: pd.DataFrame, p95: float, p99: float):
    """Explainable rule-based risk score (0-100) + the reasons behind it.

    Weights come from the measured fraud rate of each signal in this dataset.
    """
    n = len(df)
    score = np.zeros(n, dtype=float)
    reasons = [[] for _ in range(n)]

    rules = [
        (df["foreign_transaction"] == 1, 25, "🌍 Foreign transaction"),
        (df["location_mismatch"] == 1, 25, "📍 Location mismatch"),
        (df["device_trust_score"] <= 25, 25, "📱 Untrusted device"),
        (df["device_trust_score"].between(26, 50), 15, "📱 Low device trust"),
        (df["device_trust_score"].between(51, 75), 3, "📱 Medium device trust"),
        (df["velocity_last_24h"].between(5, 7), 20, "⚡ Velocity burst (5-7/24h)"),
        (df["velocity_last_24h"] >= 8, 5, "⚡ Very high velocity"),
        (df["transaction_hour"] <= 5, 20, "🌙 Night-time (00:00-05:59)"),
        (df["amount"] > p99, 14, "💸 Top 1% amount"),
        ((df["amount"] > p95) & (df["amount"] <= p99), 8, "💰 Top 5% amount"),
    ]

    for mask, points, label in rules:
        hit = np.asarray(mask, dtype=bool)
        score += hit * points
        for i in np.flatnonzero(hit):
            reasons[i].append(label)

    score = np.clip(score, 0, 100)
    reason_text = [" · ".join(r) if r else "— no red flags" for r in reasons]
    band = np.select(
        [score >= 75, score >= 50, score >= 25],
        ["Critical", "High", "Medium"],
        default="Low",
    )
    return score, reason_text, band


@st.cache_data(show_spinner="Loading transactions…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df["hour_band"] = pd.cut(
        df["transaction_hour"], bins=[-1, 5, 11, 17, 23], labels=HOUR_BANDS
    ).astype(str)
    df["trust_band"] = pd.cut(
        df["device_trust_score"],
        bins=[-1, 25, 50, 75, 100],
        labels=["0-25 (untrusted)", "26-50 (low)", "51-75 (medium)", "76-100 (trusted)"],
    ).astype(str)
    df["age_band"] = pd.cut(
        df["cardholder_age"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["≤25", "26-35", "36-45", "46-55", "56+"],
    ).astype(str)
    df["status"] = np.where(df["is_fraud"] == 1, "🚨 Fraud", "✅ Legit")

    p95, p99 = df["amount"].quantile([0.95, 0.99])
    score, reasons, band = score_transactions(df, p95, p99)
    df["risk_score"] = score
    df["risk_reasons"] = reasons
    df["risk_band"] = band

    df.attrs["p95"] = float(p95)
    df.attrs["p99"] = float(p99)
    return df


data = load_data()
P95, P99 = data.attrs["p95"], data.attrs["p99"]
BASE_RATE = data["is_fraud"].mean()


# ----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎛️ Filters")

    if st.button("🔄 Reset all filters", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("f_"):
                del st.session_state[key]
        st.rerun()

    categories = sorted(data["merchant_category"].unique())
    sel_cats = st.multiselect("🏪 Merchant category", categories, default=categories, key="f_cats")

    sel_class = st.radio(
        "🏷️ Transaction class",
        ["All", "🚨 Fraud only", "✅ Legit only"],
        horizontal=False,
        key="f_class",
    )
    sel_bands = st.multiselect("🎯 Risk band", BAND_ORDER, default=BAND_ORDER, key="f_bands")

    st.markdown("---")
    amt_min, amt_max = float(data["amount"].min()), float(data["amount"].max())
    sel_amount = st.slider("💵 Amount ($)", amt_min, amt_max, (amt_min, amt_max), key="f_amt")
    sel_hour = st.slider("🕐 Transaction hour", 0, 23, (0, 23), key="f_hour")
    sel_trust = st.slider("📱 Device trust score", 0, 99, (0, 99), key="f_trust")
    sel_velocity = st.slider("⚡ Velocity last 24h", 0, 9, (0, 9), key="f_vel")
    sel_age = st.slider("🎂 Cardholder age", 18, 69, (18, 69), key="f_age")

    st.markdown("---")
    sel_foreign = st.radio("🌍 Foreign transaction", ["All", "Yes", "No"], horizontal=True, key="f_for")
    sel_mismatch = st.radio("📍 Location mismatch", ["All", "Yes", "No"], horizontal=True, key="f_mis")

    st.markdown("---")
    st.caption(
        f"Dataset: **{len(data):,}** transactions · "
        f"**{int(data['is_fraud'].sum())}** confirmed frauds ({BASE_RATE:.2%})"
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    m = df["merchant_category"].isin(sel_cats or categories)
    m &= df["risk_band"].isin(sel_bands or BAND_ORDER)
    m &= df["amount"].between(*sel_amount)
    m &= df["transaction_hour"].between(*sel_hour)
    m &= df["device_trust_score"].between(*sel_trust)
    m &= df["velocity_last_24h"].between(*sel_velocity)
    m &= df["cardholder_age"].between(*sel_age)
    if sel_class == "🚨 Fraud only":
        m &= df["is_fraud"] == 1
    elif sel_class == "✅ Legit only":
        m &= df["is_fraud"] == 0
    if sel_foreign != "All":
        m &= df["foreign_transaction"] == (1 if sel_foreign == "Yes" else 0)
    if sel_mismatch != "All":
        m &= df["location_mismatch"] == (1 if sel_mismatch == "Yes" else 0)
    return df[m]


view = apply_filters(data)


# ----------------------------------------------------------------------------
# 4. HERO + KPI CARDS
# ----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🕵️ Credit Card Fraud Detection Dashboard</h1>
        <p>Hunt down fraudulent transactions in 10,000 payments — filter, rank by risk,
           and inspect every red flag.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def kpi_card(col, label, value, sub, gradient):
    col.markdown(
        f"""<div class="kpi" style="background:{gradient}">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


n_tx = len(view)
n_fraud = int(view["is_fraud"].sum())
rate = view["is_fraud"].mean() if n_tx else 0.0
fraud_loss = float(view.loc[view["is_fraud"] == 1, "amount"].sum())
n_critical = int((view["risk_band"] == "Critical").sum())

c1, c2, c3, c4, c5 = st.columns(5)
kpi_card(c1, "Transactions", f"{n_tx:,}", f"{n_tx / len(data):.1%} of dataset", CARD_GRADIENTS[0])
kpi_card(c2, "Confirmed fraud", f"{n_fraud:,}", f"of {int(data['is_fraud'].sum())} total", CARD_GRADIENTS[1])
kpi_card(c3, "Fraud rate", f"{rate:.2%}", f"baseline {BASE_RATE:.2%}", CARD_GRADIENTS[2])
kpi_card(c4, "Money at risk", f"${fraud_loss:,.0f}", "sum of fraud amounts", CARD_GRADIENTS[3])
kpi_card(c5, "Critical risk", f"{n_critical:,}", "score ≥ 75 — review now", CARD_GRADIENTS[4])

st.markdown("")

if view.empty:
    st.warning("😶 No transactions match the current filters. Reset them in the sidebar.")
    st.stop()


# ----------------------------------------------------------------------------
# 5. TABS
# ----------------------------------------------------------------------------

tab_find, tab_overview, tab_signals, tab_checker, tab_data = st.tabs(
    ["🔎 Fraud Finder", "📊 Overview", "🚩 Risk Signals", "🧮 Risk Checker", "📋 Data Explorer"]
)

TABLE_COLS = [
    "transaction_id", "risk_score", "risk_band", "status", "amount",
    "transaction_hour", "merchant_category", "foreign_transaction",
    "location_mismatch", "device_trust_score", "velocity_last_24h",
    "cardholder_age", "risk_reasons",
]

HEADERS = ["ID", "🎯 Risk", "Band", "Actual", "💵 Amount", "🕐 Hour", "🏪 Merchant",
           "🌍 Foreign", "📍 Mismatch", "📱 Trust", "⚡ Vel.", "🎂 Age", "🚩 Red flags"]

MAX_RENDER_ROWS = 400


def _bar(value: float, color: str) -> str:
    return (f'<span class="bar-bg"><span class="bar-fg" '
            f'style="width:{max(0, min(100, value)):.0f}%;background:{color}"></span></span> '
            f'<b>{value:.0f}</b>')


def show_table(df: pd.DataFrame, height: int = 520):
    """Colorful HTML table (st.dataframe needs pyarrow, blocked by this machine's policy)."""
    shown = df.head(MAX_RENDER_ROWS)
    rows = []
    for r in shown[TABLE_COLS].itertuples(index=False):
        band_color = BAND_COLORS.get(r.risk_band, GREEN)
        trust_color = FRAUD if r.device_trust_score <= 25 else (
            ORANGE if r.device_trust_score <= 50 else SAFE)
        rows.append(
            f'<tr class="{"fraud" if r.status.startswith("🚨") else ""}">'
            f"<td>{r.transaction_id}</td>"
            f"<td>{_bar(r.risk_score, band_color)}</td>"
            f'<td><span class="pill" style="background:{band_color}">{r.risk_band}</span></td>'
            f"<td>{r.status}</td>"
            f"<td><b>${r.amount:,.2f}</b></td>"
            f"<td>{r.transaction_hour:02d}:00</td>"
            f"<td>{r.merchant_category}</td>"
            f'<td>{"🌍 yes" if r.foreign_transaction else "—"}</td>'
            f'<td>{"📍 yes" if r.location_mismatch else "—"}</td>'
            f"<td>{_bar(r.device_trust_score, trust_color)}</td>"
            f"<td>{r.velocity_last_24h}</td>"
            f"<td>{r.cardholder_age}</td>"
            f'<td class="flags">{r.risk_reasons}</td>'
            "</tr>"
        )
    head = "".join(f"<th>{h}</th>" for h in HEADERS)
    st.markdown(
        f'<div class="tbl-wrap" style="max-height:{height}px">'
        f'<table class="fx"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    if len(df) > MAX_RENDER_ROWS:
        st.caption(f"Showing the first {MAX_RENDER_ROWS:,} of {len(df):,} rows — "
                   "use the download button for the full list.")


# --- 5.1 Fraud Finder -------------------------------------------------------
with tab_find:
    st.subheader("🔎 Riskiest transactions first")
    st.caption(
        "Every transaction gets an explainable 0-100 risk score built from the signals "
        "that actually drive fraud in this dataset. Work the list top-down."
    )

    left, right = st.columns([3, 1])
    top_n = left.slider(
        "How many of the riskiest transactions to review?",
        10, min(2000, len(view)), min(100, len(view)), step=10,
    )
    min_score = right.number_input("Minimum risk score", 0, 100, 0, step=5)

    flagged = (
        view[view["risk_score"] >= min_score]
        .sort_values(["risk_score", "amount"], ascending=False)
        .head(top_n)
    )

    caught = int(flagged["is_fraud"].sum())
    total_fraud = int(view["is_fraud"].sum())
    precision = caught / len(flagged) if len(flagged) else 0.0
    recall = caught / total_fraud if total_fraud else 0.0
    lift = precision / BASE_RATE if BASE_RATE else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚨 Frauds in this list", f"{caught}", f"of {total_fraud} in view")
    m2.metric("🎯 Hit rate (precision)", f"{precision:.1%}", f"baseline {BASE_RATE:.2%}")
    m3.metric("🕸️ Fraud caught (recall)", f"{recall:.1%}")
    m4.metric("🚀 Better than random", f"{lift:.1f}×")

    st.progress(min(recall, 1.0), text=f"Reviewing {len(flagged):,} transactions catches {recall:.0%} of all fraud")

    show_table(flagged)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇️ Download this review queue (CSV)",
        flagged[TABLE_COLS].to_csv(index=False).encode("utf-8"),
        file_name="fraud_review_queue.csv",
        mime="text/csv",
        use_container_width=True,
    )
    dl2.download_button(
        "🚨 Download confirmed frauds (CSV)",
        view[view["is_fraud"] == 1][TABLE_COLS].to_csv(index=False).encode("utf-8"),
        file_name="confirmed_frauds.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### 🎯 How well each risk band separates fraud")
    band_stats = (
        view.groupby("risk_band")
        .agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"))
        .reindex(BAND_ORDER)
        .fillna(0)
        .reset_index()
    )
    band_stats["fraud_rate"] = np.where(
        band_stats["transactions"] > 0, band_stats["frauds"] / band_stats["transactions"], 0
    )
    fig = px.bar(
        band_stats,
        x="risk_band", y="fraud_rate", text=band_stats["fraud_rate"].map("{:.1%}".format),
        color="risk_band", color_discrete_map=BAND_COLORS,
        title="Fraud rate by risk band (dashed line = dataset baseline)",
        hover_data=["transactions", "frauds"],
    )
    fig.add_hline(y=BASE_RATE, line_dash="dash", line_color="white",
                  annotation_text=f"baseline {BASE_RATE:.2%}")
    fig.update_traces(textposition="outside")
    fig.update_layout(**CHART_LAYOUT, showlegend=False, yaxis_tickformat=".1%",
                      xaxis_title="", yaxis_title="Fraud rate")
    st.plotly_chart(fig, use_container_width=True)


# --- 5.2 Overview -----------------------------------------------------------
with tab_overview:
    a, b = st.columns([1, 2])

    donut = view["status"].value_counts().reset_index()
    donut.columns = ["status", "count"]
    fig = px.pie(
        donut, names="status", values="count", hole=0.58,
        color="status", color_discrete_map={"🚨 Fraud": FRAUD, "✅ Legit": SAFE},
        title="Fraud vs legit split",
    )
    fig.update_traces(textinfo="percent+label", textfont_size=14)
    fig.update_layout(**CHART_LAYOUT)
    a.plotly_chart(fig, use_container_width=True)

    by_cat = (
        view.groupby("merchant_category")
        .agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"))
        .reset_index()
    )
    by_cat["fraud_rate"] = by_cat["frauds"] / by_cat["transactions"]
    by_cat = by_cat.sort_values("fraud_rate", ascending=False)
    fig = px.bar(
        by_cat, x="merchant_category", y="fraud_rate", color="merchant_category",
        color_discrete_sequence=PALETTE, text=by_cat["fraud_rate"].map("{:.2%}".format),
        title="🏪 Fraud rate by merchant category", hover_data=["transactions", "frauds"],
    )
    fig.add_hline(y=BASE_RATE, line_dash="dash", line_color="white")
    fig.update_traces(textposition="outside")
    fig.update_layout(**CHART_LAYOUT, showlegend=False, yaxis_tickformat=".1%",
                      xaxis_title="", yaxis_title="Fraud rate")
    b.plotly_chart(fig, use_container_width=True)

    by_hour = (
        view.groupby("transaction_hour")
        .agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"))
        .reindex(range(24), fill_value=0)
        .reset_index()
        .rename(columns={"index": "transaction_hour"})
    )
    by_hour["fraud_rate"] = np.where(
        by_hour["transactions"] > 0, by_hour["frauds"] / by_hour["transactions"], 0
    )
    fig = go.Figure()
    fig.add_bar(x=by_hour["transaction_hour"], y=by_hour["transactions"],
                name="Transactions", marker_color="rgba(123,97,255,.45)", yaxis="y2")
    fig.add_scatter(x=by_hour["transaction_hour"], y=by_hour["fraud_rate"],
                    name="Fraud rate", mode="lines+markers",
                    line=dict(color=FRAUD, width=4), marker=dict(size=9))
    fig.add_vrect(x0=-0.5, x1=5.5, fillcolor=FRAUD, opacity=0.12, line_width=0,
                  annotation_text="🌙 night danger zone", annotation_position="top left")
    fig.update_layout(
        **CHART_LAYOUT,
        title="🕐 Fraud rate by hour of day (bars = volume)",
        xaxis=dict(title="Hour", dtick=1),
        yaxis=dict(title="Fraud rate", tickformat=".1%"),
        yaxis2=dict(title="Transactions", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.08, x=0.55),
    )
    st.plotly_chart(fig, use_container_width=True)

    c, d = st.columns(2)
    fig = px.histogram(
        view, x="amount", color="status", nbins=60, barmode="overlay", histnorm="percent",
        color_discrete_map={"🚨 Fraud": FRAUD, "✅ Legit": SAFE},
        title="💵 Amount distribution — fraud vs legit (% within class)",
        marginal="box",
    )
    fig.update_traces(opacity=0.75)
    fig.update_layout(**CHART_LAYOUT, xaxis_title="Amount ($)", yaxis_title="% of class",
                      legend_title="")
    c.plotly_chart(fig, use_container_width=True)

    by_age = (
        view.groupby("age_band")
        .agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"))
        .reset_index()
    )
    by_age["fraud_rate"] = by_age["frauds"] / by_age["transactions"]
    fig = px.bar(
        by_age, x="age_band", y="frauds", color="fraud_rate",
        color_continuous_scale=["#08D9D6", "#FFD93D", "#FF2E63"],
        text="frauds", title="🎂 Fraud count by cardholder age (color = fraud rate)",
        hover_data=["transactions", "fraud_rate"],
        category_orders={"age_band": ["≤25", "26-35", "36-45", "46-55", "56+"]},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**CHART_LAYOUT, xaxis_title="", yaxis_title="Frauds",
                      coloraxis_colorbar_tickformat=".1%")
    d.plotly_chart(fig, use_container_width=True)


# --- 5.3 Risk Signals -------------------------------------------------------
with tab_signals:
    st.subheader("🚩 Which red flags actually predict fraud?")

    signals = {
        "🌍 Foreign transaction": view["foreign_transaction"] == 1,
        "📍 Location mismatch": view["location_mismatch"] == 1,
        "📱 Device trust ≤ 25": view["device_trust_score"] <= 25,
        "📱 Device trust 26-50": view["device_trust_score"].between(26, 50),
        "⚡ Velocity 5-7 / 24h": view["velocity_last_24h"].between(5, 7),
        "🌙 Night 00:00-05:59": view["transaction_hour"] <= 5,
        "💸 Amount in top 5%": view["amount"] > P95,
    }
    rows = []
    for name, mask in signals.items():
        sub = view[mask]
        if len(sub) == 0:
            continue
        r = sub["is_fraud"].mean()
        rows.append({
            "signal": name, "fraud_rate": r, "transactions": len(sub),
            "frauds": int(sub["is_fraud"].sum()),
            "lift": r / BASE_RATE if BASE_RATE else 0,
        })
    lift_df = pd.DataFrame(rows).sort_values("fraud_rate")

    if not lift_df.empty:
        fig = px.bar(
            lift_df, x="fraud_rate", y="signal", orientation="h", color="lift",
            color_continuous_scale=["#08D9D6", "#FFD93D", "#FF2E63"],
            text=lift_df.apply(lambda r: f"{r['fraud_rate']:.1%}  ({r['lift']:.1f}× base)", axis=1),
            title="Fraud rate when each red flag is present",
            hover_data=["transactions", "frauds"],
        )
        fig.add_vline(x=BASE_RATE, line_dash="dash", line_color="white",
                      annotation_text=f"baseline {BASE_RATE:.2%}")
        fig.update_traces(textposition="outside")
        fig.update_layout(**CHART_LAYOUT, xaxis_tickformat=".1%", xaxis_title="Fraud rate",
                          yaxis_title="", height=460,
                          xaxis_range=[0, lift_df["fraud_rate"].max() * 1.35])
        st.plotly_chart(fig, use_container_width=True)

    e, f = st.columns(2)

    pivot = view.pivot_table(index="hour_band", columns="merchant_category",
                             values="is_fraud", aggfunc="mean")
    pivot = pivot.reindex([h for h in HOUR_BANDS if h in pivot.index])
    fig = px.imshow(
        pivot, text_auto=".2%", aspect="auto",
        color_continuous_scale=["#0E1117", "#7B61FF", "#FF2E63"],
        title="🔥 Fraud rate heatmap — time of day × merchant",
    )
    fig.update_layout(**CHART_LAYOUT, xaxis_title="", yaxis_title="")
    e.plotly_chart(fig, use_container_width=True)

    by_vel = (
        view.groupby("velocity_last_24h")
        .agg(transactions=("is_fraud", "size"), frauds=("is_fraud", "sum"))
        .reset_index()
    )
    by_vel["fraud_rate"] = by_vel["frauds"] / by_vel["transactions"]
    fig = px.bar(
        by_vel, x="velocity_last_24h", y="fraud_rate", color="fraud_rate",
        color_continuous_scale=["#08D9D6", "#FFD93D", "#FF2E63"],
        text=by_vel["fraud_rate"].map("{:.1%}".format),
        title="⚡ Fraud rate by transactions in the last 24h",
        hover_data=["transactions", "frauds"],
    )
    fig.add_hline(y=BASE_RATE, line_dash="dash", line_color="white")
    fig.update_traces(textposition="outside")
    fig.update_layout(**CHART_LAYOUT, yaxis_tickformat=".1%", xaxis_title="Velocity (last 24h)",
                      yaxis_title="Fraud rate", coloraxis_showscale=False)
    f.plotly_chart(fig, use_container_width=True)

    plot_df = view.sort_values("is_fraud")  # draw frauds on top
    fig = px.scatter(
        plot_df, x="device_trust_score", y="amount", color="status",
        size="risk_score", size_max=16, opacity=0.75,
        color_discrete_map={"🚨 Fraud": FRAUD, "✅ Legit": SAFE},
        hover_data=["transaction_id", "merchant_category", "transaction_hour", "risk_reasons"],
        title="📱 Device trust vs amount — fraud clusters on the low-trust side",
    )
    fig.update_layout(**CHART_LAYOUT, height=520, xaxis_title="Device trust score",
                      yaxis_title="Amount ($)", legend_title="")
    st.plotly_chart(fig, use_container_width=True)


# --- 5.4 Risk Checker -------------------------------------------------------
with tab_checker:
    st.subheader("🧮 Score a single transaction")
    st.caption("Same scoring rules as the Fraud Finder — enter a transaction and see its verdict.")

    with st.form("checker"):
        g1, g2, g3 = st.columns(3)
        in_amount = g1.number_input("💵 Amount ($)", 0.0, 100_000.0, 850.0, step=10.0)
        in_hour = g2.slider("🕐 Transaction hour", 0, 23, 3)
        in_cat = g3.selectbox("🏪 Merchant category", categories)
        in_trust = g1.slider("📱 Device trust score", 0, 99, 18)
        in_vel = g2.slider("⚡ Transactions in last 24h", 0, 9, 5)
        in_age = g3.slider("🎂 Cardholder age", 18, 69, 34)
        in_foreign = g1.checkbox("🌍 Foreign transaction", value=True)
        in_mismatch = g2.checkbox("📍 Location mismatch", value=True)
        submitted = st.form_submit_button("🔍 Check this transaction", use_container_width=True)

    if submitted:
        one = pd.DataFrame([{
            "amount": in_amount, "transaction_hour": in_hour, "merchant_category": in_cat,
            "foreign_transaction": int(in_foreign), "location_mismatch": int(in_mismatch),
            "device_trust_score": in_trust, "velocity_last_24h": in_vel,
            "cardholder_age": in_age,
        }])
        score, reasons, band = score_transactions(one, P95, P99)
        score, reasons, band = float(score[0]), reasons[0], str(band[0])

        v1, v2 = st.columns([1, 1])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 44}},
            title={"text": f"Risk band: <b>{band}</b>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": BAND_COLORS[band]},
                "steps": [
                    {"range": [0, 25], "color": "rgba(56,229,77,.25)"},
                    {"range": [25, 50], "color": "rgba(255,217,61,.25)"},
                    {"range": [50, 75], "color": "rgba(255,159,28,.25)"},
                    {"range": [75, 100], "color": "rgba(255,46,99,.30)"},
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "value": score},
            },
        ))
        gauge.update_layout(**CHART_LAYOUT, height=330)
        v1.plotly_chart(gauge, use_container_width=True)

        with v2:
            if band == "Critical":
                st.error("🚨 **BLOCK / manual review** — this transaction looks like fraud.")
            elif band == "High":
                st.warning("⚠️ **Step-up authentication** — several strong red flags.")
            elif band == "Medium":
                st.info("🔎 **Monitor** — some risk signals present.")
            else:
                st.success("✅ **Approve** — no meaningful red flags.")

            st.markdown("**Triggered red flags**")
            if reasons.startswith("—"):
                st.markdown('<span class="badge" style="background:#38E54D">✅ none</span>',
                            unsafe_allow_html=True)
            else:
                badges = "".join(
                    f'<span class="badge" style="background:{BAND_COLORS[band]}">{r}</span>'
                    for r in reasons.split(" · ")
                )
                st.markdown(badges, unsafe_allow_html=True)

            similar = data[
                (data["foreign_transaction"] == int(in_foreign))
                & (data["location_mismatch"] == int(in_mismatch))
                & (data["merchant_category"] == in_cat)
            ]
            if len(similar):
                st.metric(
                    "📚 Historical fraud rate of similar transactions",
                    f"{similar['is_fraud'].mean():.2%}",
                    f"{similar['is_fraud'].mean() / BASE_RATE:.1f}× baseline"
                    if BASE_RATE else None,
                )
                st.caption(f"Based on {len(similar):,} past transactions with the same "
                           f"merchant / foreign / mismatch profile.")


# --- 5.5 Data Explorer ------------------------------------------------------
with tab_data:
    st.subheader("📋 Explore the filtered transactions")

    s1, s2 = st.columns([1, 3])
    tx_search = s1.text_input("🔍 Find transaction ID", placeholder="e.g. 4271")
    sort_by = s2.selectbox(
        "Sort by", ["risk_score", "amount", "transaction_id", "device_trust_score",
                    "velocity_last_24h", "transaction_hour"],
    )

    table = view.copy()
    if tx_search.strip().isdigit():
        table = table[table["transaction_id"] == int(tx_search.strip())]
        if table.empty:
            st.warning(f"No transaction with ID {tx_search} in the current filters.")
    table = table.sort_values(sort_by, ascending=False)

    st.caption(f"Showing **{len(table):,}** transactions · "
               f"**{int(table['is_fraud'].sum())}** confirmed frauds")
    show_table(table, height=560)

    st.download_button(
        "⬇️ Download filtered data (CSV)",
        table[TABLE_COLS].to_csv(index=False).encode("utf-8"),
        file_name="filtered_transactions.csv",
        mime="text/csv",
    )

    with st.expander("📈 Summary statistics"):
        stats = view[["amount", "transaction_hour", "device_trust_score",
                      "velocity_last_24h", "cardholder_age", "risk_score"]].describe().T
        st.markdown(
            '<div class="tbl-wrap">'
            + stats.round(2).to_html(classes="fx", border=0)
            + "</div>",
            unsafe_allow_html=True,
        )
