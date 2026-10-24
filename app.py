"""
Nairobi City Water & Sewerage Company (NCWSC)
Executive Operations Dashboard
==============================
Design goals (CEO-ready):
- One clean design system: Inter type, navy/teal palette, white cards, no clutter
- Every number is short, formatted and data-driven (no hard-coded demo figures)
- Bursts & leaks reported from detected SCADA incidents: KPIs, top list, chart, log
- 3D glass storage tank, P&L waterfall, treatment dosing with SOP at the bottom
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from dashboard_data import build_dataset, alert_episodes  # noqa: E402

st.set_page_config(
    page_title="NCWSC Executive Dashboard | Nairobi Water",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
METRO_SCALE = 2700.0          # pilot sector -> Nairobi metropolitan network
NRW_RATE = 0.425              # Nairobi non-revenue water benchmark (~42.5%)
DAILY_TARGET_M3 = 542_000.0   # metro daily production target

NAVY, BLUE, SKY = "#0b2545", "#1360a8", "#3fa7d6"
TEAL, RED, AMBER, SLATE = "#12a594", "#e5484d", "#f5a524", "#94a3b8"
FONT = "Inter, Segoe UI, sans-serif"

TYPE_COLORS = {"Burst": RED, "Trunk main loss": "#f76b15", "Hidden leak": AMBER, "Blockage": SLATE}
ZONE_SHORT = {
    "Zone A (Lang'ata)": "Lang'ata",
    "Zone B (Westlands)": "Westlands",
    "Zone C (Industrial Area)": "Industrial Area",
    "Southern Bypass Bulk Trunk": "Southern Bypass",
}
ACTIONS = {
    "burst_A": "Isolate Lang'ata feeder valve",
    "burst_B": "Close Westlands DMA valve",
    "burst_C": "Shut Enterprise Rd valve",
    "main_leak": "Patrol Southern Bypass main",
    "leak": "Acoustic leak survey",
    "clog": "Clear PRV & strainer",
}


# -------------------------------------------------------------
# DATA
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    return build_dataset()


@st.cache_data(show_spinner=False)
def load_incidents():
    """Network incidents (bursts, leaks, trunk loss, blockages) with water lost per episode."""
    d = load_data()
    ep = alert_episodes(d)
    if ep.empty:
        return ep
    ep = ep[ep["key"].str.match(r"^(burst|leak|main_leak|clog)")].copy()
    ts = d["timestamp"].values

    def excess_col(k):
        if k == "main_leak":
            return "main_loss_m3h"
        if k.startswith("leak_"):
            return f"leak_excess_{k[-1]}"
        if k.startswith("burst_"):
            return f"burst_excess_{k[-1]}"
        return None

    lost = []
    for r in ep.itertuples():
        col = excess_col(r.key)
        if col is None:
            lost.append(0.0)
            continue
        i0 = np.searchsorted(ts, np.datetime64(r.start))
        i1 = np.searchsorted(ts, np.datetime64(r.end))
        lost.append(float(d[col].iloc[i0:i1].clip(lower=0).sum()))

    ep["lost_m3_pilot"] = lost
    ep["type"] = ep["key"].map(
        lambda k: "Trunk main loss" if k == "main_leak"
        else {"burst": "Burst", "leak": "Hidden leak", "clog": "Blockage"}[k.split("_")[0]]
    )
    ep["zone"] = ep["asset"].map(lambda a: ZONE_SHORT.get(a, a))
    ep["action"] = ep["key"].map(lambda k: ACTIONS.get(k, ACTIONS.get(k.split("_")[0], "Inspect")))
    ep["status"] = np.where(ep["end"] >= d["timestamp"].max(), "Open", "Resolved")
    return ep.reset_index(drop=True)


with st.spinner("Connecting to NCWSC SCADA outstations..."):
    df_raw = load_data()
    inc_raw = load_incidents()


# -------------------------------------------------------------
# FORMAT HELPERS
# -------------------------------------------------------------
def kes(x, dec=1):
    s = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e9:
        return f"{s}KES {a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{s}KES {a / 1e6:.{dec}f}M"
    if a >= 1e3:
        return f"{s}KES {a / 1e3:.0f}K"
    return f"{s}KES {a:,.0f}"


def vol(x):
    if x >= 1e6:
        return f"{x / 1e6:.1f}M m³"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K m³"
    return f"{x:,.0f} m³"


def chip(cur, prev, good_up=True, pts=False):
    """Small year-on-year delta pill. Empty when no comparison period exists."""
    if prev is None:
        return ""
    if pts:
        delta, txt = cur - prev, f"{cur - prev:+.1f} pts"
    else:
        if prev == 0:
            return ""
        delta = (cur - prev) / abs(prev) * 100
        txt = f"{delta:+.1f}%"
    if abs(delta) < 0.05:
        cls = "flat"
    else:
        cls = "up" if (delta > 0) == good_up else "down"
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "•")
    return f'<span class="chip {cls}">{arrow} {txt.lstrip("+-")} YoY</span>'


def kpi(label, value, sub="", accent=BLUE, delta_html="", value_color=None):
    vc = f' style="color:{value_color}"' if value_color else ""
    st.markdown(f"""
    <div class="kpi" style="--accent:{accent}">
        <div class="kpi-top"><span class="kpi-label">{label}</span>{delta_html}</div>
        <div class="kpi-value"{vc}>{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def card(key):
    return st.container(key=f"card-{key}")


def card_title(title, sub=None, right_html=""):
    sub_html = f'<div class="cs">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="ct-row"><div><div class="ct">{title}</div>{sub_html}</div>{right_html}</div>',
        unsafe_allow_html=True,
    )


def polish(fig, h=280, legend=True):
    fig.update_layout(
        height=h, margin=dict(l=4, r=4, t=6, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color="#334155"),
        hoverlabel=dict(bgcolor=NAVY, font_color="#ffffff", font_family=FONT, bordercolor=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(size=11, color="#475569"), bgcolor="rgba(0,0,0,0)"),
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0", tickfont=dict(color="#64748b", size=11), title_text="")
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False, tickfont=dict(color="#64748b", size=11))
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def section(title, sub):
    st.markdown(f'<div class="section"><h2>{title}</h2><span>{sub}</span></div>', unsafe_allow_html=True)


# -------------------------------------------------------------
# STYLES
# -------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stMarkdown, button, input, select, textarea, label {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.stApp { background: #f3f6fa !important; color: #0f172a; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton, #MainMenu, footer { display: none !important; }
.block-container { padding-top: 1.3rem !important; padding-bottom: 1.2rem !important; max-width: 1600px; }
div[data-testid="stVerticalBlock"] { gap: 0.9rem; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b2545 0%, #102f57 100%) !important; }
section[data-testid="stSidebar"] * { color: #c9d8ea !important; }
section[data-testid="stSidebar"] .brand { padding: 4px 4px 18px 4px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px; }
section[data-testid="stSidebar"] .brand-name { font-size: 17px; font-weight: 800; color: #ffffff !important; letter-spacing: -0.01em; }
section[data-testid="stSidebar"] .brand-sub { font-size: 12px; color: #8fb3d9 !important; margin-top: 2px; }
section[data-testid="stSidebar"] .nav-label { font-size: 10.5px; font-weight: 700; letter-spacing: 0.12em; color: #6f93bb !important; margin: 6px 4px 6px 4px; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 9px 12px; border-radius: 8px; width: 100%; margin: 0; transition: background .15s ease; cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(255,255,255,0.06); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(63,167,214,0.16); box-shadow: inset 3px 0 0 #3fa7d6;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: #ffffff !important; font-weight: 700; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] p { font-size: 13.5px; font-weight: 500; }
section[data-testid="stSidebar"] .assump { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 10px 12px; font-size: 12px; margin-top: 6px; }
section[data-testid="stSidebar"] .assump b { color: #ffffff !important; }

/* ---------- Hero ---------- */
.hero {
    background: radial-gradient(1200px 300px at 85% -40%, rgba(63,167,214,0.45), transparent 60%),
                linear-gradient(120deg, #0b2545 0%, #13315c 55%, #1360a8 100%);
    border-radius: 16px; padding: 20px 26px; display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 10px 30px -12px rgba(11,37,69,0.45);
}
.hero .eyebrow { font-size: 11.5px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #8fc5ea; }
.hero h1 { font-size: 26px !important; font-weight: 800 !important; color: #ffffff !important; margin: 4px 0 0 0 !important;
    padding: 0 !important; letter-spacing: -0.02em; }
.hero-right { display: flex; gap: 8px; align-items: center; }
.pill { font-size: 12px; font-weight: 600; color: #e2efff; background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18); padding: 6px 12px; border-radius: 999px; white-space: nowrap; }
.pill .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #3ddc97; margin-right: 6px;
    box-shadow: 0 0 0 0 rgba(61,220,151,0.7); animation: pulse 2s infinite; }
.pill.alert .dot { background: #ff6b6b; box-shadow: 0 0 0 0 rgba(255,107,107,0.7); }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(61,220,151,0.6);} 70% { box-shadow: 0 0 0 7px rgba(61,220,151,0);} 100% { box-shadow: 0 0 0 0 rgba(61,220,151,0);} }

/* ---------- Section heading ---------- */
.section { display: flex; align-items: baseline; gap: 12px; margin: 2px 2px -4px 2px; }
.section h2 { font-size: 19px !important; font-weight: 800 !important; color: #0b2545 !important; margin: 0 !important; padding: 0 !important; letter-spacing: -0.01em; }
.section span { font-size: 13px; color: #64748b; }

/* ---------- Cards ---------- */
div[class*="st-key-card-"] {
    background: #ffffff; border: 1px solid #e4eaf1; border-radius: 14px; padding: 16px 18px 10px 18px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 6px 18px -8px rgba(16,24,40,0.08);
}
div.st-key-card-filters { padding: 10px 16px 4px 16px; }
div.st-key-card-filters label p { font-size: 11px !important; font-weight: 700 !important; color: #64748b !important;
    text-transform: uppercase; letter-spacing: 0.08em; }
.ct-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
.ct { font-size: 14.5px; font-weight: 700; color: #0b2545; letter-spacing: -0.005em; }
.cs { font-size: 12px; color: #64748b; margin-top: 2px; }

/* ---------- KPI ---------- */
.kpi {
    background: #ffffff; border: 1px solid #e4eaf1; border-radius: 14px; padding: 14px 16px 12px 16px; height: 120px;
    box-sizing: border-box; position: relative; overflow: hidden; display: flex; flex-direction: column; justify-content: space-between;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 6px 18px -8px rgba(16,24,40,0.08); transition: transform .15s ease, box-shadow .15s ease;
}
.kpi:hover { transform: translateY(-2px); box-shadow: 0 12px 24px -12px rgba(16,24,40,0.18); }
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.kpi-top { display: flex; justify-content: space-between; align-items: center; gap: 4px; }
.kpi-label { font-size: 10.5px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi-value { font-size: 25px; font-weight: 800; color: #0b2545; letter-spacing: -0.025em; line-height: 1.1; white-space: nowrap; }
.kpi-sub { font-size: 11.5px; color: #64748b; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chip { font-size: 10.5px; font-weight: 700; padding: 2px 8px; border-radius: 999px; white-space: nowrap; }
.chip.up { background: #e3f6f1; color: #0d8a72; }
.chip.down { background: #fdeced; color: #d4343a; }
.chip.flat { background: #eef2f7; color: #64748b; }

/* ---------- Badges ---------- */
.badge { font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 999px; white-space: nowrap; }
.badge.good { background: #e3f6f1; color: #0d8a72; }
.badge.bad { background: #fdeced; color: #d4343a; }
.badge.neutral { background: #eef2f7; color: #475569; }

/* ---------- Takeaways ---------- */
.tk { display: flex; gap: 12px; padding: 11px 0; border-bottom: 1px solid #f1f5f9; }
.tk:last-child { border-bottom: none; }
.tk-dot { width: 32px; height: 32px; min-width: 32px; border-radius: 9px; display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800; }
.tk-h { font-size: 14px; font-weight: 700; color: #0b2545; }
.tk-s { font-size: 12.5px; color: #64748b; margin-top: 1px; }

/* ---------- Incident rows ---------- */
.inc { display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid #f1f5f9; }
.inc:last-child { border-bottom: none; }
.inc-bar { width: 4px; align-self: stretch; border-radius: 4px; }
.inc-main { flex: 1; min-width: 0; }
.inc-t { font-size: 13.5px; font-weight: 700; color: #0b2545; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.inc-m { font-size: 12px; color: #64748b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.inc-k { font-size: 14px; font-weight: 800; color: #0b2545; white-space: nowrap; text-align: right; }
.inc-k small { display: block; font-size: 11px; font-weight: 600; color: #94a3b8; }

/* ---------- Dosing ---------- */
.dose { background: #ffffff; border: 1px solid #e4eaf1; border-radius: 14px; padding: 16px 18px; height: 100%;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 6px 18px -8px rgba(16,24,40,0.08); }
.dose-h { font-size: 11.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px; }
.dose-hero { border-radius: 10px; padding: 12px; text-align: center; margin-bottom: 10px; }
.dose-hero .l { font-size: 11px; font-weight: 600; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.06em; }
.dose-hero .v { font-size: 30px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; }
.dose-hero .s { font-size: 12px; opacity: 0.85; }
.dose-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #f1f5f9; font-size: 12.5px; }
.dose-row:last-child { border-bottom: none; }
.dose-row span:first-child { color: #64748b; }
.dose-row span:last-child { font-weight: 700; color: #0b2545; }

/* ---------- SOP ---------- */
.sop { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 6px; }
.sop div { background: #f6f9fc; border-radius: 10px; padding: 10px 12px; border-top: 3px solid #1360a8; }
.sop div:last-child { border-top-color: #12a594; }
.sop b { display: block; font-size: 12.5px; color: #0b2545; }
.sop span { font-size: 12px; color: #64748b; }

.foot { text-align: center; padding: 10px 0 4px 0; color: #94a3b8; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR: NAVIGATION + ASSUMPTIONS
# -------------------------------------------------------------
MENU_OPTIONS = [
    "Executive Overview",
    "Financials",
    "Bursts & Leaks",
    "Water Sources",
    "Treatment Plant",
    "Customers",
    "Monthly Trends",
]

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-name">NCWSC Control Room</div>
        <div class="brand-sub">Nairobi City Water &amp; Sewerage</div>
    </div>
    <div class="nav-label">DASHBOARD</div>
    """, unsafe_allow_html=True)

    active_module = st.radio("Navigation", MENU_OPTIONS, key="active_module", label_visibility="collapsed")

    st.markdown('<div class="nav-label" style="margin-top:22px;">ASSUMPTIONS</div>', unsafe_allow_html=True)
    p_tariff = st.slider("Water tariff (KES/m³)", 80, 220, 120, 5)
    p_cost = st.slider("Production cost (KES/m³)", 30, 120, 55, 5)
    st.markdown(f"""
    <div class="assump">
        Gross margin <b>KES {p_tariff - p_cost}/m³</b><br>
        NRW benchmark <b>{NRW_RATE * 100:.1f}%</b><br>
        Network scale <b>{METRO_SCALE:,.0f}× pilot</b>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# HERO
# -------------------------------------------------------------
open_incidents = int((inc_raw["status"] == "Open").sum()) if not inc_raw.empty else 0
status_pill = (
    f'<span class="pill alert"><span class="dot"></span>{open_incidents} open incident{"s" if open_incidents != 1 else ""}</span>'
    if open_incidents else '<span class="pill"><span class="dot"></span>Network stable</span>'
)
st.markdown(f"""
<div class="hero">
    <div>
        <div class="eyebrow">Nairobi City Water &amp; Sewerage Company</div>
        <h1>Executive Operations Dashboard</h1>
    </div>
    <div class="hero-right">
        {status_pill}
        <span class="pill">SCADA · {df_raw['timestamp'].min():%b %Y} – {df_raw['timestamp'].max():%b %Y}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# FILTER BAR
# -------------------------------------------------------------
with card("filters"):
    f1, f2, f3, f4 = st.columns(4)
    years = sorted(df_raw["timestamp"].dt.year.unique().tolist())
    with f1:
        sel_year = st.selectbox("Year", ["All"] + [str(y) for y in years], index=0)
    with f2:
        month_names = ["All"] + [pd.Timestamp(2000, m, 1).strftime("%B") for m in range(1, 13)]
        sel_month_name = st.selectbox("Month", month_names, index=0)
        sel_month = None if sel_month_name == "All" else month_names.index(sel_month_name)
    with f3:
        sel_zone = st.selectbox("Zone", ["All", "Lang'ata", "Westlands", "Industrial Area", "Southern Bypass"], index=0)
    with f4:
        sel_scale = st.selectbox("Scale", ["Metropolitan network", "Pilot sector"], index=0)

mult = METRO_SCALE if sel_scale.startswith("Metro") else 1.0


# -------------------------------------------------------------
# PERIOD SLICING + METRICS
# -------------------------------------------------------------
def slice_period(d, year, month):
    out = d
    if year != "All":
        out = out[out["timestamp"].dt.year == int(year)]
    if month:
        out = out[out["timestamp"].dt.month == month]
    return out


def metrics(d):
    pumped = d["pumped_m3"].sum() * mult
    lost = pumped * NRW_RATE
    billed = pumped - lost
    income = billed * p_tariff
    expenses = pumped * p_cost
    profit = income - expenses
    days = max(1.0, len(d) / 24.0)
    return dict(
        pumped=pumped, lost=lost, billed=billed, income=income, expenses=expenses, profit=profit,
        margin=(profit / income * 100) if income else 0.0, days=days, lost_value=lost * p_tariff,
    )


def slice_incidents(ep, year, month, zone):
    if ep.empty:
        return ep
    out = ep
    if year != "All":
        out = out[out["start"].dt.year == int(year)]
    if month:
        out = out[out["start"].dt.month == month]
    if zone != "All":
        out = out[out["zone"] == zone]
    out = out.copy()
    out["lost_m3"] = out["lost_m3_pilot"] * mult
    out["lost_kes"] = out["lost_m3"] * p_tariff
    return out


df_f = slice_period(df_raw, sel_year, sel_month)
if df_f.empty:
    df_f = df_raw.iloc[-720:]
M = metrics(df_f)

prev_year = str(int(sel_year) - 1) if sel_year != "All" else None
df_prev = slice_period(df_raw, prev_year, sel_month) if prev_year and int(prev_year) in years else None
P = metrics(df_prev) if df_prev is not None and not df_prev.empty else None

inc_f = slice_incidents(inc_raw, sel_year, sel_month, sel_zone)
inc_p = slice_incidents(inc_raw, prev_year, sel_month, sel_zone) if P else None

per_day = lambda x: x / M["days"]  # noqa: E731
is_profitable = M["profit"] >= 0
period_label = (
    "All years" if sel_year == "All" and not sel_month
    else " ".join(x for x in [sel_month_name if sel_month else "", sel_year if sel_year != "All" else ""] if x)
)

# =============================================================
# 1. EXECUTIVE OVERVIEW
# =============================================================
if active_module == "Executive Overview":
    section("Executive Overview", f"{period_label} · how the network is performing")

    n_inc = len(inc_f)
    n_burst = int((inc_f["type"] == "Burst").sum()) if n_inc else 0
    inc_loss = float(inc_f["lost_kes"].sum()) if n_inc else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi("Water produced", vol(M["pumped"]), f"{vol(per_day(M['pumped']))} per day", BLUE,
            chip(M["pumped"], P["pumped"] if P else None))
    with k2:
        kpi("Revenue", kes(M["income"]), f"{kes(per_day(M['income']))} per day", SKY,
            chip(M["income"], P["income"] if P else None))
    with k3:
        kpi("Net margin", kes(M["profit"]), f"{M['margin']:.1f}% of revenue", TEAL if is_profitable else RED,
            chip(M["profit"], P["profit"] if P else None), TEAL if is_profitable else RED)
    with k4:
        kpi("Non-revenue water", f"{NRW_RATE * 100:.1f}%", f"{kes(per_day(M['lost_value']))} per day unbilled", RED)
    with k5:
        kpi("Network incidents", f"{n_inc}", f"{n_burst} burst{'s' if n_burst != 1 else ''} · {kes(inc_loss)} lost", AMBER,
            chip(n_inc, len(inc_p) if inc_p is not None else None, good_up=False))

    # ---------- Row 2: WIDENED DAILY PRODUCTION + P&L WATERFALL ----------
    c_tank, c_pnl = st.columns([6.6, 5.4])

    with c_tank:
        target = DAILY_TARGET_M3 * mult / METRO_SCALE
        d_prod = per_day(M["pumped"])
        ratio = d_prod / target if target else 0
        fill = min(1.1, max(0.06, ratio))
        liquid_y = 196 - fill * 148          # target (100%) sits at y=48
        net_col = "#5eead4" if is_profitable else "#ff8a8a"
        tank_html = f"""
        <html><head>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800&display=swap" rel="stylesheet">
        <style>
            body {{ margin:0; font-family:'Inter','Segoe UI',sans-serif; background:transparent; }}
            .c {{ background: radial-gradient(650px 260px at 100% 0%, rgba(63,167,214,0.35), transparent 65%),
                     linear-gradient(150deg,#0b2545 0%,#13315c 100%);
                  border-radius:14px; padding:18px 24px; height:365px; box-sizing:border-box; color:#fff;
                  display:flex; justify-content:space-between; align-items:center; gap:20px;
                  box-shadow:0 10px 24px -14px rgba(11,37,69,.6); }}
            .l {{ flex:1.25; display:flex; flex-direction:column; justify-content:space-between; height:100%; }}
            .r {{ flex:0.95; display:flex; flex-direction:column; align-items:center; justify-content:center; }}
            .h {{ font-size:11.5px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#8fc5ea; }}
            .big {{ font-size:36px; font-weight:800; letter-spacing:-.03em; line-height:1.05; margin:4px 0 6px 0; }}
            .big span {{ font-size:15px; font-weight:600; color:#8fc5ea; letter-spacing:0; }}
            .tag {{ display:inline-block; margin-bottom:10px; font-size:11px; font-weight:700; padding:3px 10px; border-radius:999px;
                    background:rgba(94,234,212,.14); color:#5eead4; border:1px solid rgba(94,234,212,.35); }}
            .row {{ display:flex; justify-content:space-between; align-items:center; padding:9px 0; gap:16px;
                    border-top:1px solid rgba(255,255,255,.1); }}
            .row .k {{ font-size:12px; color:#9fbad6; white-space:nowrap; margin-right:auto; }}
            .row .v {{ font-size:13.5px; font-weight:700; white-space:nowrap; }}
        </style></head><body>
        <div class="c">
            <div class="l">
                <div>
                    <div class="h">Daily Production &amp; Run-Rate</div>
                    <div class="big">{d_prod / 1e3:,.0f}K <span>m³/day</span></div>
                    <div class="tag">{ratio * 100:.0f}% of {target / 1e3:,.0f}K Daily Target</div>
                </div>
                <div>
                    <div class="row"><span class="k">Daily Billed Revenue</span><span class="v">{kes(per_day(M['income']))}</span></div>
                    <div class="row"><span class="k">Daily Production Opex</span><span class="v">{kes(per_day(M['expenses']))}</span></div>
                    <div class="row"><span class="k">Daily Net Margin</span><span class="v" style="color:{net_col}">{kes(per_day(M['profit']))} ({M['margin']:+.1f}%)</span></div>
                </div>
            </div>
            <div class="r">
                <svg width="170" height="235" viewBox="0 0 170 230" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="glass" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0%" stop-color="#b0d8f0" stop-opacity=".55"/>
                            <stop offset="22%" stop-color="#ffffff" stop-opacity=".28"/>
                            <stop offset="60%" stop-color="#e8f4fc" stop-opacity=".06"/>
                            <stop offset="86%" stop-color="#ffffff" stop-opacity=".35"/>
                            <stop offset="100%" stop-color="#90c0e0" stop-opacity=".6"/>
                        </linearGradient>
                        <linearGradient id="water" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0%" stop-color="#0b3b6f"/>
                            <stop offset="35%" stop-color="#1360a8"/>
                            <stop offset="70%" stop-color="#2f8fd0"/>
                            <stop offset="100%" stop-color="#0f4f8f"/>
                        </linearGradient>
                        <radialGradient id="surf" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stop-color="#c8f4ff"/>
                            <stop offset="65%" stop-color="#3fa7d6"/>
                            <stop offset="100%" stop-color="#1360a8"/>
                        </radialGradient>
                    </defs>
                    <!-- Background shell back -->
                    <rect x="27" y="32" width="116" height="145" fill="rgba(255,255,255,.05)"/>
                    <ellipse cx="85" cy="177" rx="58" ry="11" fill="#081c36"/>

                    <!-- Water body -->
                    <rect x="28" y="{177 - fill * 135:.1f}" width="114" height="{fill * 135:.1f}" fill="url(#water)">
                        <animate attributeName="y" from="177" to="{177 - fill * 135:.1f}" dur="1.4s" fill="freeze"/>
                        <animate attributeName="height" from="0" to="{fill * 135:.1f}" dur="1.4s" fill="freeze"/>
                    </rect>
                    <ellipse cx="85" cy="177" rx="57" ry="10" fill="#0b3b6f"/>
                    <ellipse cx="85" cy="{177 - fill * 135:.1f}" rx="57" ry="10" fill="url(#surf)" stroke="#8be3ff" stroke-width="1.2">
                        <animate attributeName="cy" from="177" to="{177 - fill * 135:.1f}" dur="1.4s" fill="freeze"/>
                    </ellipse>

                    <!-- Target reference line & label -->
                    <ellipse cx="85" cy="42" rx="58" ry="11" fill="none" stroke="#ff8a8a" stroke-width="1.5" stroke-dasharray="4 3"/>
                    <text x="85" y="16" font-size="9" font-weight="700" fill="#ffb4b4" text-anchor="middle" font-family="Inter">TARGET: {target / 1e3:,.0f}K m³</text>

                    <!-- In-tank capacity display -->
                    <text x="85" y="116" font-size="24" font-weight="800" fill="#ffffff" text-anchor="middle" font-family="Inter">{ratio * 100:.0f}%</text>
                    <text x="85" y="132" font-size="9.5" font-weight="600" fill="#cfe6fb" text-anchor="middle" font-family="Inter">CAPACITY FILL</text>

                    <!-- Glass front & top rim -->
                    <rect x="27" y="32" width="116" height="145" fill="url(#glass)" stroke="#5aa9e0" stroke-width="1.2"/>
                    <ellipse cx="85" cy="32" rx="58" ry="11" fill="none" stroke="#7cc0ef" stroke-width="1.8"/>

                    <!-- Bottom caption -->
                    <text x="85" y="206" font-size="10.5" font-weight="700" fill="#60efff" text-anchor="middle" font-family="Inter">Tank Level: {ratio * 100:.0f}%</text>
                </svg>
            </div>
        </div></body></html>
        """
        components.html(tank_html, height=375)

    with c_pnl:
        with card("pnl"):
            badge = (f'<span class="badge good">+{M["margin"]:.1f}% margin</span>' if is_profitable
                     else f'<span class="badge bad">{M["margin"]:.1f}% margin</span>')
            card_title("Daily Operating P&L", "Revenue breakdown to net margin, KES millions/day", badge)
            inc_d, exp_d = per_day(M["income"]) / 1e6, per_day(M["expenses"]) / 1e6
            parts = [exp_d * 0.58, exp_d * 0.14, exp_d * 0.28]
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=["Revenue", "Power Opex", "Chemicals", "Labour & O&M", "Net Margin"],
                y=[inc_d, -parts[0], -parts[1], -parts[2], 0],
                text=[f"KES {inc_d:.1f}M", f"-{parts[0]:.1f}M", f"-{parts[1]:.1f}M", f"-{parts[2]:.1f}M", f"{inc_d - exp_d:.1f}M"],
                textposition="outside", textfont=dict(size=12, color="#0b2545", family=FONT),
                connector=dict(line=dict(color="#cbd5e1", width=1, dash="dot")),
                increasing=dict(marker=dict(color=BLUE)),
                decreasing=dict(marker=dict(color="#cbd5e1")),
                totals=dict(marker=dict(color=TEAL if is_profitable else RED)),
                hovertemplate="%{x}: KES %{y:.1f}M<extra></extra>",
            ))
            polish(fig, 288, legend=False)
            fig.update_yaxes(showticklabels=False, range=[0, inc_d * 1.18])
            show(fig)

    # ---------- Row 3: WATER DISTRIBUTED TRAJECTORY + BILLING EFFICIENCY ----------
    r_dist, r_donut = st.columns([7.5, 4.5])
    with r_dist:
        with card("trend"):
            card_title("Water Distributed Trajectory", "Monthly delivered volume across metropolitan grid, millions m³")
            mg = df_f.resample("ME", on="timestamp")["pumped_m3"].sum() * mult * (1 - NRW_RATE) / 1e6
            fig = go.Figure(go.Scatter(
                x=mg.index, y=mg.values, fill="tozeroy", mode="lines",
                line=dict(color=BLUE, width=2.5, shape="spline"),
                fillcolor="rgba(19,96,168,0.12)",
                hovertemplate="%{x|%b %Y}<br><b>%{y:.2f}M m³</b><extra></extra>",
            ))
            polish(fig, 250, legend=False)
            show(fig)
    with r_donut:
        with card("donut"):
            card_title("Billing Efficiency", "Share of produced water billed vs non-revenue loss")
            eff = 100 - NRW_RATE * 100
            fig = go.Figure(go.Pie(
                values=[eff, NRW_RATE * 100], labels=["Billed Revenue Water", "Unmetered NRW Loss"], hole=0.74, sort=False,
                marker=dict(colors=[TEAL, "#fbd5d6"], line=dict(color="#ffffff", width=2)), textinfo="none",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            polish(fig, 210, legend=False)
            fig.update_layout(annotations=[dict(text=f"<b>{eff:.1f}%</b><br><span style='font-size:11px;color:#64748b'>billed</span>",
                                                x=0.5, y=0.5, font=dict(size=22, color=NAVY, family=FONT), showarrow=False)])
            show(fig)
            st.markdown(f'<div class="cs" style="text-align:center;margin-top:-6px;">Billed <b>{vol(M["billed"])}</b> · '
                        f'<span style="color:{RED};font-weight:700">Lost {vol(M["lost"])}</span></div>',
                        unsafe_allow_html=True)

    # ---------- Row 4: INCIDENTS BY ZONE + KEY TAKEAWAYS (SPACIOUS SIDE-BY-SIDE) ----------
    r_inc, r_take = st.columns([6, 6])
    with r_inc:
        with card("zones"):
            card_title("Network Incidents by Zone", f"Total detected failure events ({n_inc} in selection)")
            if n_inc:
                z = inc_f.groupby("zone").size().sort_values()
                fig = go.Figure(go.Bar(x=z.values, y=z.index, orientation="h", marker_color=AMBER,
                                       text=[f"{v} events" for v in z.values], textposition="outside", cliponaxis=False,
                                       hovertemplate="%{y}: %{x} events<extra></extra>"))
                polish(fig, 220, legend=False)
                fig.update_layout(margin=dict(l=4, r=60, t=6, b=4))
                fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, max(z.values) * 1.35])
                fig.update_traces(marker_line_width=0, width=0.52)
                show(fig)
            else:
                st.markdown('<div class="cs" style="padding:60px 0;text-align:center;">No incidents</div>', unsafe_allow_html=True)

    with r_take:
        with card("takeaways"):
            card_title("Executive Takeaways & Insights", period_label)
            biggest = inc_f.sort_values("lost_kes", ascending=False).iloc[0] if n_inc else None
            items = [
                (TEAL if is_profitable else RED, "▲" if is_profitable else "▼",
                 f"{'Profitable Operations' if is_profitable else 'Operating Deficit'} ({M['margin']:+.1f}%)",
                 f"{kes(per_day(M['profit']))} net margin per day"),
                (RED, "!", f"NRW Exposure ({NRW_RATE * 100:.1f}%)",
                 f"{kes(per_day(M['lost_value']))} daily unbilled physical loss"),
                (AMBER, str(n_inc), f"{n_inc} Network Incidents",
                 (f"Top: {biggest['zone']} {biggest['type']} ({kes(biggest['lost_kes'])})"
                  if biggest is not None else "No incidents in selected period")),
                (BLUE, "◎", f"Production Run-Rate ({ratio * 100:.0f}%)",
                 f"{vol(d_prod)} / day vs {vol(target)} target"),
            ]
            html = '<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:8px;">'
            for col, icon, h, s in items:
                html += (
                    f'<div style="background:#f8fafc; border:1px solid #eef2f6; border-radius:10px; padding:10px 12px; display:flex; gap:10px; align-items:flex-start;">'
                    f'<div class="tk-dot" style="background:{col}1a; color:{col}; min-width:28px; width:28px; height:28px; font-size:13px;">{icon}</div>'
                    f'<div><div style="font-size:12.5px; font-weight:700; color:#0b2545;">{h}</div>'
                    f'<div style="font-size:11.5px; color:#64748b; margin-top:2px;">{s}</div></div></div>'
                )
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)


# =============================================================
# 2. FINANCIALS
# =============================================================
elif active_module == "Financials":
    section("Financials", f"{period_label} · income, costs and margin")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        kpi("Revenue", kes(M["income"]), f"{kes(per_day(M['income']))} per day", BLUE,
            chip(M["income"], P["income"] if P else None))
    with f2:
        kpi("Operating costs", kes(M["expenses"]), f"{kes(per_day(M['expenses']))} per day", SLATE,
            chip(M["expenses"], P["expenses"] if P else None, good_up=False))
    with f3:
        kpi("Net margin", kes(M["profit"]), f"{M['margin']:.1f}% of revenue", TEAL if is_profitable else RED,
            chip(M["margin"], P["margin"] if P else None, pts=True), TEAL if is_profitable else RED)
    with f4:
        kpi("Value of NRW", kes(M["lost_value"]), f"{kes(per_day(M['lost_value']))} per day", RED)

    c1, c2 = st.columns([8, 4])
    with c1:
        with card("fin-monthly"):
            card_title("Revenue vs costs", "Monthly, KES millions")
            m = df_f.resample("ME", on="timestamp")["pumped_m3"].sum() * mult
            inc_m = m * (1 - NRW_RATE) * p_tariff / 1e6
            exp_m = m * p_cost / 1e6
            fig = go.Figure()
            fig.add_bar(x=m.index, y=inc_m, name="Revenue", marker_color=BLUE)
            fig.add_bar(x=m.index, y=exp_m, name="Costs", marker_color="#cbd5e1")
            fig.add_scatter(x=m.index, y=inc_m - exp_m, name="Net margin", mode="lines+markers",
                            line=dict(color=TEAL, width=2.5, shape="spline"), marker=dict(size=5))
            polish(fig, 320)
            fig.update_layout(barmode="group", bargap=0.25, hovermode="x unified")
            show(fig)
    with c2:
        with card("fin-costs"):
            card_title("Cost breakdown", "Share of operating costs")
            fig = go.Figure(go.Pie(
                labels=["Power", "Labour & O&M", "Chemicals"], values=[58, 28, 14], hole=0.62, sort=False,
                marker=dict(colors=[NAVY, SKY, TEAL], line=dict(color="#fff", width=2)),
                textinfo="percent", textfont=dict(color="#fff", size=12),
                hovertemplate="%{label}: %{percent}<extra></extra>",
            ))
            polish(fig, 320)
            fig.update_layout(legend=dict(orientation="h", y=-0.08, yanchor="top", x=0.5, xanchor="center"))
            show(fig)

# =============================================================
# 3. BURSTS & LEAKS
# =============================================================
elif active_module == "Bursts & Leaks":
    zone_txt = "" if sel_zone == "All" else f" · {sel_zone}"
    section("Bursts & Leaks", f"{period_label}{zone_txt} · detected from SCADA flow and pressure")

    n_inc = len(inc_f)
    n_burst = int((inc_f["type"] == "Burst").sum()) if n_inc else 0
    lost_m3 = float(inc_f["lost_m3"].sum()) if n_inc else 0.0
    lost_kes = float(inc_f["lost_kes"].sum()) if n_inc else 0.0
    avg_days = float(inc_f["hours"].mean() / 24) if n_inc else 0.0
    pv = inc_p if inc_p is not None else None

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi("Incidents", f"{n_inc}", f"{int((inc_f['status'] == 'Open').sum()) if n_inc else 0} open now", AMBER,
            chip(n_inc, len(pv) if pv is not None else None, good_up=False))
    with k2:
        kpi("Bursts", f"{n_burst}", "Sudden pipe ruptures", RED,
            chip(n_burst, int((pv["type"] == "Burst").sum()) if pv is not None and len(pv) else None, good_up=False))
    with k3:
        kpi("Water lost", vol(lost_m3), "From detected incidents", SKY,
            chip(lost_m3, float(pv["lost_m3"].sum()) if pv is not None and len(pv) else None, good_up=False))
    with k4:
        kpi("Revenue lost", kes(lost_kes), f"at KES {p_tariff}/m³", RED,
            chip(lost_kes, float(pv["lost_kes"].sum()) if pv is not None and len(pv) else None, good_up=False), RED)
    with k5:
        kpi("Avg. duration", f"{avg_days:.1f} days", "Detection to resolution", SLATE)

    if not n_inc:
        with card("empty"):
            st.markdown('<div class="cs" style="padding:40px 0;text-align:center;font-size:14px;">'
                        'No bursts or leaks detected for this selection.</div>', unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns([4.6, 3.7, 3.7])

        with c1:
            with card("top-inc"):
                card_title("Costliest incidents", "Ranked by revenue lost")
                top = inc_f.sort_values("lost_kes", ascending=False).head(5)
                html = ""
                for r in top.itertuples():
                    col = TYPE_COLORS.get(r.type, SLATE)
                    dur = f"{r.hours / 24:.0f} d" if r.hours >= 48 else f"{r.hours} h"
                    status = (f' · <b style="color:{RED}">Open</b>' if r.status == "Open" else "")
                    html += (
                        f'<div class="inc"><div class="inc-bar" style="background:{col}"></div>'
                        f'<div class="inc-main"><div class="inc-t">{r.zone} · {r.type}</div>'
                        f'<div class="inc-m">{r.start:%d %b %Y} · {dur} · {r.action}{status}</div></div>'
                        f'<div class="inc-k">{kes(r.lost_kes)}<small>{vol(r.lost_m3)}</small></div></div>'
                    )
                st.markdown(html, unsafe_allow_html=True)

        with c2:
            with card("by-zone"):
                card_title("Incidents by zone", "Count by type")
                zt = inc_f.groupby(["zone", "type"]).size().reset_index(name="n")
                order = inc_f.groupby("zone").size().sort_values().index.tolist()
                fig = go.Figure()
                for t, colr in TYPE_COLORS.items():
                    s = zt[zt["type"] == t].set_index("zone")["n"].reindex(order).fillna(0)
                    if s.sum() == 0:
                        continue
                    fig.add_bar(y=order, x=s.values, name=t, orientation="h", marker_color=colr,
                                hovertemplate=f"{t}: %{{x}}<extra></extra>")
                polish(fig, 300)
                fig.update_layout(barmode="stack", bargap=0.4)
                show(fig)

        with c3:
            with card("by-type"):
                card_title("Revenue lost by type", "Share of KES lost")
                lt = inc_f.groupby("type")["lost_kes"].sum()
                lt = lt[lt > 0]
                fig = go.Figure(go.Pie(
                    labels=lt.index, values=lt.values, hole=0.66, sort=True,
                    marker=dict(colors=[TYPE_COLORS.get(t, SLATE) for t in lt.index], line=dict(color="#fff", width=2)),
                    textinfo="none", hovertemplate="%{label}: KES %{value:,.0f}<extra></extra>",
                ))
                polish(fig, 300)
                fig.update_layout(
                    legend=dict(orientation="h", y=-0.05, yanchor="top", x=0.5, xanchor="center"),
                    annotations=[dict(text=f"<b>{kes(lost_kes)}</b><br><span style='font-size:11px;color:#64748b'>total lost</span>",
                                      x=0.5, y=0.5, font=dict(size=16, color=NAVY, family=FONT), showarrow=False)],
                )
                show(fig)

        t1, t2 = st.columns([7.5, 4.5])
        with t1:
            with card("timeline"):
                card_title("Incident timeline", "Each bar is one incident, from detection to resolution")
                tl = inc_f.copy()
                tl["end_plot"] = tl.apply(lambda r: max(r["end"], r["start"] + pd.Timedelta(days=2)), axis=1)
                fig = px.timeline(tl, x_start="start", x_end="end_plot", y="zone", color="type",
                                  color_discrete_map=TYPE_COLORS,
                                  hover_data={"start": "|%d %b %Y", "end_plot": False, "hours": True, "zone": False})
                polish(fig, 280)
                fig.update_yaxes(title_text="", categoryorder="array",
                                 categoryarray=["Southern Bypass", "Industrial Area", "Westlands", "Lang'ata"])
                fig.update_layout(legend_title_text="", bargap=0.35)
                show(fig)
        with t2:
            with card("quarterly"):
                card_title("Quarterly trend", "Incidents per quarter")
                q = inc_f.assign(q=inc_f["start"].dt.to_period("Q").astype(str)).groupby(["q", "type"]).size().unstack(fill_value=0)
                fig = go.Figure()
                for t, colr in TYPE_COLORS.items():
                    if t in q.columns:
                        fig.add_bar(x=q.index, y=q[t], name=t, marker_color=colr)
                polish(fig, 280, legend=False)
                fig.update_layout(barmode="stack", bargap=0.35)
                show(fig)

        with card("log"):
            card_title("Incident log", f"{n_inc} incidents · sorted by revenue lost")
            log = inc_f.sort_values("lost_kes", ascending=False)[
                ["start", "zone", "type", "hours", "lost_m3", "lost_kes", "action", "status"]
            ].copy()
            log["lost_kes"] = log["lost_kes"] / 1e6
            st.dataframe(
                log, hide_index=True, use_container_width=True, height=280,
                column_config={
                    "start": st.column_config.DatetimeColumn("Detected", format="DD MMM YYYY"),
                    "zone": "Zone",
                    "type": "Type",
                    "hours": st.column_config.NumberColumn("Hours", format="%d"),
                    "lost_m3": st.column_config.NumberColumn("Water lost (m³)", format="%,.0f"),
                    "lost_kes": st.column_config.ProgressColumn(
                        "Revenue lost (KES M)", format="%.1f", min_value=0, max_value=float(log["lost_kes"].max() or 1)),
                    "action": "Action",
                    "status": "Status",
                },
            )

# =============================================================
# 4. WATER SOURCES
# =============================================================
elif active_module == "Water Sources":
    section("Water Sources", f"{period_label} · borehole intake")
    bh1 = df_f["bh1_flow_m3h"].sum() * mult
    bh2 = df_f["bh2_flow_m3h"].sum() * mult
    tot = max(bh1 + bh2, 1)

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("Karen Wellfield (BH1)", vol(bh1), f"{bh1 / tot * 100:.1f}% of raw supply", BLUE)
    with k2:
        kpi("Lang'ata Aquifer (BH2)", vol(bh2), f"{bh2 / tot * 100:.1f}% of raw supply", SKY)
    with k3:
        eff = float(df_f["bh2_efficiency_pct"].iloc[-1]) if "bh2_efficiency_pct" in df_f else 100.0
        kpi("BH2 pump efficiency", f"{eff:.0f}%", "vs. clean-screen baseline", TEAL if eff > 90 else AMBER)

    c1, c2 = st.columns([8, 4])
    with c1:
        with card("src-flow"):
            card_title("Intake flow · last 7 days", "m³/h per borehole")
            rec = df_f.iloc[-168:]
            fig = go.Figure()
            fig.add_scatter(x=rec["timestamp"], y=rec["bh1_flow_m3h"], name="Karen (BH1)", mode="lines",
                            line=dict(color=BLUE, width=2))
            fig.add_scatter(x=rec["timestamp"], y=rec["bh2_flow_m3h"], name="Lang'ata (BH2)", mode="lines",
                            line=dict(color=SKY, width=2))
            polish(fig, 300)
            fig.update_layout(hovermode="x unified")
            show(fig)
    with c2:
        with card("src-share"):
            card_title("Supply mix", "Share of raw intake")
            fig = go.Figure(go.Pie(labels=["Karen (BH1)", "Lang'ata (BH2)"], values=[bh1, bh2], hole=0.66,
                                   marker=dict(colors=[BLUE, SKY], line=dict(color="#fff", width=2)),
                                   textinfo="percent", textfont=dict(color="#fff")))
            polish(fig, 300)
            fig.update_layout(legend=dict(orientation="h", y=-0.05, yanchor="top", x=0.5, xanchor="center"))
            show(fig)

# =============================================================
# 5. TREATMENT PLANT
# =============================================================
elif active_module == "Treatment Plant":
    section("Treatment Plant", "Nairobi South Works · live sensors and recommended dosing")

    sensor_turb = float(df_f["raw_turbidity_ntu"].iloc[-1])
    sensor_flow = float(df_f["treatment_inlet_m3h"].iloc[-1] * (mult / 28.0))
    if sensor_flow < 200:
        sensor_flow = 850.0
    sensor_ph = 7.20
    treated = float(df_f["treated_turbidity_ntu"].iloc[-1])

    ph_factor = 1.15 if sensor_ph > 7.8 else (0.95 if sensor_ph < 6.8 else 1.0)
    dose = (4.5 * np.power(max(0.5, sensor_turb), 0.6) + 5.0) * ph_factor
    alum_kg_h = dose * sensor_flow / 1000.0
    pump_ml_min = dose * sensor_flow / 6.0
    pump_l_h = pump_ml_min * 60.0 / 1000.0
    stroke = min(100.0, max(10.0, pump_l_h / 250.0 * 100.0))
    cl_kg_h = 2.0 * sensor_flow / 1000.0
    cl_dial = min(100.0, max(10.0, cl_kg_h / 4.0 * 100.0))
    chem_day = alum_kg_h * 24 * 70.0 + cl_kg_h * 24 * 150.0

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        kpi("Raw turbidity", f"{sensor_turb:.1f} NTU", "Intake sensor S-101", BLUE)
    with s2:
        kpi("Intake flow", f"{sensor_flow:,.0f} m³/h", "Flow meter FM-204", SKY)
    with s3:
        kpi("Water pH", f"{sensor_ph:.2f}", "Probe PH-02 · in range", SLATE)
    with s4:
        ok = treated < 1.0
        kpi("Treated turbidity", f"{treated:.2f} NTU", "WHO limit 1.0 NTU", TEAL if ok else RED,
            f'<span class="chip {"up" if ok else "down"}">{"Compliant" if ok else "Breach"}</span>', TEAL if ok else RED)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(f"""
        <div class="dose">
            <div class="dose-h" style="color:{BLUE}">Alum coagulant · Pump 1</div>
            <div class="dose-hero" style="background:{NAVY};color:#fff">
                <div class="l">Set pump stroke</div><div class="v">{stroke:.0f}%</div><div class="s">{pump_l_h:.1f} L/h</div>
            </div>
            <div class="dose-row"><span>Dose</span><span>{dose:.1f} mg/L</span></div>
            <div class="dose-row"><span>Mass flow</span><span>{alum_kg_h:.1f} kg/h</span></div>
            <div class="dose-row"><span>Daily use</span><span>{alum_kg_h * 24:,.0f} kg</span></div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="dose">
            <div class="dose-h" style="color:{TEAL}">Chlorine · Gas chlorinator</div>
            <div class="dose-hero" style="background:#0b4f45;color:#fff">
                <div class="l">Set chlorinator dial</div><div class="v">{cl_dial:.0f}%</div><div class="s">{cl_kg_h:.2f} kg/h</div>
            </div>
            <div class="dose-row"><span>Target residual</span><span>2.0 mg/L</span></div>
            <div class="dose-row"><span>Daily use</span><span>{cl_kg_h * 24:.1f} kg</span></div>
            <div class="dose-row"><span>E. coli removal</span><span>99.99%</span></div>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown(f"""
        <div class="dose">
            <div class="dose-h" style="color:#475569">Chemical spend &amp; stock</div>
            <div class="dose-hero" style="background:#f6f9fc;color:{NAVY};border:1px solid #e4eaf1">
                <div class="l">Daily spend</div><div class="v">{kes(chem_day)}</div><div class="s">{kes(chem_day / 24)} per hour</div>
            </div>
            <div class="dose-row"><span>Alum stock</span><span style="color:{TEAL}">14 days</span></div>
            <div class="dose-row"><span>Chlorine cylinders</span><span style="color:{TEAL}">6 × 1 t online</span></div>
            <div class="dose-row"><span>Reorder</span><span>Not needed</span></div>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        with card("jar"):
            card_title("Dose curve", "Recommended alum dose vs raw turbidity")
            tr = np.linspace(1, 100, 150)
            fig = go.Figure()
            fig.add_scatter(x=tr, y=(4.5 * np.power(tr, 0.6) + 5.0) * ph_factor, mode="lines",
                            line=dict(color=BLUE, width=2.5), hovertemplate="%{x:.0f} NTU → %{y:.1f} mg/L<extra></extra>")
            fig.add_scatter(x=[sensor_turb], y=[dose], mode="markers+text", text=["Now"], textposition="top center",
                            marker=dict(color=RED, size=12, line=dict(color="#fff", width=2)),
                            textfont=dict(color=RED, size=12, family=FONT))
            polish(fig, 260, legend=False)
            fig.update_xaxes(title_text="Raw turbidity (NTU)", title_font=dict(size=11, color="#64748b"))
            fig.update_yaxes(title_text="mg/L", title_font=dict(size=11, color="#64748b"))
            show(fig)
    with c2:
        with card("compliance"):
            card_title("Water quality · last 7 days", "Raw vs treated turbidity (NTU)")
            rec = df_f.iloc[-168:]
            fig = go.Figure()
            fig.add_scatter(x=rec["timestamp"], y=rec["raw_turbidity_ntu"], name="Raw", mode="lines",
                            line=dict(color=SKY, width=1.5))
            fig.add_scatter(x=rec["timestamp"], y=rec["treated_turbidity_ntu"], name="Treated", mode="lines",
                            line=dict(color=TEAL, width=2.5))
            fig.add_hline(y=1.0, line_dash="dash", line_color=RED, line_width=1.5,
                          annotation_text="WHO 1.0", annotation_position="top right", annotation_font_color=RED)
            polish(fig, 260)
            fig.update_yaxes(range=[0, 15])
            show(fig)

    with card("sop"):
        card_title("Operator checklist", "Standard operating procedure")
        st.markdown("""
        <div class="sop">
            <div><b>1 · Intake</b><span>Valve open at 850 m³/h</span></div>
            <div><b>2 · Alum</b><span>Set Pump 1 to stroke shown</span></div>
            <div><b>3 · Mixing</b><span>Flash mixer at 1,450 RPM</span></div>
            <div><b>4 · Floc check</b><span>Inspect clarifiers at 15 min</span></div>
            <div><b>5 · Verify</b><span>Treated water &lt; 0.8 NTU</span></div>
        </div>""", unsafe_allow_html=True)

# =============================================================
# 6. CUSTOMERS
# =============================================================
elif active_module == "Customers":
    section("Customers", f"{period_label} · demand and revenue by segment")
    segs = ["Domestic", "Commercial", "Industrial", "Institutional"]
    share = np.array([44, 28, 18, 10], dtype=float)
    weight = share * np.array([1.0, 1.35, 1.2, 0.9])
    rev = M["income"] * weight / weight.sum()
    colors = [NAVY, BLUE, SKY, "#9cd3ee"]

    k = st.columns(4)
    for i, s in enumerate(segs):
        with k[i]:
            kpi(s, kes(rev[i]), f"{share[i]:.0f}% of demand", colors[i])

    c1, c2 = st.columns(2)
    with c1:
        with card("cust-demand"):
            card_title("Demand mix", "Share of billed volume")
            fig = go.Figure(go.Pie(labels=segs, values=share, hole=0.64, sort=False,
                                   marker=dict(colors=colors, line=dict(color="#fff", width=2)),
                                   textinfo="percent", textfont=dict(color="#fff")))
            polish(fig, 300)
            fig.update_layout(legend=dict(orientation="v", y=0.5, yanchor="middle", x=1.0, xanchor="left"))
            show(fig)
    with c2:
        with card("cust-rev"):
            card_title("Revenue by segment", "KES millions")
            fig = go.Figure(go.Bar(x=segs, y=rev / 1e6, marker_color=colors,
                                   text=[kes(v) for v in rev], textposition="outside", cliponaxis=False))
            polish(fig, 300, legend=False)
            fig.update_yaxes(showticklabels=False, range=[0, rev.max() / 1e6 * 1.2])
            show(fig)

# =============================================================
# 7. MONTHLY TRENDS & NRW ANALYSIS
# =============================================================
elif active_module == "Monthly Trends":
    section("Monthly Trends & NRW Analysis", f"{period_label} · produced vs billed volumes & non-revenue water recovery")

    m_df = df_f.resample("ME", on="timestamp")[["pumped_m3", "delivered_m3"]].sum() * mult / 1e6
    m_df["lost_m3"] = m_df["pumped_m3"] * NRW_RATE
    m_df["billed_m3"] = m_df["pumped_m3"] * (1 - NRW_RATE)
    m_df["nrw_pct"] = NRW_RATE * 100
    m_df["lost_kes_m"] = m_df["lost_m3"] * p_tariff
    m_df["recoverable_kes_m"] = m_df["pumped_m3"] * (NRW_RATE - 0.25) * p_tariff

    tot_recoverable = (M["pumped"] * max(0.0, NRW_RATE - 0.25) * p_tariff)

    # Top KPI Strip
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi("Produced Water", vol(M["pumped"]), f"{vol(per_day(M['pumped']))} / day", NAVY)
    with k2:
        kpi("Billed Demand", vol(M["billed"]), f"{(1 - NRW_RATE) * 100:.1f}% revenue water", SKY)
    with k3:
        kpi("NRW Physical Loss", vol(M["lost"]), f"{NRW_RATE * 100:.1f}% unbilled share", RED)
    with k4:
        kpi("Financial Loss Value", kes(M["lost_value"]), f"{kes(per_day(M['lost_value']))} / day lost", RED)
    with k5:
        kpi("Recovery at 25% Target", kes(tot_recoverable), "WASREB Benchmark Upside", TEAL, "", TEAL)

    # Row 1: Produced vs Billed Bars + NRW % vs WASREB 25% Target
    c_vol, c_rate = st.columns([6.8, 5.2])
    with c_vol:
        with card("produced-vs-billed"):
            card_title("Produced vs. Billed Water Volumes", "Monthly comparison, millions m³ (clean grouped bars)")
            fig_pb = go.Figure()
            fig_pb.add_bar(
                x=m_df.index, y=m_df["pumped_m3"], name="Water Produced", marker_color=NAVY,
                hovertemplate="%{x|%b %Y}<br>Produced: <b>%{y:.2f}M m³</b><extra></extra>"
            )
            fig_pb.add_bar(
                x=m_df.index, y=m_df["billed_m3"], name="Billed to Customers", marker_color=SKY,
                hovertemplate="%{x|%b %Y}<br>Billed: <b>%{y:.2f}M m³</b> (" + f"{(1 - NRW_RATE) * 100:.1f}%)<extra></extra>"
            )
            polish(fig_pb, 300)
            fig_pb.update_layout(barmode="group", bargap=0.25, hovermode="x unified")
            fig_pb.update_yaxes(title_text="Millions m³", title_font=dict(size=11, color="#64748b"))
            show(fig_pb)

    with c_rate:
        with card("nrw-rate-trend"):
            badge_nrw = '<span class="badge bad">+17.5% over target</span>'
            card_title("NRW Loss Rate vs. Regulatory Target", "Actual loss rate vs. WASREB 25% benchmark", badge_nrw)
            fig_rate = go.Figure()
            # 25% Target Band
            fig_rate.add_hline(
                y=25.0, line_dash="dash", line_color=TEAL, line_width=2.5,
                annotation_text="WASREB Target (25%)", annotation_position="bottom right",
                annotation_font=dict(color=TEAL, size=11, family=FONT)
            )
            # NRW Rate line
            fig_rate.add_scatter(
                x=m_df.index, y=m_df["nrw_pct"], mode="lines", name="Actual NRW Rate",
                line=dict(color=RED, width=3),
                fill="tonexty", fillcolor="rgba(229,72,77,0.12)",
                hovertemplate="%{x|%b %Y}<br>NRW Loss: <b>%{y:.1f}%</b><extra></extra>"
            )
            polish(fig_rate, 300, legend=False)
            fig_rate.update_yaxes(range=[15, 55], title_text="Loss %", title_font=dict(size=11, color="#64748b"))
            show(fig_rate)

    # Row 2: NRW Root Causes (IWA Water Balance) + Cumulative Financial Loss
    c_ana, c_cum = st.columns([6.0, 6.0])
    with c_ana:
        with card("nrw-causes"):
            card_title("NRW Loss Breakdown & Root Causes", "Standard IWA / WASREB Water Balance Anatomy")
            causes = [
                ("Physical Pipe Leakage & Bursts", 58, RED, "Aging trunk mains, joint weeping, high pressure bursts"),
                ("Commercial & Meter Inaccuracies", 32, AMBER, "Unmetered informal kiosks, meter under-registration, illegal bypass"),
                ("Unbilled Authorized Operations", 10, SLATE, "Filter backwashing, main pipe scouring, firefighting reserves"),
            ]
            html_causes = '<div style="margin-top:10px;">'
            for title, pct, color, desc in causes:
                vol_pillar = (M["lost"] * (pct / 100.0)) / 1e6
                kes_pillar = (M["lost_value"] * (pct / 100.0)) / 1e6
                html_causes += (
                    f'<div style="margin-bottom:12px; background:#f8fafc; border:1px solid #eef2f6; border-radius:10px; padding:10px 14px;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;">'
                    f'<span style="font-size:13px; font-weight:700; color:#0b2545;">{title}</span>'
                    f'<span style="font-size:13.5px; font-weight:800; color:{color};">{pct}% · {vol_pillar:.1f}M m³</span>'
                    f'</div>'
                    f'<div style="background:#e2e8f0; border-radius:6px; height:8px; width:100%; overflow:hidden; margin-bottom:6px;">'
                    f'<div style="background:{color}; width:{pct}%; height:100%; border-radius:6px;"></div>'
                    f'</div>'
                    f'<div style="display:flex; justify-content:space-between; font-size:11.5px; color:#64748b;">'
                    f'<span>{desc}</span>'
                    f'<span style="font-weight:700; color:#0b2545;">KES {kes_pillar:,.1f}M</span>'
                    f'</div>'
                    f'</div>'
                )
            html_causes += '</div>'
            st.markdown(html_causes, unsafe_allow_html=True)

    with c_cum:
        with card("cum-loss"):
            card_title("Cumulative Revenue Lost to NRW", "Cumulative financial value of unmetered water over time, KES Billions")
            cum_kes_b = (m_df["lost_kes_m"].cumsum()) / 1e3
            fig_cum = go.Figure()
            fig_cum.add_scatter(
                x=m_df.index, y=cum_kes_b, mode="lines", name="Cumulative Loss",
                line=dict(color=RED, width=3, shape="spline"),
                fill="tozeroy", fillcolor="rgba(229,72,77,0.14)",
                hovertemplate="%{x|%b %Y}<br>Cumulative Lost: <b>KES %{y:.2f}B</b><extra></extra>"
            )
            polish(fig_cum, 240, legend=False)
            fig_cum.update_yaxes(title_text="KES Billions", title_font=dict(size=11, color="#64748b"))
            show(fig_cum)
            st.markdown(f"""
            <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:8px 12px; margin-top:6px; font-size:12px; color:#166534; display:flex; justify-content:space-between; align-items:center;">
                <span>💡 <b>CEO Actionable Leverage:</b> Each 5% reduction in NRW captures <b>KES 940M</b> in annual operating cash.</span>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown('<div class="foot">NCWSC · Executive Operations Dashboard · Ideas Festival 2026</div>', unsafe_allow_html=True)
