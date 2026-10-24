"""
Dashboard calculation layer
===========================
Everything the dashboard shows is computed here, separately from the UI, so it
can be tested against the ground-truth fault log.

- Water balance: pumped -> used in treatment -> clean water -> delivered / lost
- Money (KES): revenue, production cost, value of lost water, energy, chemicals
- Alerts: one detector per asset, written in plain English with an action
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from preprocessing import load_and_clean_data
except ImportError:  # pragma: no cover
    from .preprocessing import load_and_clean_data

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Prices (editable from the dashboard sidebar)
# ---------------------------------------------------------------------------
@dataclass
class Prices:
    production_cost_kes_m3: float = 110.0   # cost to pump + treat 1 m3
    tariff_kes_m3: float = 150.0            # what customers pay per m3
    electricity_kes_kwh: float = 25.0       # commercial power tariff
    alum_kes_kg: float = 70.0               # aluminium sulphate, bulk


# ---------------------------------------------------------------------------
# Alert catalogue: plain-English text for every detector
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Alert catalogue: plain-English text for every detector (NCWSC Nairobi Water)
# ---------------------------------------------------------------------------
ALERTS = {
    "leak_A":  dict(asset="Zone A (Lang'ata)", title="Night leak detected", severity="warning",
                    meaning="Unusual baseline flow detected during minimum night hours (02:00-04:00).",
                    action="Deploy NCWSC acoustic pipe leak patrol team along Lang'ata estate branch mains."),
    "leak_B":  dict(asset="Zone B (Westlands)", title="Night leak detected", severity="warning",
                    meaning="Unusual baseline flow detected during minimum night hours (02:00-04:00).",
                    action="Inspect Westlands distribution boundary valves and institutional consumer meters."),
    "leak_C":  dict(asset="Zone C (Industrial Area)", title="Night leak detected", severity="warning",
                    meaning="Unusual baseline flow detected during minimum night hours (02:00-04:00).",
                    action="Check industrial ring mains and commercial meters for unmetered loss."),
    "burst_A": dict(asset="Zone A (Lang'ata)", title="Trunk pipe burst", severity="critical",
                    meaning="Sudden surge in flow (+3σ) accompanied by rapid pressure drop.",
                    action="Urgent dispatch: Isolate section valve on Lang'ata feeder line and deploy repair crew."),
    "burst_B": dict(asset="Zone B (Westlands)", title="Trunk pipe burst", severity="critical",
                    meaning="Sudden surge in flow (+3σ) accompanied by rapid pressure drop.",
                    action="Urgent dispatch: Close Westlands sector DMA valve and mobilize emergency excavation."),
    "burst_C": dict(asset="Zone C (Industrial Area)", title="Trunk pipe burst", severity="critical",
                    meaning="Sudden surge in flow (+3σ) accompanied by rapid pressure drop.",
                    action="Urgent dispatch: Shut isolation valve on Enterprise Road connector to stop pressure loss."),
    "clog_A":  dict(asset="Zone A (Lang'ata)", title="Hydraulic blockage", severity="warning",
                    meaning="Delivery flow dropped significantly below normal while line pressure rose.",
                    action="Inspect Lang'ata pressure reducing valve (PRV) and clear pipeline debris/sediment."),
    "clog_B":  dict(asset="Zone B (Westlands)", title="Hydraulic blockage", severity="warning",
                    meaning="Delivery flow dropped significantly below normal while line pressure rose.",
                    action="Check Westlands intake strainer and verify gate valves are fully open."),
    "clog_C":  dict(asset="Zone C (Industrial Area)", title="Hydraulic blockage", severity="warning",
                    meaning="Delivery flow dropped significantly below normal while line pressure rose.",
                    action="Inspect Industrial Area branch flow regulator and strainer for sediment throttling."),
    "main_leak": dict(asset="Southern Bypass Bulk Trunk", title="Bulk transmission loss (NRW)", severity="critical",
                    meaning="Water discharged from treatment plant is lost before reaching district DMA bulk meters.",
                    action="Immediate bulk main patrol: Inspect Southern Bypass corridor for ground water surfacing."),
    "bh1_out": dict(asset="Karen Wellfield (BH1)", title="Intake pump outage", severity="critical",
                    meaning="Borehole 1 flow dropped to zero; Lang'ata wellfield forced into continuous emergency duty.",
                    action="Check borehole electrical switchgear, VFD inverter, and motor thermal overload relay."),
    "bh2_out": dict(asset="Lang'ata Aquifer (BH2)", title="Intake pump outage", severity="critical",
                    meaning="Borehole 2 flow dropped to zero; Karen wellfield forced into continuous emergency duty.",
                    action="Check borehole electrical switchgear, VFD inverter, and motor thermal overload relay."),
    "bh2_screen": dict(asset="Lang'ata Aquifer (BH2)", title="Well screen sand clogging", severity="warning",
                    meaning="Pumping energy per m³ rising steadily due to borehole screen fouling and sand ingress.",
                    action="Schedule well air-lift flushing and acidization rehabilitation with hydrogeology crew."),
    "filter_dirty": dict(asset="Rapid Sand Filter Works", title="Filter media fouling", severity="warning",
                    meaning="Post-backwash clean headloss (dP floor) exceeds 26 kPa threshold.",
                    action="Perform extended air-scour backwash cycle. Plan silica sand media regeneration."),
    "turbidity": dict(asset="Treated Water Quality", title="WHO turbidity breach", severity="critical",
                    meaning="Treated effluent turbidity exceeds safe drinking water standard (1.0 NTU limit).",
                    action="Increase alum coagulant injection pump rate immediately; divert non-compliant effluent."),
    "sensor_stuck": dict(asset="Telemetry Sensors", title="Turbidity sensor flatline", severity="warning",
                    meaning="Turbidity sensor output has frozen at static reading with zero variance.",
                    action="Clean optical sensor cuvette and recalibrate with 4000 NTU formazin standard."),
    "comms": dict(asset="Telemetry Sensors", title="IoT telemetry dropout", severity="warning",
                    meaning="Data packets lost across multiple remote telemetry units (RTUs).",
                    action="Verify cellular GSM/GPRS modem link and solar battery backup at outstation."),
}

ASSETS = [
    "Karen Wellfield (BH1)", "Lang'ata Aquifer (BH2)", "Rapid Sand Filter Works",
    "Treated Water Quality", "Southern Bypass Bulk Trunk", "Zone A (Lang'ata)",
    "Zone B (Westlands)", "Zone C (Industrial Area)", "Telemetry Sensors"
]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _sustained(mask: pd.Series, hours: int) -> pd.Series:
    """True only when the condition has held for `hours` consecutive hours."""
    return mask.astype(int).rolling(hours, min_periods=hours).min().fillna(0).astype(bool)


def detect_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Adds one boolean column per alert (prefix `a_`) plus loss estimates.

    Daily detectors (hidden leak, blocked pipe) use completed days only and are
    applied from the following day, so a live replay never sees future data.
    """
    d = df.copy()
    ts = d["timestamp"]
    hour = ts.dt.hour
    day = ts.dt.normalize()

    daily = d.set_index("timestamp")[["bldgA_flow_m3h", "bldgB_flow_m3h", "bldgC_flow_m3h"]].resample("D").mean()
    total = daily.sum(axis=1)

    def same_weekday_ref(s):
        return pd.concat([s.shift(7 * k) for k in range(1, 6)], axis=1).median(axis=1)

    total_ratio = (total / same_weekday_ref(total)).rolling(3, min_periods=2).median()
    normal_week = (total_ratio > 0.85) & (total_ratio < 1.15)   # not a holiday / term change

    for b in ["A", "B", "C"]:
        col = f"bldg{b}_flow_m3h"
        q, p = d[col], d[f"bldg{b}_pressure_bar"]

        # Hidden leak: night/day flow ratio. Occupancy scales night and day together,
        # a leak adds a constant flow, so the ratio rises.
        nmask = hour.isin([2, 3, 4])
        night = q[nmask].groupby(day[nmask]).mean().reindex(daily.index)
        ratio_raw = night / daily[col]
        ratio = ratio_raw.rolling(5, min_periods=3).mean()
        base = ratio_raw.shift(7).rolling(56, min_periods=21).quantile(0.25)
        sd = ratio_raw.shift(7).rolling(56, min_periods=21).std()
        z = (ratio - base) / sd
        excess = ((ratio - base) * daily[col]).clip(lower=0)
        d[f"leak_excess_{b}"] = day.map(excess.shift(1)).astype(float).fillna(0)
        d[f"a_leak_{b}"] = day.map((z > 1.5).astype(float).shift(1)).fillna(0).astype(bool)

        # Burst: flow far above the same hour last week AND pressure drop
        q_ref = q.shift(168).rolling(3, center=True, min_periods=1).mean()
        p_ref = p.rolling(168, min_periods=48).median().shift(24)
        d[f"burst_excess_{b}"] = (q - q_ref).clip(lower=0)
        d[f"a_burst_{b}"] = _sustained(((q - q_ref) > 3.0) & ((p_ref - p) > 0.8), 2)

        # Blocked pipe: building's share of campus water drops vs the same weekday
        # in previous weeks, while total campus demand is normal (not a holiday).
        share = daily[col] / total
        share_ratio = (share / same_weekday_ref(share)).rolling(3, min_periods=2).median()
        clog = (share_ratio < 0.85) & normal_week
        d[f"a_clog_{b}"] = day.map(clog.astype(float).shift(1)).fillna(0).astype(bool)

    # Main pipe leak: water unaccounted between plant and buildings (2-day average)
    d["main_loss_2d"] = d["main_loss_m3h"].rolling(48, min_periods=12).mean()
    d["a_main_leak"] = d["main_loss_2d"] > 0.6

    # Pump outage: a whole day with no output while the other pump works harder
    for me, other in [("bh1", "bh2"), ("bh2", "bh1")]:
        out24 = d[f"{me}_flow_m3h"].rolling(24, min_periods=12).sum()
        typ = d[f"{me}_flow_m3h"].shift(24).rolling(24 * 14, min_periods=48).mean() * 24
        oth24 = d[f"{other}_runtime_min"].rolling(24, min_periods=12).sum()
        oth_typ = d[f"{other}_runtime_min"].shift(24).rolling(24 * 14, min_periods=48).mean() * 24
        d[f"a_{me}_out"] = (out24 < 0.05 * typ) & (oth24 > 1.3 * oth_typ)

    # Borehole 2 screen clogging: pump minutes per m3 drift up (normal = 3.0)
    mpm = d["bh2_min_per_m3"].where(d["bh2_flow_m3h"] > 0.5)
    d["bh2_min_per_m3_7d"] = mpm.rolling(168, min_periods=24).median().ffill().fillna(3.0)
    d["a_bh2_screen"] = d["bh2_min_per_m3_7d"] > 3.25

    # Filter: dP right after backwash should be ~18-20 kPa
    d["a_filter_dirty"] = d["filter_dp_floor_kpa"] > 26.0

    # Treated water turbidity: 3 h average above the WHO drinking-water limit (1 NTU)
    d["turbidity_3h"] = d["treated_turbidity_ntu"].rolling(3, min_periods=1).mean()
    d["a_turbidity"] = d["turbidity_3h"] > 1.0

    # Sensors
    std12 = d["treated_turbidity_ntu"].rolling(12, min_periods=12).std()
    d["a_sensor_stuck"] = std12 < 0.002
    d["a_comms"] = d["is_logger_dropout"] == 1

    # Overall status
    keys = list(ALERTS.keys())
    crit = [f"a_{k}" for k in keys if ALERTS[k]["severity"] == "critical"]
    warn = [f"a_{k}" for k in keys if ALERTS[k]["severity"] == "warning"]
    d["n_critical"] = d[crit].sum(axis=1)
    d["n_warning"] = d[warn].sum(axis=1)
    d["status"] = np.select([d["n_critical"] > 0, d["n_warning"] > 0], ["critical", "warning"], "ok")
    return d


# ---------------------------------------------------------------------------
# Water balance, money, chemicals, energy
# ---------------------------------------------------------------------------
def add_operational_metrics(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["pumped_m3"] = d["bh1_flow_m3h"] + d["bh2_flow_m3h"]
    d["clean_m3"] = d["treatment_outlet_m3h"]
    d["treatment_use_m3"] = (d["treatment_inlet_m3h"] - d["treatment_outlet_m3h"]).clip(lower=0)
    d["delivered_m3"] = d["sum_bldg_flow_m3h"]
    d["lost_m3"] = (d["clean_m3"] - d["delivered_m3"]).clip(lower=0)
    d["energy_kwh"] = d["bh1_pump_kw"] + d["bh2_pump_kw"]

    # Coagulant (alum) dose from raw turbidity: standard power-law jar-test fit
    raw = d["raw_turbidity_ntu"].clip(lower=0.5)
    d["dose_mg_l"] = 4.5 * np.power(raw, 0.6) + 5.0
    # 100 g/L stock solution -> mL/min = dose(mg/L) * flow(m3/h) / 6
    d["dose_pump_ml_min"] = d["dose_mg_l"] * d["treatment_inlet_m3h"].clip(lower=0) / 6.0
    d["alum_kg"] = d["dose_mg_l"] * d["treatment_inlet_m3h"].clip(lower=0) / 1000.0

    d["filter_health_pct"] = (100 * (1 - (d["filter_dp_floor_kpa"] - 18) / (48 - 18))).clip(0, 100)
    d["bh2_efficiency_pct"] = (100 * 3.0 / d["bh2_min_per_m3_7d"]).clip(0, 100)
    return d


def money(period: pd.DataFrame, prices: Prices) -> dict:
    """KES summary for a slice of hourly rows."""
    pumped = period["pumped_m3"].sum()
    delivered = period["delivered_m3"].sum()
    # totals first, then difference: hourly meter jitter cancels out
    lost = max(period["clean_m3"].sum() - delivered, 0.0)
    used = max(period["treatment_inlet_m3h"].sum() - period["clean_m3"].sum(), 0.0)
    return dict(
        pumped_m3=pumped, delivered_m3=delivered, lost_m3=lost, treatment_use_m3=used,
        loss_pct=100 * lost / pumped if pumped > 0 else 0.0,
        revenue=delivered * prices.tariff_kes_m3,
        production_cost=pumped * prices.production_cost_kes_m3,
        lost_value=lost * prices.tariff_kes_m3,
        energy_kwh=period["energy_kwh"].sum(),
        energy_cost=period["energy_kwh"].sum() * prices.electricity_kes_kwh,
        alum_kg=period["alum_kg"].sum(),
        alum_cost=period["alum_kg"].sum() * prices.alum_kes_kg,
        margin=delivered * prices.tariff_kes_m3 - pumped * prices.production_cost_kes_m3,
    )


def active_alerts(row: pd.Series, prices: Prices) -> list:
    """Alerts active at one hour, with KES/day impact where it can be measured."""
    out = []
    for key, meta in ALERTS.items():
        if not bool(row.get(f"a_{key}", False)):
            continue
        cost_day = None
        if key.startswith("leak_"):
            cost_day = row[f"leak_excess_{key[-1]}"] * 24 * prices.tariff_kes_m3
        elif key.startswith("burst_"):
            cost_day = row[f"burst_excess_{key[-1]}"] * 24 * prices.tariff_kes_m3
        elif key == "main_leak":
            cost_day = max(row["main_loss_2d"], 0) * 24 * prices.tariff_kes_m3
        elif key == "bh2_screen":
            extra = max(row["bh2_min_per_m3_7d"] / 3.0 - 1, 0)
            cost_day = extra * row.get("bh2_kwh_day", 0) * prices.electricity_kes_kwh
        out.append(dict(key=key, cost_day=cost_day, **meta))
    out.sort(key=lambda a: (a["severity"] != "critical", -(a["cost_day"] or 0)))
    return out


def alert_episodes(d: pd.DataFrame) -> pd.DataFrame:
    """Contiguous alert periods, for the incident timeline."""
    rows = []
    for key, meta in ALERTS.items():
        col = f"a_{key}"
        s = d[col].astype(bool).values
        if not s.any():
            continue
        t = d["timestamp"].values
        edges = np.diff(np.concatenate([[0], s.astype(int), [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
        for a, b in zip(starts, ends):
            rows.append(dict(key=key, asset=meta["asset"], title=meta["title"],
                             severity=meta["severity"], start=t[a], end=t[b] + np.timedelta64(1, "h"),
                             hours=b - a + 1))
    ep = pd.DataFrame(rows)
    if ep.empty:
        return ep
    # merge episodes of the same alert separated by < 12 h so the timeline is readable
    ep = ep.sort_values(["key", "start"])
    merged = []
    for _, g in ep.groupby("key"):
        cur = None
        for r in g.to_dict("records"):
            if cur and (r["start"] - cur["end"]) < np.timedelta64(12, "h"):
                cur["end"] = r["end"]
                cur["hours"] += r["hours"]
            else:
                if cur:
                    merged.append(cur)
                cur = r
        merged.append(cur)
    return pd.DataFrame(merged).sort_values("start").reset_index(drop=True)


def maintenance_outlook(d: pd.DataFrame, now_idx: int) -> dict:
    """Simple, explainable trend projections (linear, last 14 days)."""
    win = d.iloc[max(0, now_idx - 336): now_idx + 1]
    res = {}

    def project(series, limit, rising=True):
        y = series.dropna()
        if len(y) < 48:
            return None
        x = np.arange(len(y)) / 24.0
        slope = np.polyfit(x, y.values, 1)[0]
        cur = y.iloc[-1]
        if (rising and slope <= 1e-4) or (not rising and slope >= -1e-4):
            return None
        days = (limit - cur) / slope
        return float(days) if days > 0 else 0.0

    res["filter_days"] = project(win["filter_dp_floor_kpa"], 40.0)
    res["bh2_days"] = project(win["bh2_min_per_m3_7d"], 3.9)
    return res


# ---------------------------------------------------------------------------
# One-shot loader used by the app (cached there)
# ---------------------------------------------------------------------------
def build_dataset() -> pd.DataFrame:
    df = load_and_clean_data(BASE_DIR / "water_hourly_2023_2025.csv")
    df = detect_alerts(df)
    df = add_operational_metrics(df)
    df["bh2_kwh_day"] = df["bh2_pump_kw"].rolling(24, min_periods=1).sum()
    return df


def load_ground_truth() -> pd.DataFrame:
    gt = pd.read_csv(BASE_DIR / "fault_log_ground_truth.csv")
    gt = gt[gt["start"] != "random"].copy()
    gt["start"] = pd.to_datetime(gt["start"])
    gt["end"] = pd.to_datetime(gt["end"])
    return gt
