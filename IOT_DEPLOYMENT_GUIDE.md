# 💧 HydroTwin IoT & Machine Learning Deployment Guide

This guide explains the end-to-end architecture, why each machine learning and physical model was chosen, how the real-time dashboard works, and how to connect physical IoT sensors.

---

## 🏗️ 1. Architecture: What Was Built & Why

Water networks and treatment plants behave as coupled physical systems: pressure, flow rate, electric pump power, and water quality are tightly bound by physical laws. Standard machine learning models without physical context fail because leaks and clogs look very different across hours of the day.

We designed a **hybrid physics-informed machine learning system**:

```
                         ┌──────────────────────────────────────────────┐
                         │   Physical IoT Telemetry Ingestion (14 Pts)  │
                         └──────────────────────┬───────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
  ┌─────────────────────────────┐                               ┌─────────────────────────────┐
  │   Physics Feature Engine    │                               │  Time-Series Harmonic Lags  │
  │  - DMA Mass Balance (Loss)  │                               │  - Sin/Cos Fourier (Hour)   │
  │  - Specific Energy (kWh/m³) │                               │  - Multi-Horizon Lags       │
  │  - Filter dP Floor (Clean)  │                               │    (t-1, t-2, t-24, t-168)  │
  │  - Minimum Night Flow (MNF) │                               │  - Rolling Means & Stds     │
  └──────────────┬──────────────┘                               └──────────────┬──────────────┘
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐                  ┌──────────────────┐
│ Time-Series      │                  │ Supervised Multi-│                  │ Unsupervised     │
│ Forecaster       │                  │ Fault Classifier │                  │ Anomaly Detector │
│ (HGB Regressor)  │                  │ (HGB Classifier) │                  │(Isolation Forest)│
└────────┬─────────┘                  └────────┬─────────┘                  └────────┬─────────┘
         │                                      │                                      │
         │ Residual e_t = y - ŷ                 │ Fault Probabilities P(Fault)         │ Anomaly Score
         ▼                                      ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                      Unified Predictive Breakdown Sentinel & RUL Prognostics                 │
│              - Real-Time Fault Alert (Burst, Clog, Fouling, Outage, Screen Sanding)          │
│              - Days to Filter Media Replacement | Days to Borehole Rehabilitation            │
│              - Recommended Alum Coagulant Dosing Flow (mL/min)                               │
└──────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                               │
                                               ▼
                             ┌──────────────────────────────────┐
                             │  Real-Time Interactive Dashboard │
                             │          (app.py / Streamlit)    │
                             └──────────────────────────────────┘
```

---

### Why Each Model Was Chosen:

1. **Time-Series Demand Forecaster (`HistGradientBoostingRegressor`)**:
   - **Why:** Water consumption has heavy daily, weekly, and academic seasonal cycles. By training multi-horizon regressors ($R^2 \ge 0.97$), the model learns the exact expected demand profile for any hour.
   - **Why Residuals?** If actual flow exceeds the predicted flow by $+3\sigma$, a **burst** has occurred. If actual flow drops by $-3\sigma$ while pressure spikes, a **pipe clog or valve shutoff** has occurred.

2. **Supervised Multi-Fault Breakdown Classifier (`HistGradientBoostingClassifier`)**:
   - **Why:** When a failure happens, operators need to know *what* failed and *where*.
   - **Categories:** `LEAK_BURST`, `PIPE_CLOG`, `FILTER_FOULING`, `PUMP_OUTAGE`, `SCREEN_CLOGGING`, `SENSOR_FAULT`.
   - **Why HistGradientBoosting?** It natively handles high feature dimensionality, missing telemetry, non-linear coupling, and executes in sub-millisecond speeds on edge microcomputers (Raspberry Pi).

3. **Unsupervised Anomaly Detector (`IsolationForest`)**:
   - **Why:** Real-world IoT systems encounter novel faults never seen in historical logs. Isolation Forest scores multivariate deviations without requiring training labels.

4. **Predictive Prognostics & Remaining Useful Life (RUL)**:
   - **Why:** Catastrophic breakdowns rarely occur instantly.
     - *Filter media clogging:* Post-backwash pressure drop baseline ($dP_{floor}$) creeps from $18\text{ kPa}$ to $48\text{ kPa}$.
     - *Borehole screen clogging:* Specific energy creeps from $0.38$ to $>0.55\text{ kWh}/\text{m}^3$.
   - Extrapolating these degradation rates gives operators **Days to Failure**, allowing proactive scheduled maintenance before total shutdown.

5. **Automated Coagulant Dosing Advisory**:
   - **Why:** Doses coagulant (Alum / PAC) based on raw water turbidity power law ($Dose = 4.5 \cdot \text{Turb}^{0.6} + 5.0$) to guarantee WHO $<0.5\text{ NTU}$ compliance.

---

## 🖥️ 2. Running the Real-Time Dashboard

The dashboard is built with **Streamlit** and provides:
- Live IoT playback controller (scrub to real historical bursts, clogs, and pump outages, or run live streaming).
- Live KPI cards with color-coded alerts (Normal 🟢, Warning 🟡, Critical 🔴).
- Live Time-Series Forecast vs Telemetry with 95% Confidence Interval.
- Breakdown diagnosis and class probabilities.
- Asset prognostics gauges (Filter RUL & Well RUL).
- Coagulant chemical dosing calculator.

### How to Run Locally:
In your terminal, navigate to the folder and run:
```bash
streamlit run app.py
```
This opens the dashboard in your browser at `http://localhost:8501`.

---

## ☁️ 3. Deploying to the Cloud (Free & Shareable)

You can deploy this dashboard so anyone can access it over the internet:

### Option A: Streamlit Community Cloud (Recommended - 100% Free)
1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "HydroTwin ML & IoT Dashboard"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select your repository.
4. Set Main file path: `app.py`.
5. Click **Deploy**. Your dashboard will be live on a public URL (e.g. `https://hydrotwin.streamlit.app`).

### Option B: Google Colab Running Streamlit via Localtunnel
In Google Colab, you can launch Streamlit directly:
```python
!pip install streamlit localtunnel -q
!streamlit run app.py & npx localtunnel --port 8501
```

---

## 📡 4. Connecting Physical IoT Hardware

When your physical IoT devices are deployed, they can push data into this pipeline via **MQTT** or **HTTP REST API**.

### Expected JSON Telemetry Payload:
```json
{
  "timestamp": "2026-10-21T16:00:00",
  "rainfall_mm": 0.0,
  "bldgA_flow_m3h": 1.25,
  "bldgA_pressure_bar": 2.45,
  "bldgB_flow_m3h": 0.85,
  "bldgB_pressure_bar": 2.50,
  "bldgC_flow_m3h": 1.40,
  "bldgC_pressure_bar": 2.38,
  "bh1_flow_m3h": 3.80,
  "bh1_runtime_min": 60.0,
  "bh1_water_level_m": 35.2,
  "bh1_pump_kw": 1.35,
  "bh2_flow_m3h": 0.0,
  "bh2_runtime_min": 0.0,
  "bh2_water_level_m": 38.0,
  "bh2_pump_kw": 0.0,
  "raw_turbidity_ntu": 4.5,
  "treated_turbidity_ntu": 0.28,
  "filter_dp_kpa": 22.0,
  "backwash_flag": 0,
  "treatment_inlet_m3h": 3.95,
  "treatment_outlet_m3h": 3.55
}
```

### Python IoT Ingestion Hook:
To connect incoming MQTT packets into the model pipeline:
```python
from ml_models import WaterMLPipeline
import pandas as pd

# Load pretrained pipeline
pipeline = WaterMLPipeline()
pipeline.load("models_saved")

def on_mqtt_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    new_reading_df = pd.DataFrame([payload])
    
    # Run ML breakdown inference
    predictions = pipeline.predict(new_reading_df)
    
    # Check for alerts
    risk = predictions['composite_breakdown_risk'].iloc[-1]
    fault = predictions['ml_predicted_fault'].iloc[-1]
    
    if risk > 0.65:
        trigger_scada_alarm(fault, risk)
```
