# Synthetic hourly water dataset, 2023-01-01 to 2025-12-31

Synthetic (simulated, seed 2026). 26,304 hourly rows (2024 is a leap year). Timestamps are local time (EAT).
Meant for testing leak, clog and flow-analysis methods. It is not real measurement data.

## System layout
Borehole 1 + Borehole 2 -> treatment plant (filter + turbidity monitoring) -> Building A (hostel), Building B (offices/lecture), Building C (cafeteria + labs).

## Columns
| Column | Unit | Meaning |
|---|---|---|
| timestamp | local | Start of the hour |
| rainfall_mm | mm/h | Synthetic rain; long rains Mar-May, short rains Oct-Dec |
| bldgX_flow_m3h | m3/h | Metered inflow to building A, B, C (volume in that hour) |
| bldgX_pressure_bar | bar | Pressure at each building inlet meter |
| bhN_flow_m3h | m3/h | Borehole N average flow in the hour |
| bhN_runtime_min | min | Pump run time in the hour (0-60) |
| bhN_water_level_m | m | Depth to water below ground (higher = deeper) |
| bhN_pump_kw | kW | Average pump power in the hour |
| raw_turbidity_ntu | NTU | Blended raw water, before filter |
| treated_turbidity_ntu | NTU | After filter |
| filter_dp_kpa | kPa | Pressure drop across the filter (sawtooth; resets on backwash) |
| backwash_flag | 0/1 | 1 in hours when a backwash occurred |
| treatment_inlet_m3h | m3/h | Water entering treatment (about outlet / 0.96, plus backwash water) |
| treatment_outlet_m3h | m3/h | Treated water leaving the plant (includes unmetered main-line losses) |

## Built-in realism
Daily and weekly demand cycles per building, academic-break dips, mild seasonality and 3%/yr growth, weekly lead/lag rotation between boreholes, slow aquifer decline,
filter ripening spikes after backwash, missing values (~0.07%), a stuck sensor, a logger dropout and random spikes.

## Faults
Ground truth with dates and expected signatures is in `fault_log_ground_truth.csv`. Open it after your own analysis if you want a blind test.

## Quick diagnostic hints
- Leak: higher night-time (02:00-04:00) flow, pressure down. Unmetered main leak: treatment_outlet minus sum of building meters rises.
- Clog (pipe): flow down, pressure up. Filter clog: dp floor after backwash creeps up, backwashes get more frequent, treated turbidity rises.
- Borehole screen clog: more runtime and drawdown per m3, specific energy (kWh/m3) up.
