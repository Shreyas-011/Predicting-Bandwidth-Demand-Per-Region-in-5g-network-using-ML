"""
5G Network Bandwidth Demand Dataset Generator
Generates a realistic synthetic dataset for ML-based bandwidth prediction per region.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────
REGIONS = [
    "Mumbai_Urban", "Delhi_Metro", "Bangalore_Tech", "Chennai_Port",
    "Kolkata_Central", "Hyderabad_IT", "Pune_Industrial", "Ahmedabad_Commercial",
    "Jaipur_Tourist", "Surat_Trade"
]

NETWORK_TYPES = ["5G_SA", "5G_NSA", "5G_mmWave"]

START_DATE = datetime(2023, 1, 1)
NUM_DAYS   = 730   # 2 years
SAMPLES    = 10000


def region_profile(region):
    """Return base bandwidth (Gbps) and peak-hour multiplier per region."""
    profiles = {
        "Mumbai_Urban":        (85, 2.4, "high_density"),
        "Delhi_Metro":         (92, 2.6, "high_density"),
        "Bangalore_Tech":      (78, 2.2, "tech_hub"),
        "Chennai_Port":        (62, 1.9, "mixed"),
        "Kolkata_Central":     (55, 1.8, "medium"),
        "Hyderabad_IT":        (70, 2.1, "tech_hub"),
        "Pune_Industrial":     (60, 1.7, "industrial"),
        "Ahmedabad_Commercial":(58, 1.6, "commercial"),
        "Jaipur_Tourist":      (40, 1.5, "tourist"),
        "Surat_Trade":         (45, 1.6, "commercial"),
    }
    return profiles.get(region, (50, 1.8, "medium"))


def time_of_day_factor(hour):
    """Hour 0-23 → multiplier (peak evening, low night)."""
    if 0 <= hour < 6:
        return 0.3 + 0.05 * hour
    elif 6 <= hour < 9:
        return 0.5 + 0.15 * (hour - 6)
    elif 9 <= hour < 12:
        return 0.85 + 0.03 * (hour - 9)
    elif 12 <= hour < 14:
        return 0.95 + 0.02 * (hour - 12)
    elif 14 <= hour < 18:
        return 0.90 + 0.025 * (hour - 14)
    elif 18 <= hour < 21:
        return 1.00 + 0.05 * (hour - 18)   # peak
    else:
        return max(0.6, 1.15 - 0.1 * (hour - 21))


def day_of_week_factor(dow):
    """Monday=0 … Sunday=6."""
    factors = [0.95, 0.97, 0.98, 0.96, 1.05, 1.20, 1.15]
    return factors[dow]


def season_factor(month):
    """1-12 → multiplier."""
    seasonal = {1:1.0, 2:0.95, 3:1.0, 4:1.05, 5:1.1,
                6:1.15, 7:1.2, 8:1.18, 9:1.05, 10:1.0, 11:1.1, 12:1.25}
    return seasonal[month]


# ─── Main generation ─────────────────────────────────────────────────────────
rows = []

for _ in range(SAMPLES):
    region = random.choice(REGIONS)
    base_bw, peak_mult, region_type = region_profile(region)
    network_type = random.choice(NETWORK_TYPES)

    # Random timestamp within 2 years
    offset = timedelta(days=random.randint(0, NUM_DAYS - 1),
                       hours=random.randint(0, 23),
                       minutes=random.randint(0, 59))
    ts = START_DATE + offset

    hour  = ts.hour
    dow   = ts.weekday()
    month = ts.month

    # Composite factors
    tod_f  = time_of_day_factor(hour)
    dow_f  = day_of_week_factor(dow)
    seas_f = season_factor(month)

    # Active users (thousands)
    base_users = {"high_density": 850, "tech_hub": 620, "mixed": 450,
                  "medium": 380, "industrial": 320, "commercial": 340, "tourist": 250}
    active_users = int(base_users.get(region_type, 400)
                       * tod_f * dow_f * np.random.normal(1.0, 0.08))
    active_users = max(50, active_users)

    # Latency (ms) — inversely related to available capacity
    latency = round(np.random.normal(8, 2) + (1 - tod_f) * 5, 2)
    latency = max(1.5, min(50, latency))

    # Signal strength (dBm)
    signal_strength = round(np.random.normal(-72, 8), 1)
    signal_strength = max(-110, min(-40, signal_strength))

    # Packet loss (%)
    packet_loss = round(max(0, np.random.exponential(0.3) + (1 - tod_f) * 0.5), 3)

    # IoT devices connected
    iot_devices = int(np.random.poisson(200 * (base_bw / 100)))

    # Video streaming ratio (% of traffic)
    video_ratio = round(min(0.95, max(0.2,
                        np.random.normal(0.55 + 0.15 * tod_f, 0.1))), 3)

    # Network slicing factor (0-1, higher = more slices active)
    slicing_factor = round(np.random.uniform(0.4, 1.0), 3)

    # Temperature (affects hardware performance)
    temp_celsius = round(np.random.normal(28 + 6 * np.sin((month - 4) * np.pi / 6), 3), 1)

    # Cell tower load (%)
    tower_load = round(min(100, max(10,
                       tod_f * peak_mult * 42 * np.random.normal(1, 0.12))), 1)

    # Special event flag
    is_special_event = int(np.random.random() < 0.05)

    # Bandwidth demand (Gbps) — the TARGET
    bandwidth = (
        base_bw
        * tod_f * peak_mult
        * dow_f * seas_f
        * (1 + 0.3 * is_special_event)
        * np.random.normal(1.0, 0.07)
        * (slicing_factor * 0.5 + 0.5)
    )
    bandwidth = round(max(5.0, min(300.0, bandwidth)), 3)

    rows.append({
        "timestamp":           ts.strftime("%Y-%m-%d %H:%M:%S"),
        "region":              region,
        "network_type":        network_type,
        "hour_of_day":         hour,
        "day_of_week":         dow,
        "month":               month,
        "is_weekend":          int(dow >= 5),
        "is_special_event":    is_special_event,
        "active_users_k":      active_users,
        "latency_ms":          latency,
        "signal_strength_dbm": signal_strength,
        "packet_loss_pct":     packet_loss,
        "iot_devices":         iot_devices,
        "video_ratio":         video_ratio,
        "slicing_factor":      slicing_factor,
        "tower_load_pct":      tower_load,
        "temperature_c":       temp_celsius,
        "bandwidth_gbps":      bandwidth,   # ← TARGET
    })

df = pd.DataFrame(rows)

# Region-type lookup (numeric encoding helper)
region_type_map = {r: region_profile(r)[2] for r in REGIONS}
df["region_type"] = df["region"].map(region_type_map)

# One-hot encode categoricals for ML convenience (saved as separate columns)
region_dummies     = pd.get_dummies(df["region"],       prefix="reg")
net_type_dummies   = pd.get_dummies(df["network_type"], prefix="net")
reg_type_dummies   = pd.get_dummies(df["region_type"],  prefix="rtype")

df = pd.concat([df, region_dummies, net_type_dummies, reg_type_dummies], axis=1)

out_path = "/mnt/user-data/outputs/5g_bandwidth_dataset.csv"
df.to_csv(out_path, index=False)

print(f"✅  Dataset saved → {out_path}")
print(f"   Shape  : {df.shape}")
print(f"   Target : bandwidth_gbps  [{df['bandwidth_gbps'].min():.1f} – {df['bandwidth_gbps'].max():.1f} Gbps]")
print(f"\nFeature summary:")
print(df.describe().T[["mean","std","min","max"]].round(2))
