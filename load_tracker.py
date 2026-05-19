import fitdecode
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

#Upload your .fit files to the sessions/ folder inside the repo before running this script. It will automatically find and process all .fit files in that directory, so you can easily track your training load over time as you add new sessions.
SESSIONS_DIR = Path("sessions")

def load_session(filepath):
    """Parse a single .fit file and return a session summary dict, or None on failure."""
    records = []
    try:
        with fitdecode.FitReader(str(filepath)) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage) and frame.name == "record":
                    row = {f.name: f.value for f in frame.fields if f.value is not None}
                    if row:
                        records.append(row)
    except Exception as e:
        print(f"  Skipping {filepath.name}: {e}")
        return None

    if not records:
        return None

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    duration_min = (df["timestamp"].max() - df["timestamp"].min()).seconds / 60
    avg_hr = df["heart_rate"].mean() if "heart_rate" in df.columns else 130
    trimp = duration_min * (avg_hr / 185) * 0.64 * (2.718 ** (1.92 * avg_hr / 185))

    return {
        "date": df["timestamp"].min().date(),
        "duration_min": round(duration_min, 1),
        "avg_hr": round(avg_hr, 1),
        "load_au": round(trimp, 1),
        "file": filepath.name,
    }


# ── Load all sessions ─────────────────────────────────────────────────────────
fit_files = sorted(SESSIONS_DIR.glob("*.fit"))
if not fit_files:
    raise FileNotFoundError(f"No .fit files found in {SESSIONS_DIR}")

print(f"Found {len(fit_files)} .fit file(s) in {SESSIONS_DIR}\n")
sessions = [load_session(f) for f in fit_files]
sessions = [s for s in sessions if s]

raw = pd.DataFrame(sessions).sort_values("date").reset_index(drop=True)
print(raw[["date", "duration_min", "avg_hr", "load_au"]].to_string(index=False))
print()

# ── Calendar-day rolling averages ─────────────────────────────────────────────
# Reindex to a continuous daily series so rest days count as zero load.
# Without this, a 7-session window could span months on a low-volume athlete.
raw["date"] = pd.to_datetime(raw["date"])
daily = (
    raw.groupby("date")["load_au"]
    .sum()
    .reindex(pd.date_range(raw["date"].min(), raw["date"].max(), freq="D"), fill_value=0)
    .rename("load_au")
    .reset_index()
    .rename(columns={"index": "date"})
)

daily["load_7d"] = daily["load_au"].rolling(7,  min_periods=1).mean()
daily["load_28d"] = daily["load_au"].rolling(28, min_periods=1).mean()
daily["acwr"] = (daily["load_7d"] / daily["load_28d"].replace(0, float("nan"))).round(2)

# ── Colour helpers ────────────────────────────────────────────────────────────
def acwr_color(val):
    if pd.isna(val) or val < 0.8:
        return "#8A96A8"   # grey  – undertraining
    elif val <= 1.3:
        return "#0EDF6C"   # green – optimal
    elif val <= 1.5:
        return "#FCC547"   # amber – caution
    else:
        return "#E22007"   # red   – overtraining risk

bar_colors = [acwr_color(v) for v in daily["acwr"]]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.patch.set_facecolor("#1A1A1A")
for ax in (ax1, ax2):
    ax.set_facecolor("#1A1A1A")
    ax.tick_params(colors="#CCCCCC")
    ax.xaxis.label.set_color("#CCCCCC")
    ax.yaxis.label.set_color("#CCCCCC")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

x = range(len(daily))

ax1.bar(x, daily["load_au"], color=bar_colors, alpha=0.85, width=0.9)
ax1.plot(x, daily["load_7d"],  color="#534AB7", linewidth=2, label="7-day avg (ATL)")
ax1.plot(x, daily["load_28d"], color="#0F6E56", linewidth=2, label="28-day avg (CTL)")
ax1.set_ylabel("Load (AU)", color="#CCCCCC")
ax1.set_title("Training Load & Acute:Chronic Workload Ratio", color="#EEEEEE", fontsize=13)
ax1.legend(fontsize=9, facecolor="#2A2A2A", labelcolor="#CCCCCC", framealpha=0.8)

ax2.plot(x, daily["acwr"], color="#EEEEEE", linewidth=2, marker="o", markersize=3)
ax2.axhspan(0,    0.8,  alpha=0.10, color="gray")
ax2.axhspan(0.8,  1.3,  alpha=0.10, color="green")
ax2.axhspan(1.3,  1.5,  alpha=0.10, color="orange")
ax2.axhspan(1.5,  3.0,  alpha=0.10, color="red")
ax2.axhline(0.8,  color="#8A96A8", linewidth=0.7, linestyle="--")
ax2.axhline(1.3,  color="#FCC547", linewidth=0.7, linestyle="--")
ax2.axhline(1.5,  color="#E22007", linewidth=0.7, linestyle="--")
ax2.set_ylabel("ACWR", color="#CCCCCC")
ax2.set_ylim(0, 2.2)

# x-axis: show one label per week to avoid crowding
tick_step = max(1, len(daily) // 20)
ax2.set_xticks(list(range(0, len(daily), tick_step)))
ax2.set_xticklabels(
    [str(daily["date"].iloc[i].date()) for i in range(0, len(daily), tick_step)],
    rotation=45, ha="right", fontsize=8, color="#CCCCCC",
)

legend_patches = [
    mpatches.Patch(color="#8A96A8", label="< 0.8  Undertraining"),
    mpatches.Patch(color="#0EDF6C", label="0.8–1.3  Optimal"),
    mpatches.Patch(color="#FCC547", label="1.3–1.5  Caution"),
    mpatches.Patch(color="#E22007", label="> 1.5  High injury risk"),
]
ax2.legend(handles=legend_patches, fontsize=8, facecolor="#2A2A2A",
           labelcolor="#CCCCCC", framealpha=0.8, loc="upper left")

plt.tight_layout()
plt.savefig("load_tracker_output.png", dpi=150, bbox_inches="tight", facecolor="#1A1A1A")
plt.show()
print("Chart saved to load_tracker_output.png")
