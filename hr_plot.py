import fitdecode
import pandas as pd
import matplotlib.pyplot as plt

#Confirm this worked and check versions
print("fitdecode version:", fitdecode.__version__)
print("pandas version:", pd.__version__)
print("All imports working - ready to go")

#Place your .fit file in the session/ folder inside the repo and update the path below
import os
filepath = 'session/example_run.fit'

#Check the file exists and can be read (messsage should appear in terminal)
print("FIT file loaded successfully")

with fitdecode.FitReader(filepath) as fitfile:
    for frame in fitfile:
        if isinstance(frame, fitdecode.FitDataMessage) and frame.name == 'record':
            print("Record message found:")
            for field in frame.fields:
                print(f"  {field.name}: {field.value}")

def load_fit_records(filepath):
    "load all 'record' messages from a .fit file into a list of dictionaries, then convert to a DataFrame"
    records = []
    with fitdecode.FitReader(filepath) as fitfile:
        for frame in fitfile:
            if isinstance(frame, fitdecode.FitDataMessage) and frame.name == 'record':
                row = {
                    field.name: field.value
                    for field in frame.fields
                    if field.value is not None
                }
                if row:
                  records.append(row)
    return records

#Load records and calculate metrics
df = pd.DataFrame(load_fit_records(filepath))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['elapsed_min'] = (df['timestamp'] - df['timestamp'].min()).dt.seconds / 60

duration_min = (df['timestamp'].max() - df['timestamp'].min()).seconds / 60
avg_hr = df['heart_rate'].mean()
max_hr_recorded = df['heart_rate'].max()

#Calculate your HR zones based on Heart Rate Reserve (HRR)
hr_max = 199 #Replace with your actual max HR if known, or use 220-age as a rough estimate 
hr_rest = 42 #Replace with your actual resting HR if known
hr_reserve = hr_max - hr_rest
zones = {
    'Zone 1 (<60%)': (df['heart_rate'] < hr_rest + hr_reserve*0.60).sum(),
    'Zone 2 (60-70%)': ((df['heart_rate'] >= hr_rest + hr_reserve*0.60) & (df['heart_rate'] < hr_rest + hr_reserve*0.70)).sum(),
    'Zone 3 (70-80%)': ((df['heart_rate'] >= hr_rest + hr_reserve*0.70) & (df['heart_rate'] < hr_rest + hr_reserve*0.80)).sum(),
    'Zone 4 (80-90%)': ((df['heart_rate'] >= hr_rest + hr_reserve*0.80) & (df['heart_rate'] < hr_rest + hr_reserve*0.90)).sum(),
    'Zone 5 (>90%)': (df['heart_rate'] >= hr_rest + hr_reserve*0.90).sum(),
}

#When you run this, it will prompt you to enter your RPE for the session (1-10). This allows you to calculate sRPE load as well, which is a simple and effective way to quantify training load based on perceived exertion.
rpe = int(input("Enter your RPE for this session (1-10): "))
srpe_load = rpe * duration_min

print(f"\nDuration:  {duration_min:1f} min")
print(f"Avg HR:  {avg_hr:.0f} bpm | Max HR:  {max_hr_recorded:.0f} bpm")
print(f"sRPE load:  {srpe_load:0f} AU")
print("\nHR zone breakdown: ")
for zone, secs in zones.items():
    print(f" {zone}: {secs}s ({secs/60:.1f} min)")

fig, ax = plt.subplots(figsize=(12, 4))

ax.axhspan(0, hr_rest + hr_reserve*0.60, alpha=0.08, color='blue', label='Zone 1')
ax.axhspan(hr_rest + hr_reserve*0.60, hr_rest + hr_reserve*0.70, alpha=0.08, color='green', label='Zone 2')
ax.axhspan(hr_rest + hr_reserve*0.70, hr_rest + hr_reserve*0.80, alpha=0.08, color='yellow', label='Zone 3')
ax.axhspan(hr_rest + hr_reserve*0.80, hr_rest + hr_reserve*0.90, alpha=0.08, color='orange', label='Zone 4')
ax.axhspan(hr_rest + hr_reserve*0.90, 220, alpha=0.08, color='red', label='Zone 5')

ax.plot(df['elapsed_min'], df['heart_rate'],
        color='#D85A30', linewidth=1.2, label='Heart rate')

ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Heart rate (bpm)')
ax.set_title('Time spent in HR zones')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(60, 200)
plt.tight_layout()
plt.show()

from pathlib import Path
output_path = Path(__file__).parent/"hr_chart.png"
plt.savefig(output_path, dpi=150, bbox_inch='tight')
