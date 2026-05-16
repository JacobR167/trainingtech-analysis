import fitdecode
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#Set filepath to your .fit file - place it in a Sessions/ folder inside the repo and update the path below
import os
filepath = 'session/example_run.fit'

#Check the file exists and can be read
def load_session(filepath):
    """Load a .fit file and return session summary dict."""
    records = []
    with fitdecode.FitReader(str(filepath)) as fit:
        for frame in fit:
            if isinstance(frame, fitdecode.FitDataMessage):
                if frame.name == 'record':
                    row = {f.name: f.value for f in frame.fields if f.value is not None}
                    if row:
                        records.append(row)
    if not records:
        return None 
    
    #Convert to DataFrame and calculate metrics
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    duration_min = (df['timestamp'].max() - df['timestamp'].min()).seconds / 60
    avg_hr = df['heart_rate'].mean() if 'heart_rate' in df.columns else 130
    trimp = duration_min * (avg_hr/185) * 0.64 * (2.718 ** (1.92 * avg_hr/185))

#Return summary dict for this session
    return {
        'date': df['timestamp'].min().date(),
        'duration_min': round(duration_min, 1),
        'avg_hr': round(avg_hr, 0),
        'load_au': round(trimp, 0),
        'file': filepath.name,
    }

sessions_dir = Path('Sessions')
Sessions = [load_session(f) for f in sorted(sessions_dir.glob('*.fit'))]
Sessions = [s for s in Sessions if s]

summary = pd.DataFrame(Sessions).sort_values('date').reset_index(drop=True)
summary['load_7d'] = summary['load_au'].rolling(7, min_periods=1).mean()
summary['load_28d'] = summary['load_au'].rolling(28, min_periods=1).mean()

print(summary[['date', 'duration_min', 'avg_hr', 'load_au']].to_string())

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(summary)), summary['load_au'], color='#534AB7', alpha=0.7, label='Session load')
ax.plot(range(len(summary)), summary['load_7d'], color='#D85A30', alpha=0.7, label='7-day avg')
ax.plot(range(len(summary)), summary['load_28d'], color='#0F6E56', alpha=0.7, label='28-day avg')
ax.set_xticks(range(len(summary)))
ax.set_xticklabels([str(d) for d in summary['date']], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Load (AU)')
ax.set_title('Training load tracker | TrainingTech')
ax.legend()
plt.tight_layout()
plt.show()
