# TrainingTech - Training Load Analysis

Process raw COROS/Garmin .fit files in Python to quantify training load and visualise performance trends.

## What it does
- Parses .fit files from COROS Training Hub or Garmin Connect using fitdecode
- Calculates session load using TRIMP and sRPE methods
- Plots heart rate with HR zone shading
- Tracks 7-day and 28-day rolling load averages

## Stack
Python . pandas . matplotlib . fitdecode

## Why I built this
This is my project on my journey to learning data science!
Most training platforms only show cardiovascular load. This is the foundation of a mechanical load estimator for endurance and hybrid athletes.

## Getting started
Add your .fit files to the 'sessions' folder.
Don't worry, the folder is gitignored, so your session data stays private.
