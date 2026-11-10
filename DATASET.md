# Dataset Guide: Condition Monitoring of Hydraulic Systems

This repository uses the **Condition Monitoring of Hydraulic Systems** dataset from the UCI Machine Learning Repository. Raw sensor matrices are excluded from version control due to file size (~175 MB combined for the full archive, ~99 MB for `PS1.txt` and `FS1.txt`).

## 1. Official Source and Citation

- **Dataset Name:** Condition Monitoring of Hydraulic Systems (UCI Dataset ID: 447)
- **Creators:** Nikolai Helwig, Eliseo Pignanelli, and Andreas Schütze (ZeMA gGmbH and Saarland University)
- **Repository URL:** https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
- **DOI:** https://doi.org/10.24432/C5CW21
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### Primary Reference

> Helwig, N., Pignanelli, E., & Schütze, A. (2015). Condition Monitoring of a Complex Hydraulic System Using Multivariate Statistics. *2015 IEEE International Instrumentation and Measurement Technology Conference (I2MTC)*, Pisa, Italy, pp. 210-215. https://doi.org/10.1109/I2MTC.2015.7151267

### Additional Dataset References

> Helwig, N., & Schütze, A. (2015). Detecting and compensating sensor faults in a hydraulic condition monitoring system. *Proceedings of SENSOR 2015 - 17th International Conference on Sensors and Measurement Technology*, Nuremberg, Germany, pp. 641-646. https://doi.org/10.5162/sensor2015/D8.1

> Schneider, T., Helwig, N., & Schütze, A. (2017). Automatic feature extraction and selection for classification of cyclical time series data. *tm - Technisches Messen*, 84(3), 198-206. https://doi.org/10.1515/teme-2016-0072

---

## 2. Required Files for This Project

Although the full UCI archive contains 17 sensor files across 2,205 cycles, the experiments in this repository require only three tab-delimited text files:

| File | Description | Physical Unit | Sampling Rate | Matrix Shape (Rows x Columns) | Approximate Size |
|---|---|---|---:|---|---:|
| `PS1.txt` | Pressure sensor 1 | bar | 100 Hz | 2,205 x 6,000 | ~91.4 MB |
| `FS1.txt` | Volume flow sensor 1 | L/min | 10 Hz | 2,205 x 600 | ~7.6 MB |
| `profile.txt` | Cycle-level component states and stability flag | Categorical / integer codes | 1 row per 60 s cycle | 2,205 x 5 | ~31.2 KB |

Optional documentation files included in the UCI archive (`description.txt` and `documentation.txt`) provide background metadata but are not read by the analysis code.

---

## 3. Directory Setup Instructions

1. Download `condition+monitoring+of+hydraulic+systems.zip` from https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems.
2. Extract `PS1.txt`, `FS1.txt`, and `profile.txt`.
3. Place the three files inside a `data/` directory at the root of this repository:

```text
hydraulic-pump-fault-detection-github/
├── data/
│   ├── PS1.txt
│   ├── FS1.txt
│   └── profile.txt
├── notebooks/
│   └── P10_Hydraulic_DSA_Evaluasi_Risiko.ipynb
...
```

The notebook automatically searches for `PS1.txt`, `FS1.txt`, and `profile.txt` in `./data`, `../data`, or the current working directory.

---

## 4. Target and Operating Condition Encoding (`profile.txt`)

`profile.txt` has no header row and contains five whitespace-delimited columns per operating cycle:

1. **Column 1 (`cooler`): Cooler condition (%):**
   - `3`: Close to total failure (732 cycles)
   - `20`: Reduced efficiency (732 cycles)
   - `100`: Full efficiency (741 cycles)
2. **Column 2 (`valve`): Valve condition (%):**
   - `100`: Optimal switching behavior (1,125 cycles)
   - `90`: Small lag (360 cycles)
   - `80`: Severe lag (360 cycles)
   - `73`: Close to total failure (360 cycles)
3. **Column 3 (`pump_leakage`): Internal pump leakage (Target Variable):**
   - `0`: No leakage / normal (1,221 total cycles; 489 stable cycles)
   - `1`: Weak leakage (492 total cycles; 480 stable cycles)
   - `2`: Severe leakage (492 total cycles; 480 stable cycles)
4. **Column 4 (`accumulator`): Hydraulic accumulator pressure (bar):**
   - `130`: Optimal pressure (599 cycles)
   - `115`: Slightly reduced pressure (399 cycles)
   - `100`: Severely reduced pressure (399 cycles)
   - `90`: Close to total failure (808 cycles)
5. **Column 5 (`stable_flag`): Operating regime flag:**
   - `0`: Stable operating conditions reached (1,449 cycles; used for primary analysis)
   - `1`: Static conditions might not have been reached yet (756 cycles)
