# Dataset Description — NASA CMAPSS FD001

## Source

NASA Ames Research Center  
**Prognostics Data Repository**  
https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6

Original paper:  
> Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling
> for Aircraft Engine Run-to-Failure Simulation*. 1st International Conference on
> Prognostics and Health Management (PHM08). Denver, CO.

---

## File Format

Each file is a space-delimited text file with **no header row**.

| File | Contents |
|---|---|
| `train_FD001.txt` | Training set — full run-to-failure trajectories |
| `test_FD001.txt` | Test set — trajectories cut off before failure |
| `RUL_FD001.txt` | Ground-truth RUL for each test engine at cutoff |

---

## Column Definitions (26 columns)

| Column | Name | Description |
|---|---|---|
| 1 | `engine_id` | Engine unit number (1–100) |
| 2 | `cycle` | Time cycle (flight cycle number) |
| 3 | `op1` | Operational setting 1 |
| 4 | `op2` | Operational setting 2 |
| 5 | `op3` | Operational setting 3 |
| 6 | `s1` | Total temperature at fan inlet (°R) |
| 7 | `s2` | Total temperature at LPC outlet (°R) |
| 8 | `s3` | Total temperature at HPC outlet (°R) |
| 9 | `s4` | Total temperature at LPT outlet (°R) |
| 10 | `s5` | Pressure at fan inlet (psia) |
| 11 | `s6` | Total pressure in bypass-duct (psia) |
| 12 | `s7` | Total pressure at HPC outlet (psia) |
| 13 | `s8` | Physical fan speed (rpm) |
| 14 | `s9` | Physical core speed (rpm) |
| 15 | `s10` | Engine pressure ratio (P50/P2) |
| 16 | `s11` | Static pressure at HPC outlet (psia) |
| 17 | `s12` | Ratio of fuel flow to Ps30 (pps/psi) |
| 18 | `s13` | Corrected fan speed (rpm) |
| 19 | `s14` | Corrected core speed (rpm) |
| 20 | `s15` | Bypass ratio |
| 21 | `s16` | Burner fuel-air ratio |
| 22 | `s17` | Bleed enthalpy |
| 23 | `s18` | Required fan speed (rpm) |
| 24 | `s19` | Required fan conversion speed (rpm) |
| 25 | `s20` | HP turbine cool air flow (lbm/s) |
| 26 | `s21` | LP turbine cool air flow (lbm/s) |

---

## Dataset Characteristics (FD001)

| Property | Value |
|---|---|
| Operating conditions | 1 (sea level) |
| Fault modes | 1 (HPC degradation) |
| Training engines | 100 |
| Test engines | 100 |
| Avg. training cycles per engine | ~206 |
| Min training cycles | ~150 |
| Max training cycles | ~350 |

---

## RUL Labelling

**Training set:** RUL is computed as:

```
RUL[t] = max_cycle[engine] - cycle[t]
```

Then clipped at **125 cycles** to handle the early-life stable period.

**Why clip at 125?**  
In early flight cycles, engines are healthy and sensor readings are stable.
The detectable degradation signal appears only in the last ~125 cycles before failure.
Clipping prevents the model from being penalised for predicting low degradation
when true RUL is very high (and sensors are flat).

**Test set:** Trajectories are cut off before failure. The true RUL at the
cutoff point is given in `RUL_FD001.txt`, one value per test engine.

---

## Selected Sensors (post feature engineering)

After removing flat sensors (std < 0.5) and low-correlation sensors (|r| < 0.3 with RUL),
the following 14 sensors were retained:

`s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s17, s18, s20, s21`

These correspond to outlet temperatures, pressures, fan/core speeds, and turbine airflow —
all physically meaningful indicators of progressive engine degradation.
