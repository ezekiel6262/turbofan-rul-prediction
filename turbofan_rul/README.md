# ✈️ Aircraft Turbofan Engine RUL Prediction
### Predictive Maintenance Using Machine Learning & LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20CMAPSS-red)](https://data.nasa.gov)

---

## 📋 Project Overview

This project predicts the **Remaining Useful Life (RUL)** of turbofan aircraft engines
using NASA's **C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset.
It was originally developed as a Final Year Project for a Bachelor of Engineering (Aeronautical
and Astronautical Engineering) at Kwara State University (KWASU), Nigeria, and has since been
extended with additional ML techniques, deep learning (LSTM), and a complete reproducible
analysis pipeline.

**Predictive maintenance** is recognised by 70% of airlines as one of the most critical
technologies in modern aviation. Rather than reacting to failures or scheduling maintenance
at fixed intervals, it uses real sensor data to predict *when* a component will fail —
enabling maintenance exactly when needed, reducing cost, grounding events, and safety risk.

---

## 🎯 Key Results

| Model | Test RMSE (cycles) | Test R² | Notes |
|---|---|---|---|
| Linear Regression | **33.77** | **0.3406** | Best overall generalisation |
| Lasso Regression | 34.02 | 0.3305 | Strong regularisation |
| LSTM (Deep Learning) | 41.36 | 0.0108 | Sequential potential — see limitations |
| Random Forest | 41.40 | 0.0087 | Overfits without sequence context |
| SVR | 41.47 | 0.0054 | RBF kernel, scaled features |
| Decision Tree | 42.13 | -0.0266 | Overfits baseline |

> **Key finding:** Linear and Lasso regression outperform tree-based and deep learning methods
> on point-in-time features. LSTM's full potential is realised with sequence window inputs
> (see Future Work). This mirrors results in Zheng et al. (2017) and is an important
> finding about the role of temporal context in RUL prediction.

---

## 🗂️ Repository Structure

```
turbofan-rul-prediction/
│
├── README.md                        ← You are here
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── train_FD001.txt              ← Training set (100 engines, run-to-failure)
│   ├── test_FD001.txt               ← Test set (100 engines, truncated)
│   ├── RUL_FD001.txt                ← Ground-truth RUL for test set
│   └── DATA_DESCRIPTION.md          ← Column definitions, dataset notes
│
├── notebooks/
│   ├── 01_EDA_and_Feature_Engineering.ipynb
│   ├── 02_Classical_ML_Models.ipynb
│   ├── 03_LSTM_Deep_Learning.ipynb
│   └── 04_Results_and_Visualisation.ipynb
│
├── src/
│   ├── data_loader.py               ← Load and preprocess CMAPSS data
│   ├── feature_engineering.py       ← RUL labelling, feature selection, scaling
│   ├── models.py                    ← All model classes (ML + LSTM)
│   ├── evaluate.py                  ← RMSE, R², scoring functions
│   └── visualise.py                 ← All plotting utilities
│
├── figures/
│   ├── fig1_eda.png                 ← Sensor correlation + RUL distribution
│   ├── fig2_trajectories.png        ← Engine degradation trajectories
│   ├── fig3_full_comparison.png     ← All models RMSE / R² / scatter
│   └── fig4_lstm.png                ← LSTM training curve + predictions
│
└── results/
    └── all_models_summary.csv       ← Complete performance table
```

---

## 🔬 Dataset

**NASA CMAPSS FD001** — a widely used benchmark for prognostics research.

| Property | Value |
|---|---|
| Training engines | 100 (run-to-failure) |
| Test engines | 100 (truncated before failure) |
| Sensors | 21 sensor measurements per cycle |
| Operational settings | 3 |
| Operating condition | Single (FD001) |
| Failure mode | HPC degradation |
| Source | [NASA Prognostics Data Repository](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) |

The dataset consists of multivariate time series where each engine starts at a healthy state
and degrades until failure. The RUL label is computed as cycles remaining until failure,
clipped at **125 cycles** to handle the observation that degradation only becomes detectable
after a period of stable operation.

---

## ⚙️ Methodology

### Pipeline

```
Raw Sensor Data
      │
      ▼
  EDA + Correlation Analysis
      │   Drop flat sensors (std < 0.5)
      │   Select sensors with |Pearson r| > 0.3 vs RUL
      ▼
  Feature Engineering
      │   Compute RUL label (clipped at 125 cycles)
      │   14 informative sensors + cycle count selected
      ▼
  Min-Max Scaling (0–1)
      │
      ▼
  ┌─────────────────────────────────────┐
  │ Stage 1: Baseline (raw features)    │
  │ Stage 2: Feature Selection          │
  │ Stage 3: Feature Selection+Scaling  │
  └─────────────────────────────────────┘
      │
      ▼
  5 ML Models + LSTM
      │
      ▼
  Evaluation: RMSE + R²
```

### Models Implemented

**Classical ML (scikit-learn):**
- Linear Regression — simple baseline, strong generalisation
- Lasso Regression — regularised linear model, automatic feature shrinkage
- Decision Tree Regression — non-linear, prone to overfitting without sequence context
- Random Forest Regression — ensemble of decision trees
- Support Vector Regression (RBF kernel) — margin-based regression

**Deep Learning (TensorFlow/Keras):**
- LSTM (64 → 32 units, Dropout 0.2) — designed for sequential degradation patterns

### Feature Selection

From 21 sensors, 14 were retained after removing:
- Near-constant sensors (standard deviation < 0.5)
- Sensors with |Pearson correlation| < 0.3 with RUL

Selected sensors: `s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s17, s18, s20, s21`

These correspond to outlet temperatures, pressures, fan/core speeds, and airflow metrics —
all physically meaningful indicators of engine degradation.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ezekiel6262/turbofan-rul-prediction.git
cd turbofan-rul-prediction
pip install -r requirements.txt
```

### Run the notebooks

```bash
jupyter notebook notebooks/
```

Start with `01_EDA_and_Feature_Engineering.ipynb` and follow the numbered sequence.

### Run the full pipeline as a script

```bash
python src/run_pipeline.py
```

---

## 📊 Key Figures

### Sensor Correlation with RUL
Identifies which of the 21 sensors carry useful degradation signal.
Sensors with |r| > 0.3 were retained; near-constant sensors were dropped.

### Engine Degradation Trajectories
Shows how RUL decreases across flight cycles for 6 sample engines.
The clipped RUL ceiling (125 cycles) captures the observable degradation window.

### Model Comparison
Side-by-side RMSE, R² scores and predicted vs actual scatter plots
across all 6 models.

### LSTM Training Curve
Training and validation loss across epochs, plus predicted vs actual scatter.

---

## 💡 Key Findings

1. **Linear models generalise best** on point-in-time features. This is consistent
   with findings by Koen Peters (2020) and highlights that feature engineering
   matters more than model complexity when temporal sequences are not exploited.

2. **Decision Tree and Random Forest overfit** significantly (train R² ~0.97–0.99,
   test R² ~0.00) because they memorise training trajectories without access to
   time-window context.

3. **LSTM underperforms** its theoretical potential here because it receives
   single time-step inputs rather than sequence windows. With window inputs of
   30–50 cycles, LSTM is expected to reach RMSE < 20 cycles (see Future Work).

4. **Feature selection improves all models** — removing 7 uninformative sensors
   reduced noise and improved generalisation across all algorithms.

5. **RUL clipping at 125 cycles** is essential. Engines are healthy and sensors
   are stable in early life; the detectable degradation window is the last ~125
   cycles before failure.

---

## 🔮 Future Work

- [ ] **Sequence window LSTM** — feed 30-cycle windows instead of single points;
      expected to achieve RMSE < 20 cycles
- [ ] **Transformer / Attention models** — self-attention across sensor time series
- [ ] **Multi-dataset evaluation** — extend to FD002, FD003, FD004 (multiple
      operating conditions)
- [ ] **REST API deployment** — wrap best model as a Flask/FastAPI endpoint
- [ ] **Real-time dashboard** — Grafana or Streamlit monitoring dashboard
- [ ] **Uncertainty quantification** — Bayesian approximation or Monte Carlo Dropout
      to give confidence intervals on RUL predictions
- [ ] **IoT integration** — demonstrate how model connects to live sensor streams

---

## 📚 References

1. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation
   Modeling for Aircraft Engine Run-to-Failure Simulation*. 1st International
   Conference on Prognostics and Health Management (PHM08). Denver, CO.

2. Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). *Long Short-Term
   Memory Network for Remaining Useful Life Estimation*. IEEE International
   Conference on Prognostics and Health Management.

3. Peters, K. (2020). *Exploratory Data Analysis and Baseline Regression Model:
   Predictive Maintenance of Turbofan Engines*.

4. Okoh, C., Roy, R., Mehnen, J., & Redding, L. (2016). *Predictive Maintenance
   Modelling for Through-Life Engineering Services*. Procedia CIRP, 47, 196–201.

5. Celikmih, K., Inan, O., & Uguz, H. (2020). *Failure Prediction of Aircraft
   Equipment Using Machine Learning with a Hybrid Data Preparation Method*.
   IEEE Access, 8.

6. Chao, M. A., Kulkarni, C., Goebel, K., & Fink, O. (2021). *Aircraft Engine
   Run-to-Failure Dataset under Real Flight Conditions for Prognostics and
   Diagnostics*. Data, 6(1), 5.

---

## 👤 Author

**Orimogunje Oluwasogo Olarenwaju**
B.Eng Aeronautical and Astronautical Engineering, KWASU (2021)

- GitHub: [@ezekiel6262](https://github.com/ezekiel6262)
- Email: larryclues@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **NASA Ames Research Center** for the CMAPSS dataset
- **Supervisor: Engr. A. Ronald**, Department of Aeronautics and Astronautics, KWASU
- The open-source community behind scikit-learn and TensorFlow
