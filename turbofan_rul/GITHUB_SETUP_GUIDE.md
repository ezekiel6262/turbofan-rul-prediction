# GitHub Setup & Publication Guide
## turbofan-rul-prediction

Step-by-step instructions to publish your project to GitHub,
set up the repository correctly, and prepare for journal submission.

---

## PART 1 — Create the GitHub Repository

### Step 1: Create account (if you don't have one)
Go to https://github.com and sign up with your email (larryclues@gmail.com).
Username: **ezekiel6262** (you already have this)

### Step 2: Create new repository
1. Click the **+** icon (top right) → **New repository**
2. Repository name: `turbofan-rul-prediction`
3. Description: `Predictive maintenance for aircraft turbofan engines — RUL prediction using ML & LSTM on NASA CMAPSS dataset`
4. Set to **Public**
5. **DO NOT** tick "Add README" (we already have one)
6. Click **Create repository**

---

## PART 2 — Upload Your Files

### Option A: Upload via GitHub website (easiest)

1. Open your new empty repository on GitHub
2. Click **uploading an existing file** (shown on the empty repo page)
3. Drag and drop ALL files from the zip you downloaded
4. Keep the folder structure exactly as shown below
5. Commit message: `Initial commit: Complete RUL prediction project`
6. Click **Commit changes**

### Option B: Upload via Git command line

```bash
# 1. Install git if not installed
# Windows: https://git-scm.com/download/win
# Mac: brew install git
# Linux: sudo apt install git

# 2. Configure git (one time)
git config --global user.name "Orimogunje Oluwasogo"
git config --global user.email "larryclues@gmail.com"

# 3. Navigate to your project folder
cd turbofan-rul-prediction

# 4. Initialise and push
git init
git add .
git commit -m "Initial commit: Complete RUL prediction project with ML and LSTM"
git branch -M main
git remote add origin https://github.com/ezekiel6262/turbofan-rul-prediction.git
git push -u origin main
```

---

## PART 3 — Configure the Repository

### Add topics (makes it discoverable)
1. Go to your repo page on GitHub
2. Click the gear icon next to **About** (top right)
3. Add these topics:
   ```
   predictive-maintenance
   machine-learning
   deep-learning
   lstm
   aerospace
   turbofan
   remaining-useful-life
   nasa-cmapss
   python
   jupyter-notebook
   ```
4. Add website: (leave blank for now)
5. Click **Save changes**

### Pin the repository
1. Go to your GitHub profile page (github.com/ezekiel6262)
2. Click **Customize your profile**
3. Pin `turbofan-rul-prediction` as a featured repository

---

## PART 4 — GitHub Profile README (Important for employers)

Create a special README that appears on your GitHub profile page.

1. Create new repository named exactly: **ezekiel6262**
   (same as your username — this is GitHub's special profile repo)
2. Add README.md with this content:

```markdown
# Hi, I'm Zek 👋

Aerospace Engineer × AI/ML Developer × Web3 Builder

## What I build
- 🛩️ **Predictive Maintenance** — RUL prediction for aircraft turbofan engines (NASA CMAPSS)
- 🔗 **Web3 Infrastructure** — On-chain invoicing (Proof Slip, deployed on Polygon)
- 📊 **Risk Analytics** — DeFi position risk calculator for Extended Protocol (Starknet)

## Tech Stack
`Python` `TensorFlow` `scikit-learn` `Next.js` `Solidity` `AWS` `TypeScript`

## Featured Projects
- [✈️ Turbofan RUL Prediction](https://github.com/ezekiel6262/turbofan-rul-prediction)
  — ML + LSTM predictive maintenance on NASA CMAPSS dataset
- [🧾 Proof Slip](https://github.com/ezekiel6262/proof-slip)
  — On-chain invoicing infrastructure (Ethereum, Base, Polygon, Solana)
- [📉 Extended Protocol Risk Tools](https://github.com/ezekiel6262/RiskCalculator-Extended)
  — Position risk calculator + whale monitor for Starknet DEX

## Currently
🎓 Pursuing Masters (MIT/Agribusiness) in New Zealand — 2027
📚 Building: AWS ML Specialty | TensorFlow Developer Certificate

📫 larryclues@gmail.com
```

---

## PART 5 — For Academic Publication

Your FYP is strong enough to submit to:

### Option 1: IEEE Xplore (Conference Paper — recommended first step)
Target conference: **IEEE International Conference on Prognostics and Health Management (PHM)**
- Annual conference, deadline typically October–November for the following year
- Website: https://phmsociety.org/conference/annual-phm-society-conference/
- Your paper scope: "Comparative Study of ML and LSTM for Aircraft Engine RUL
  Prediction Using NASA CMAPSS FD001"
- Length: 8–10 pages IEEE format

### Option 2: Mendeley Data / Zenodo (Dataset + Code Publication)
Publish your dataset and notebook as a citable research object:
1. Go to https://zenodo.org
2. Create account → New Upload
3. Upload: notebook + figures + results CSV
4. Add metadata:
   - Title: "Turbofan Engine RUL Prediction using ML and LSTM — NASA CMAPSS FD001"
   - Authors: Orimogunje Oluwasogo Olarenwaju
   - Keywords: predictive maintenance, remaining useful life, LSTM, CMAPSS
5. Zenodo gives you a **DOI** — a permanent citation link
6. Add this DOI to your GitHub README

### Option 3: MDPI Aerospace (Open Access Journal)
- Free to read, relatively accessible for first publications
- Website: https://www.mdpi.com/journal/aerospace
- Section: "Aeronautics and Astronautics"
- Article type: "Article" (~5,000–8,000 words)
- Processing fee: ~1,800 CHF (check for waivers for authors from developing countries)
- MDPI has a waiver program — apply as a Nigerian author

### Option 4: arXiv (Preprint — free, immediate, citable)
The fastest way to get your work citable RIGHT NOW:
1. Go to https://arxiv.org
2. Create account
3. Submit to: cs.LG (Machine Learning) + eess.SP (Signal Processing)
4. Upload your PDF
5. You get an arXiv ID within 1–2 business days
6. Add arXiv link to your GitHub README

**Recommended sequence:**
arXiv (immediate) → Zenodo DOI → PHM conference → journal paper

---

## PART 6 — Paper Draft Outline

If you want to write this up formally, here is the structure:

```
Title: Predictive Maintenance for Aircraft Turbofan Engines:
       A Comparative Study of Machine Learning Algorithms and
       LSTM for Remaining Useful Life Prediction

Abstract (250 words)
  - Problem: aircraft maintenance cost and safety
  - Method: 5 ML algorithms + LSTM on NASA CMAPSS FD001
  - Results: LR best generalisation (RMSE=33.77, R²=0.34)
  - Conclusion: feature engineering > model complexity for point-in-time inputs

1. Introduction
   - Predictive vs preventive vs reactive maintenance
   - Importance in aerospace
   - RUL definition
   - Paper contributions

2. Related Work
   - RCM history (Boeing 1960s)
   - CMAPSS benchmark papers (Saxena 2008, Zheng 2017)
   - ML vs deep learning approaches

3. Dataset and Preprocessing
   - CMAPSS FD001 description
   - RUL labelling and clipping rationale
   - Sensor correlation analysis
   - Feature selection (14 sensors retained)

4. Methodology
   - Pipeline diagram
   - Linear Regression
   - Lasso Regression
   - Decision Tree
   - Random Forest
   - SVR
   - LSTM architecture

5. Results
   - Table of all model results
   - Figure: RMSE and R² comparison
   - Figure: Predicted vs Actual scatter plots
   - Figure: LSTM training curve

6. Discussion
   - Why LR outperforms complex models (point-in-time inputs)
   - Overfitting analysis (DT/RF gap)
   - LSTM limitations and expected improvement with windows

7. Conclusion and Future Work

References
```

---

## PART 7 — LinkedIn Post (Announce your project)

Post this on LinkedIn to build visibility:

```
🛩️ Just published my aerospace ML project on GitHub.

During my final year at KWASU (2021), I built a predictive maintenance
system for aircraft turbofan engines — predicting Remaining Useful Life (RUL)
using NASA's CMAPSS dataset.

I recently rebuilt it from scratch, added:
✅ Full reproducible pipeline
✅ 5 ML models (Linear Regression, Lasso, SVR, Decision Tree, Random Forest)
✅ LSTM deep learning model
✅ Complete analysis notebook with visualisations
✅ Feature correlation analysis (14 of 21 sensors retained)

Best result: Linear Regression — Test RMSE = 33.77 cycles, R² = 0.34

Key finding: Feature engineering matters more than model complexity
when temporal sequences aren't exploited. LSTM needs sequence windows
to unlock its potential (next step).

🔗 GitHub: https://github.com/ezekiel6262/turbofan-rul-prediction

#MachineLearning #PredictiveMaintenance #Aerospace #DeepLearning #Python
```
