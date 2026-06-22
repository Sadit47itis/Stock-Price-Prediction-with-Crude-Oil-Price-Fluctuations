# 📈 Netflix Stock Price Prediction with Crude Oil Price Fluctuations

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sadit47itis/Stock-Price-Prediction-with-Crude-Oil-Price-Fluctuations/blob/main/Final_Project.ipynb)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

*A machine learning study investigating whether crude oil price fluctuations carry predictive signal for Netflix (NFLX) stock prices, using classical ML models, technical indicators, and Granger causality analysis.*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Research Question](#-research-question)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Technical Indicators](#-technical-indicators)
- [Models](#-models)
- [Explainability (XAI)](#-explainability-xai)
- [Granger Causality Analysis](#-granger-causality-analysis)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Limitations & Future Work](#-limitations--future-work)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

This project explores the intersection of **energy markets and equity markets** by building machine learning models to predict Netflix (NFLX) stock closing prices, and then investigating whether crude oil price data provides additional predictive power.

The study follows a rigorous pipeline:
1. Data preprocessing and cleaning for both NFLX and crude oil datasets
2. Feature engineering with four canonical technical indicators
3. Multi-model training and evaluation (Linear Regression, SVM, Random Forest)
4. Model explainability using SHAP and LIME
5. Granger causality testing to quantify the temporal relationship between oil and stock prices

---

## ❓ Research Question

> *Do crude oil price fluctuations Granger-cause Netflix stock price movements, and can incorporating oil-price features improve stock price prediction accuracy?*

---

## 📂 Dataset

| Dataset | Source | Format | Description |
|---|---|---|---|
| `NFLX Stock Price.csv` | Historical NFLX data | CSV | Daily OHLCV data for Netflix stock |
| `CrudeOilPrice.csv` | Historical WTI/crude oil data | CSV | Daily OHLCV data for crude oil prices |

**Features in both datasets:**

- `Date` — Trading date
- `Open`, `High`, `Low`, `Close` — Price columns (USD)
- `Adj Close` — Dividend-adjusted closing price
- `Volume` — Number of shares/contracts traded

**Preprocessing steps applied:**
- Removal of stock split rows
- Date parsing and chronological sorting
- Volume and price column type casting (comma-stripped string → float)
- Dropped rows with `NaN` values
- Data filtered from **2010 onwards** (post-indicator convergence)

---

## 🗂️ Project Structure

```
├── Final_Project.ipynb          # Main notebook (full pipeline)
├── NFLX Stock Price.csv         # Netflix historical price data
├── CrudeOilPrice.csv            # Crude oil historical price data
└── README.md
```

---

## 🔬 Methodology

```
Raw Data (NFLX + Oil)
        │
        ▼
  Data Preprocessing
  (cleaning, type casting, date parsing, null removal)
        │
        ▼
  Exploratory Data Analysis
  (distribution plots, correlation heatmaps, outlier detection via IQR)
        │
        ▼
  Feature Engineering
  (lag features: Prev_day, Prev_vol + technical indicators: MA, RSI, Bollinger Bands, MACD)
        │
        ▼
  Multicollinearity Reduction
  (drop features with |corr| > 0.8)
        │
        ▼
  Feature Scaling (StandardScaler)
        │
        ▼
  Model Training & Evaluation
  (Linear Regression | SVM | Random Forest)
        │
        ▼
  XAI Analysis (SHAP + LIME)
        │
        ▼
  Oil Feature Fusion
  (Merge datasets → add oil indicators + lagged oil features)
        │
        ▼
  Granger Causality Testing
  (Bidirectional: Oil→Stock & Stock→Oil, lags 1–7)
```

---

## 📊 Technical Indicators

Four standard technical indicators were computed from **lagged closing prices** (`Prev_day`) to avoid data leakage:

| Indicator | Description | Window |
|---|---|---|
| **MA50 / MA200** | Simple Moving Average — smooths price trends | 50-day / 200-day rolling |
| **RSI** (Relative Strength Index) | Momentum oscillator; overbought (>75) / oversold (<25) thresholds | 14-day (default via `pandas_ta`) |
| **Bollinger Bands** | Volatility bands: Upper, Middle (SMA), Lower | 5-day, 2σ |
| **MACD** | Trend-following momentum; MACD line (12/26 EMA) vs Signal line (9 EMA) | 12 / 26 / 9 |

**Percentage-change features** were also derived at windows of 1d, 2d, 5d, and 10d for all indicator columns, creating a rich multi-horizon feature set.

---

## 🤖 Models

### Model 1 — Linear Regression
- Baseline model; high interpretability
- Fitted on StandardScaler-normalized features
- Evaluated using MAE, MAPE, and R²

### Model 2 — Support Vector Machine (SVR)
- Kernel: `poly`
- Captures non-linear relationships in feature space

### Model 3 — Random Forest Regressor
- `n_estimators=100`, `min_samples_split=10`, `random_state=1`
- Ensemble model; robust to noisy features
- Supports native feature importance ranking

**Train / Test Split:**
- Test set: last **750 trading days** (held out chronologically — no shuffling to preserve temporal integrity)

---

## 🧠 Explainability (XAI)

To ensure model transparency and interpretability, two state-of-the-art XAI frameworks were applied:

### SHAP (SHapley Additive Explanations)
- **Global explanation:** Summary plot showing average absolute SHAP values per feature across all test samples
- **Local explanation:** Force plot illustrating individual prediction breakdown for a single test instance
- `shap.LinearExplainer` for Linear Regression; `shap.TreeExplainer` for Random Forest

### LIME (Local Interpretable Model-Agnostic Explanations)
- Provides locally faithful linear approximations for individual predictions
- Applied to both Linear Regression and Random Forest
- Displays top contributing features for a selected test instance

---

## 📉 Granger Causality Analysis

After identifying Linear Regression as the best-performing model, the relationship between oil prices and stock prices was formally tested using the **Granger Causality Test** (from `statsmodels`).

**Setup:**
- Variables: `Close_x` (NFLX stock close) and `Adj Close_y` (crude oil adjusted close)
- Lags tested: 1 through 7 trading days
- Significance threshold: **p < 0.05**

**Results:**

| Direction | Significant Lags |
|---|---|
| Oil Price → Stock Price | Lags **6, 7** |
| Stock Price → Oil Price | Lags **2, 3, 6** |

> The bidirectional Granger causality suggests a **feedback relationship** between the two markets — neither is purely exogenous. However, predictive power is lag-dependent and not consistently significant across all timeframes.

---

## 📈 Results

| Model | MAE | Avg % Error | R² |
|---|---|---|---|
| Linear Regression | ✅ Lowest | ✅ Lowest | ✅ Highest |
| SVM (poly kernel) | Higher | Higher | Lower |
| Random Forest | Moderate | Moderate | Moderate |

**Best Model: Linear Regression**

Despite its simplicity, Linear Regression outperformed both SVM and Random Forest on all three metrics, likely due to the strong linear relationships observed between features (Open, High, Low, Adj Close) and the target variable (Close).

> Note: Incorporating crude oil features into the Linear Regression model did **not** produce a measurable improvement in prediction accuracy on this dataset, which is consistent with the Granger causality finding that oil's predictive influence is only significant at lag lengths of 6–7 days.

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip

### Clone the Repository

```bash
git clone https://github.com/Sadit47itis/Stock-Price-Prediction-with-Crude-Oil-Price-Fluctuations.git
cd Stock-Price-Prediction-with-Crude-Oil-Price-Fluctuations
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pandas-ta shap lime
pip install numpy==1.23.5   # Required for pandas-ta compatibility
```

> ⚠️ **Note:** This notebook requires running the environment **twice** after install due to a NumPy version conflict with `pandas-ta`. Run all cells once, restart the runtime, then run again.

---

## 🚀 Usage

### Run Locally (Jupyter)

```bash
jupyter notebook Final_Project.ipynb
```

### Run on Google Colab

Click the badge at the top of this README or go to:
[https://colab.research.google.com/github/Sadit47itis/Stock-Price-Prediction-with-Crude-Oil-Price-Fluctuations/blob/main/Final_Project.ipynb](https://colab.research.google.com/github/Sadit47itis/Stock-Price-Prediction-with-Crude-Oil-Price-Fluctuations/blob/main/Final_Project.ipynb)

Ensure `NFLX Stock Price.csv` and `CrudeOilPrice.csv` are uploaded to your Colab session or mounted from Google Drive before running.

---

## 💡 Key Findings

- **Linear Regression dominates** — strong linear relationships between OHLC features mean complex models offer no advantage on this dataset.
- **Oil prices Granger-cause stock prices at lag 6–7**, suggesting that short-term oil shocks take roughly a week to propagate into NFLX valuations.
- **Feature engineering matters** — multi-period percentage-change features and technical indicators built from lagged (not current) prices were essential for a leakage-free pipeline.
- **Multicollinearity pruning** (dropping features with |corr| > 0.8) helped reduce redundancy without sacrificing predictive signal.
- **XAI reveals feature importance** — SHAP and LIME both confirm that near-term lagged price and moving average features dominate predictions, while volume-related features contribute minimally.

---

## ⚠️ Limitations & Future Work

| Limitation | Potential Improvement |
|---|---|
| Only 3 models tested | Add XGBoost, LSTM, or Transformer-based time series models |
| No hyperparameter tuning | Grid search / Bayesian optimization for SVM and RF |
| Oil effect not statistically improving predictions | Try VAR models or direct lag-6/7 oil features as inputs |
| No macro features | Incorporate interest rates, VIX, or sentiment data |
| Single stock (NFLX) | Extend to a portfolio or sector-level study |
| No real-time inference pipeline | Build a Streamlit dashboard for live predictions |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Data Manipulation** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Technical Indicators** | `pandas-ta` |
| **Machine Learning** | `scikit-learn` (LinearRegression, SVR, RandomForestRegressor, StandardScaler) |
| **Statistical Testing** | `statsmodels` (Granger Causality) |
| **Explainability** | `shap`, `lime` |
| **Environment** | Google Colab / Jupyter Notebook |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

Made with 💻 as a final year ML project. Feel free to open issues, fork, or contribute!

---

<div align="center">
<i>⭐ If you found this project useful, consider giving it a star!</i>
</div>