# 🤖 InstructoML: AutoML Made Simple

> A web-based Automated Machine Learning platform that lets anyone train, tune, and download ML models — no coding required.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-3.3.2-green)
![Optuna](https://img.shields.io/badge/Optuna-3.6.1-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  What is InstructoML?

InstructoML is a locally hosted AutoML web application built with **Streamlit**. It simplifies the entire supervised machine learning pipeline — from raw CSV upload to a downloadable trained model — through a clean, tab-based interface.

No Python knowledge needed. Just upload your data, click a few buttons, and get a production-ready model.

---

##  Features

-  **Auto Problem Detection** — Automatically detects classification vs regression from your target column
-  **Auto Mode** — Benchmarks all algorithms using PyCaret and ranks them on a leaderboard
-  **Manual Mode** — Select your own algorithm and train with a proper stratified train/test split
-  **Optuna Tuning** — Fast hyperparameter optimization using TPE sampler + MedianPruner (~3× faster than grid search)
-  **Robust Preprocessing** — 10-step pipeline handling missing values, encoding errors, rare classes, constant columns, and more
-  **Automated EDA** — Generates a full HTML exploratory data analysis report
-  **Feature Engineering** — Automated feature synthesis using Featuretools DFS
-  **Model Download** — One-click export of the trained model as a `.pkl` file

---

##  Demo

| Tab | Description |
|-----|-------------|
|  **Data** | Upload CSV, preview data, select target, set train/test split |
|  **EDA & FE** | Generate EDA report, run automated feature engineering |
|  **Training** | Choose Auto or Manual mode, train your model |
|  **Results** | View metrics, tune with Optuna, download model |

---

##  Project Structure

```
InstructoML/
│
├── app.py                        # Main Streamlit application
├── config.py                     # Global configuration constants
├── requirements.txt              # Python dependencies
│
├── utils/
│   ├── loader.py                 # Robust CSV loader (5-encoding fallback)
│   ├── preprocessor.py           # 10-step data cleaning pipeline
│   ├── detect_problem.py         # Auto classification/regression detection
│   ├── metrics.py                # Evaluation metrics renderer
│   ├── EDA.py                    # EDA report generation
│   └── feature_engineering.py   # Featuretools DFS feature synthesis
│
├── models/
│   ├── train.py                  # Auto (PyCaret) + Manual (sklearn) training
│   └── tuner.py                  # Optuna hyperparameter tuning
│
└── outputs/
    ├── eda_report.html           # Generated EDA report
    └── model.pkl                 # Saved trained model
```

---

##  Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kesarsinghchouhan/InstructoML.git
cd InstructoML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

##  Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.35.0 | Web UI framework |
| pycaret | 3.3.2 | Auto model benchmarking |
| scikit-learn | 1.4.2 | ML algorithms & pipelines |
| xgboost | 2.0.3 | Gradient boosting |
| optuna | 3.6.1 | Hyperparameter optimization |
| ydata-profiling | 4.6.4 | EDA report generation |
| featuretools | 1.28.0 | Automated feature engineering |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computation |
| matplotlib | ≥3.7.0 | Visualization (EDA fallback) |

---

## Supported Algorithms

### Classification
| Key | Algorithm |
|-----|-----------|
| `lr` | Logistic Regression |
| `rf` | Random Forest |
| `dt` | Decision Tree |
| `knn` | K-Nearest Neighbors |
| `svm` | Support Vector Machine |
| `nb` | Gaussian Naive Bayes |
| `xgboost` | XGBoost Classifier |

### Regression
| Key | Algorithm |
|-----|-----------|
| `lr` | Linear Regression |
| `rf` | Random Forest Regressor |
| `dt` | Decision Tree Regressor |
| `knn` | KNN Regressor |
| `svm` | Support Vector Regressor |
| `xgboost` | XGBoost Regressor |

---

## Configuration

All settings are in `config.py`:

```python
FOLD             = 3      # CV folds for PyCaret
OPTUNA_TRIALS    = 20     # Max Optuna trials
OPTUNA_TIMEOUT   = 120    # Tuning timeout in seconds
MIN_CLASS_SAMPLES = 3     # Min samples per class (rare class threshold)
MAX_CARDINALITY  = 50     # Max unique values for categorical columns
DEFAULT_TRAIN_SIZE = 0.75 # Default train/test split
```

---

## Performance

Tested on the **California Housing Dataset** (20,640 rows):

| Mode | Algorithm | R² Score | RMSE |
|------|-----------|----------|------|
| Auto (PyCaret) | LightGBM | 0.8251 | 48,344 |
| Manual + Optuna | XGBoost | **0.8305** | 47,125 |

Optuna tuning completed in **under 90 seconds** with MedianPruner active.

---

## Robustness

InstructoML handles real-world data issues automatically:

| Issue | Fix |
|-------|-----|
| Non-UTF-8 CSV files | 5-encoding fallback (UTF-8 → Latin-1 → CP1252 → byte replacement) |
| Class with 1 sample | Rare class removal + dynamic fold reduction (`_safe_fold()`) |
| Missing values | Median imputation (numeric) + mode imputation (categorical) |
| Constant columns | Auto-dropped before training |
| High-cardinality text | Auto-dropped if > 50 unique values |
| Datetime columns | Auto-detected and dropped |

---

##Acknowledgements

- [PyCaret](https://pycaret.org) — Low-code ML framework
- [Optuna](https://optuna.org) — Hyperparameter optimization
- [Streamlit](https://streamlit.io) — Web UI framework
- [Featuretools](https://www.featuretools.com) — Automated feature engineering
- [ydata-profiling](https://github.com/ydataai/ydata-profiling) — EDA reports

---

<p align="center">Built with ❤️ by Kesar Singh Chouhan</p>
