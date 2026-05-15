# ── InstructoML Configuration ────────────────────────────────────────────────

SESSION_ID   = 42
FOLD         = 3          # PyCaret CV folds (keep low for speed)

# Train/test split slider
DEFAULT_TRAIN_SIZE = 0.75
MIN_TRAIN_SIZE     = 0.50
MAX_TRAIN_SIZE     = 0.90
STEP_TRAIN_SIZE    = 0.05

# Feature engineering
FEATURE_ENGINEERING_DEPTH = 1

# Optuna – fewer trials + early pruning = much faster
OPTUNA_TRIALS    = 20
OPTUNA_CV_FOLDS  = 3       # fewer folds during HPO for speed
OPTUNA_TIMEOUT   = 120     # hard stop at 2 minutes regardless of trials

# Model keys  (display name → internal key)
CLASSIFICATION_MODELS = {
    "Logistic Regression":     "lr",
    "Random Forest":           "rf",
    "Decision Tree":           "dt",
    "K-Nearest Neighbors":     "knn",
    "Support Vector Machine":  "svm",
    "Naive Bayes":             "nb",
    "XGBoost":                 "xgboost",
}

REGRESSION_MODELS = {
    "Linear Regression":         "lr",
    "Random Forest Regressor":   "rf",
    "Decision Tree Regressor":   "dt",
    "KNN Regressor":             "knn",
    "Support Vector Regressor":  "svm",
    "XGBoost Regressor":         "xgboost",
}

# Output paths
EDA_REPORT_PATH = "outputs/eda_report.html"
MODEL_PATH      = "outputs/model.pkl"

# Preprocessing limits
MAX_CARDINALITY   = 50    # drop text cols with more unique values than this
FILL_NUM_STRATEGY = "median"
FILL_CAT_STRATEGY = "most_frequent"

# Minimum samples a class must have to be kept (must be >= FOLD for CV to work)
MIN_CLASS_SAMPLES = FOLD  # classes with fewer rows than this are dropped
