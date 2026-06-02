import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import logging
import sys
import time 
import os 

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "asample2_withlag.csv"
SHAP_FILE = ROOT / "outputs" / "shap_importance.csv"
OUT = ROOT / "outputs" / "shap_ordering_results.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE = ROOT / "logs" / "shap_experiment.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("shap_ordering")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

logger = setup_logger(LOG_FILE)
logger.info("Starting SHAP ordering exp")

df = pd.read_csv(DATA)
shap_importance = pd.read_csv(SHAP_FILE)

predictors_all = shap_importance["feature"].tolist()
ordered_features = shap_importance["feature"].tolist()  # already sorted high -> low

y = df["Cm_lhourlywage"]
X = df[predictors_all]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=333
)

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
logger.info("Using %d parallel workers", N_JOBS)
logger.info("Number of features: %d", len(ordered_features))

# convert once to numpy to avoid repeated pandas slicing
X_train_np = X_train.to_numpy(copy=False)
X_test_np = X_test.to_numpy(copy=False)
y_train_np = y_train.to_numpy(copy=False)
y_test_np = y_test.to_numpy(copy=False)

feature_to_idx = {feat: i for i, feat in enumerate(predictors_all)}
ordered_idx = np.array([feature_to_idx[f] for f in ordered_features], dtype=int)

def fit_and_score(k: int, idxs: np.ndarray, direction: str) -> dict:
    model = xgb.XGBRegressor(
        n_estimators=90,
        max_depth=2,
        learning_rate=0.3,
        objective="reg:squarederror",
        random_state=333,
        verbosity=0,
        n_jobs = N_JOBS,
        tree_method="hist"
    )
    
    model.fit(X_train_np[:, idxs], y_train_np)
    preds = model.predict(X_test_np[:, idxs])
    
    return {
        "k": k,
        "direction": direction,
        "mse": mean_squared_error(y_test_np, preds),
        "r2": r2_score(y_test_np, preds)
    }

t0 = time.perf_counter()
results = []

try:
    for k in range(1, len(ordered_features) + 1):
        hi_idx = ordered_idx[:k]
        lo_idx = ordered_idx[-k:]

        results.append(fit_and_score(k, hi_idx, "high_to_low"))
        results.append(fit_and_score(k, lo_idx, "low_to_high"))

        if k % 10 == 0 or k == len(ordered_features):
            logger.info("Finished k=%d/%d after %.2fs", k, len(ordered_features), time.perf_counter() - t0)

except Exception:
        logger.exception("Experiment failed")
        raise

results_df = pd.DataFrame(results)
results_df.to_csv(OUT, index=False)

logger.info("Saved final results to %s", OUT)
logger.info("Total runtime: %.2fs", time.perf_counter() - t0)
