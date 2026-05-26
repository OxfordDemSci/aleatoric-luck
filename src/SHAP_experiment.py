import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import logging
import sys
import time 
from joblib import Parallel, delayed
import os 

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "asample2_withlag.csv"
SHAP_FILE = ROOT / "outputs" / "shap_importance.csv"
OUT = ROOT / "outputs" / "shap_ordering_results.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE = ROOT / "logs" / "shap_experiment.log"
PARTIAL_OUT = ROOT / "outputs" / "shap_ordering_results.partial.csv"

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

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
logger.info("Using %d parallel workers", N_JOBS)
logger.info("Number of features: %d", len(ordered_features))

def run_one_k(k: int):
    feats_hi = ordered_features[:k]
    feats_lo = ordered_features[-k:]

    rows = []
    for label, feats in [("high_to_low", feats_hi), ("low_to_high", feats_lo)]:
        model = xgb.XGBRegressor(
                n_estimators=90,
                max_depth=2,
                learning_rate=0.3,
                objective="reg:squarederror",
                random_state=333,
                verbosity=0,
                n_jobs = 1,
            )
        model.fit(X_train[feats], y_train)
        preds = model.predict(X_test[feats])

        rows.append({
            "k": k,
            "direction": label,
            "mse": mean_squared_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        })

        return rows
    
ks = list(range(1, len(ordered_features) + 1))

all_results = []
batch_size = 50

t0 = time.perf_counter()



try:
    for start in range(0, len(ks), batch_size):
        batch = ks[start:start + batch_size]
        logger.info("starting batch %d-%d", batch[0], batch[-1])

        batch_results = Parallel(n_jobs = N_JOBS, backend="loky", verbose=0)(
            delayed(run_one_k)(k) for k in batch
        )

        for rows in batch_results:
            all_results.extend(rows)

        pd.DataFrame(all_results).to_csv(PARTIAL_OUT, index=False)
        logger.info(
            "saved checkpoint with %d rows after %.2fs", 
            len(all_results),
            time.perf_counter() - t0,
        )
        
except Exception:
    logger.exception("Experiment failed")
    raise

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT, index=False)

if PARTIAL_OUT.exists():
    PARTIAL_OUT.unlink()

logger.info("Saved final results to %s", OUT)
logger.info("Total runtime: %.2fs", time.perf_counter() - t0)
