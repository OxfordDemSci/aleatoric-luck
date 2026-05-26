import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import logging
import sys
import time 

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

t0 = time.perf_counter()
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

logger.info(
    "loaded data: X=%s, y=%s, train=%s, test=%s, n_features=%d",
    X.shape,
    y.shape,
    X_train.shape,
    X_test.shape,
    len(ordered_features),
)

results = []

try:
    for k in range(1, len(ordered_features) + 1):
        k_start = time.perf_counter()

        feats_hi = ordered_features[:k]
        feats_lo = ordered_features[-k:]

        for label, feats in [("high_to_low", feats_hi), ("low_to_high", feats_lo)]:
            fit_start = time.perf_counter()

            model = xgb.XGBRegressor(
                n_estimators=90,
                max_depth=2,
                learning_rate=0.3,
                objective="reg:squarederror",
                random_state=333,
                verbosity=0,
            )
            model.fit(X_train[feats], y_train)
            preds = model.predict(X_test[feats])

            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            results.append({
                "k": k,
                "direction": label,
                "mse": mse,
                "r2": r2,
            })

            logger.info(
                "k=%d | %s | fit+predict=%.2fs | mse=%.6f | r2=%.6f",
                k,
                label,
                time.perf_counter() - fit_start,
                mse,
                r2
            )
        
        pd.DataFrame(results).to_csv(PARTIAL_OUT, index = False)

        logger.info(
            "finished k=%d in %.2fs | saved checkpoint to %s",
            k, 
            time.perf_counter() - k_start,
            PARTIAL_OUT.name
        )
except Exception:
    logger.exception("Experiment failed")
    raise

results_df = pd.DataFrame(results)
results_df.to_csv(OUT, index=False)

if PARTIAL_OUT.exists():
    PARTIAL_OUT.unlink()

logger.info("Saved final results to %s", OUT)
logger.info("Total runtime: %.2fs", time.perf_counter() - t0)
