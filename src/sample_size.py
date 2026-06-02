import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "asample2_withlag.csv"
OUT = ROOT / "outputs" / "sample_size.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE = ROOT / "logs" / "sample_size.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

asample2 = pd.read_csv(DATA)

outcome = "Cm_lhourlywage"

predictors = [col for col in asample2.columns if "Aset" in col or "Bset" in col]

X = asample2[predictors]
y = asample2[outcome]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

n_min = 50
n_max = len(X_train)

samp_size = np.unique(
    np.round(np.logspace(np.log10(n_min), np.log10(n_max), 40)).astype(int)
)

results = []
rng = np.random.default_rng(333)
n_repeats = 5


model = xgb.XGBRegressor(
    random_state=333,
    n_estimators=90,
    max_depth=2,
    learning_rate=0.3,
    objective="reg:squarederror",
    verbosity=0,
    n_jobs=1,
)

train_idx = X_train.index.to_numpy()

for s in samp_size:
    for rep in range(n_repeats):
        samp_idx = rng.choice(train_idx, size=s, replace=False)
        X_samp = X_train.loc[samp_idx]
        y_samp = y_train.loc[samp_idx]

        model.fit(X_samp, y_samp)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "n_samples": s,
            "rep": rep,
            "mse": mse,
            "r2": r2
        })

results_df = pd.DataFrame(results)

results_df.to_csv(OUT, index=False)