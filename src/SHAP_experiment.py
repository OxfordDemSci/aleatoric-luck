import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "asample2_withlag.csv"
SHAP_FILE = ROOT / "outputs" / "shap_importance.csv"
OUT = ROOT / "outputs" / "shap_ordering_results.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
shap_importance = pd.read_csv(SHAP_FILE)

predictors_all = shap_importance["feature"].tolist()
ordered_features = shap_importance["feature"].tolist()  # already sorted high -> low

y = df["Cm_lhourlywage"]
X = df[predictors_all]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=333
)

results = []

for k in range(1, len(ordered_features) + 1):
    feats_hi = ordered_features[:k]
    feats_lo = ordered_features[-k:]

    for label, feats in [("high_to_low", feats_hi), ("low_to_high", feats_lo)]:
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

        results.append({
            "k": k,
            "direction": label,
            "mse": mean_squared_error(y_test, preds),
            "r2": r2_score(y_test, preds),
        })

results_df = pd.DataFrame(results)
results_df.to_csv(OUT, index=False)

print(f"Saved results to {OUT}")