import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
# load data define features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "asample2_withlag.csv"
OUT = ROOT / "outputs" / "shap_importance.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

asample2 = pd.read_csv(DATA)

predictors_all = [col for col in asample2.columns if "Aset" in col or "Bset" in col]

y = asample2["Cm_lhourlywage"]
X = asample2[predictors_all]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=333
)

model_full = xgb.XGBRegressor(
    n_estimators = 90,
    max_depth = 2, 
    learning_rate = 0.3,
    objective = "reg:squarederror",
    random_state = 333,
    verbosity = 0
)

model_full.fit(X_train, y_train)

preds_full = model_full.predict(X_test)
mse_full = mean_squared_error(y_test, preds_full)
r2_full = r2_score(y_test, preds_full)

# compute SHAP vals

explainer = shap.TreeExplainer(model_full)
shap_values = explainer.shap_values(X_train)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index = predictors_all)
shap_importance = shap_importance.sort_values(ascending = False)

shap_importance.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"}).to_csv(
    OUT, index=False
)

print("SHAP values ranking saved to ../outputs/shap_importance.csv")