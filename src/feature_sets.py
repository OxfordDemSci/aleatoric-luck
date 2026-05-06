import numpy as np
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# set seed for reproducibility 
np.random.seed(seed=333)

# read in data
df = pd.read_csv('../data/asample2_withlag.csv')

# construct X and y
y = df["Cm_lhourlywage"]
predictors = [col for col in df.columns if "Aset" in col or "Bset" in col]
X = df[predictors]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=333
)

feature_names = np.array(X_train.columns)

def run_one(k, seed):
    rng = np.random.default_rng(seed)
    cols = rng.choice(feature_names, size=k, replace=False)

    model = XGBRegressor(n_jobs=1,
                        random_state=seed, # for reproducibility
                        n_estimators = 90, 
                        max_depth = 2, 
                        learning_rate = 0.3, 
                        objective = "reg:squarederror",
                        verbosity = 0) 

    X_tr = X_train.loc[:, cols]
    X_te = X_test.loc[:, cols]

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return k, mse, r2

results = [] 

# logspace, picks 20 numbers between 1 and 4000+ (max no of features)
sizes = np.unique(np.logspace(0, np.log2(len(feature_names)), num=20, base=2).astype(int)) 
# and then does 50 random draws of combinations of that many features 
n_draws = 50 

jobs = [
    (k, seed)
    for k in sizes
    for seed in range(n_draws)
]

out = Parallel(n_jobs=-1, batch_size=10)(
    delayed(run_one)(k, seed) for k, seed in jobs
)

results = [{"k": k, "mse": mse, "r2": r2} for k, mse, r2 in out]

results_df = pd.DataFrame(results)

results_df.to_csv("../outputs/feature_sets.csv", index=False)