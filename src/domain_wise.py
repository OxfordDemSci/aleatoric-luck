import numpy as np
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pathlib import Path

# set seed for reproducibility 
np.random.seed(seed=333)

# read in data
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "asample2_withlag.csv"
OUT = ROOT / "outputs" / "feature_sets.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

# construct X and y

y = df["Cm_lhourlywage"]
Aset1 = [col for col in df.columns if "Aset1" in col]
Aset2 = [col for col in df.columns if "Aset2" in col]
Bset1 = [col for col in df.columns if "Bset1" in col]
Bset2 = [col for col in df.columns if "Bset2" in col]

data_sets = {
    "Aset1": Aset1,
    "Aset2": Aset2,
    "Bset1": Bset1,
    "Bset2": Bset2,
}

all_results = []

for data_name, data_cols in data_sets.values():

    X = df[data_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=333
    )

    feature_names = np.array(X_train.columns)
    breakpoint()

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

        return k, seed, mse, r2

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

    results = [
        {
            "data": data_name,
            "k": k,
            "seed": seed,
            "mse": mse,
            "r2": r2,

        }
        for k, seed, mse, r2 in out
    ]

    all_results.extend(results)

results_df = pd.DataFrame(results)

results_df.to_csv(OUT, index=False)