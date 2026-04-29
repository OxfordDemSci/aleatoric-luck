import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    random_state=seed,
)
model.fit(X_train_subsample, y_train_subsample,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50)])