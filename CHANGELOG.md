## v2.0 Tuned Model Comparison

### 🔄 What Changed

- Added Ridge (with added GridSearchCV for hyperparameter tuning) and RandomForestRegressor (with added RandomizedSearchCV for hyperparameter tuning). Random Forest showed most improvement compared to other models, and Random Forest hyperparameter was further tune with expanded search space with Grid Search CV (refer to this [EDA file](https://github.com/daoquangtung2411/maio_assignment3/blob/main/scripts/EDA.ipynb) for more information)

- Introduced feature selection (SelectKBest) with (f_regression, k=6) to keep the top 5 predictive features

- Introduced F1-based high-risk calibration.

- Compared all models against baseline LinearRegression.

- Delta metrics with RMSE, MAE, R²- Added performance metrics (Precision, Recall and Threshold) for the advanced models (Ridge and RandomForestRegressor)

- Final result compare to **version 0.1**:

| Version  | RMSE | R2 | MAE | Precision@75 | Recall@75
| ------------- | ------------- | ------------- | ------------- |------------- | ------------- |
| v0.1  | 53.85 | 0.45 | 42.79
| v0.2  | 52.25 | 0.48 | 42.97 | 0.68 | 0.71

