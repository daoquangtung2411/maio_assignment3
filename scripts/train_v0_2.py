# pylint: disable=duplicate-code

"""

Code base for train and save prediction model version 0.2

"""

import os
import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
                            r2_score,
                            root_mean_squared_error,
                            mean_absolute_error,
                            precision_recall_curve,
                            precision_score,
                            recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt



dataset = load_diabetes(as_frame=True)
print(dataset)

X = dataset["frame"].drop(columns=['target'])
y = dataset["frame"]['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
selector = SelectKBest(f_regression, k=5)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)
selected_features = X.columns[selector.get_support()]
print(f'Selected features for prediction: {selected_features}')

model = RandomForestRegressor(
    bootstrap=True,
    max_depth=5,
    max_features='sqrt',
    max_samples=0.7,
    min_samples_leaf=4,
    min_samples_split=2,
    n_estimators=500,
    random_state=42
)
model.fit(X_train_sel, y_train)
y_test_pred = model.predict(X_test_sel)
rmse = root_mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)

y_true_flag = (y_test > np.percentile(y_test, 75)).astype(int)
precisions, recalls, thresholds = precision_recall_curve(y_true_flag, y_test_pred)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_thresh_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_thresh_idx]
y_pred_flag = (y_test_pred > best_thresh).astype(int)
precision = precision_score(y_true_flag, y_pred_flag)
recall = recall_score(y_true_flag, y_pred_flag)

metrics = [
    ['RMSE', f'{rmse:.2f}'],
    ['R2 score', f'{r2:.2f}'],
    ['MAE', f'{mae:.2f}'],
    ['Precision@75', f'{precision:.2f}'],
    ['Recall@75', f'{recall:.2f}'],
]
print('Model performance:')
print(tabulate(metrics, headers=['Metric', 'Value'], tablefmt='grid'))

with open('models/model_v0.2_metrics.txt', 'w', encoding='utf-8') as f:
    f.write('Model version 0.2 metrics (Random Forest Regressor)\n')
    for metric_name, metric_val in metrics:
        f.write(f'{metric_name}: {metric_val}\n')

fig, axes = plt.subplots(2, figsize=(14,10))

# Scatter plot with lines
axes[0].scatter(y_test, y_test_pred, alpha=0.7, edgecolors='k', linewidth=0.5)
z = np.polyfit(y_test, y_test_pred, 1)
p = np.poly1d(z)
axes[0].plot(y_test, p(y_test), 'g-', lw=2, label='Fit Line')
axes[0].set_xlabel('True values', fontsize=12)
axes[0].set_ylabel('Predicted value', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Scatter plot True - Predicted Value')

# Residual plot

residuals = y_test - y_test_pred
axes[1].scatter(y_test_pred, residuals, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[1].axhline(y=0, color='g', linestyle='-', lw=2)
axes[1].set_xlabel('Predicted value', fontsize=12)
axes[1].set_ylabel('Residual', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Residual plot')

os.makedirs('models', exist_ok=True)
plt.tight_layout()
plt.savefig('models/model_diabetes_v0.2_result.png', dpi=300, bbox_inches='tight')

with open('models/scaler_diabetes_v0.2.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('models/model_diabetes_v0.2.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('models/selector_diabetes_v0.2.pkl', 'wb') as file:
    pickle.dump(selector, file)

print('Save scaler, selector and models to ../models folder')
