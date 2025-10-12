"""

Code base for train and save prediction model

"""

import os
import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd


dataset = load_diabetes(as_frame=True)
print(dataset)

X = dataset["frame"].drop(columns=['target'])
y = dataset["frame"]['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
metrics = [
    ['RMSE', f'{rmse:.2f}'],
    ['R2 score', f'{r2:.2f}'],
    ['MAE', f'{mae:.2f}']
]
print('Model performance:')
print(tabulate(metrics, headers=['Metric', 'Value'], tablefmt='grid'))

features_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)
print('Features importance:')
print(tabulate(
            features_importance[['Feature', 'Coefficient']].values,
            headers=['Feature', 'Coefficient'],
            tablefmt='grid',
            floatfmt='.2f'))

fig, axes = plt.subplots(2, 2, figsize=(14,10))

# Scatter plot with lines
axes[0,0].scatter(y_test, y_test_pred, alpha=0.7, edgecolors='k', linewidth=0.5)
z = np.polyfit(y_test, y_test_pred, 1)
p = np.poly1d(z)
axes[0,0].plot(y_test, p(y_test), 'g-', lw=2, label='Fit Line')
axes[0,0].set_xlabel('True values', fontsize=12)
axes[0,0].set_ylabel('Predicted value', fontsize=12)
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0, 0].set_title('Scatter plot True - Predicted Value')

# Residual plot

residuals = y_test - y_test_pred
axes[0,1].scatter(y_test_pred, residuals, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[0,1].axhline(y=0, color='g', linestyle='-', lw=2)
axes[0,1].set_xlabel('Predicted value', fontsize=12)
axes[0,1].set_ylabel('Residual', fontsize=12)
axes[0,1].grid(True, alpha=0.3)
axes[0, 1].set_title('Residual plot')

# Features importances

colors = ['green' if x > 0 else 'red' for x in features_importance['Coefficient']]
axes[1, 0].barh(features_importance['Feature'], features_importance['Coefficient'], color=colors)
axes[1, 0].axvline(x=0, color='black', linestyle='-', lw=0.8)
axes[1, 0].set_xlabel('Coefficient Value', fontsize=12)
axes[1, 0].set_ylabel('Features', fontsize=12)
axes[1,0].set_title('Feature importance')
axes[1,0].grid(True, alpha=0.3, axis='x')

os.makedirs('models', exist_ok=True)
plt.tight_layout()
plt.savefig('models/model_diabetes_v0.1_result.png', dpi=300, bbox_inches='tight')

with open('models/model_diabetes_v0.1.pkl', 'wb') as file:
    pickle.dump(model, file)

print('Save scaler and models to ../models folder')
