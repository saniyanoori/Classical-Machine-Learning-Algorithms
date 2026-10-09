y_actual = [100, 150, 200, 250, 300]
y_predicted = [110, 140, 210, 240, 290]

''' 
Calculate mean(y_actual)
Calculate SS_residual (Σ(actual - predicted)²)
Calculate SS_total (Σ(actual - mean)²)
R² = 1 - (SS_residual/SS_total)
'''

'''
R² = 1 - (SS_residual / SS_total)

SS_residual = Σ(y - ŷ)²                    (the model's error)
SS_total    = Σ(y - mean(y))²              (the error if we only predicted the average)
'''

import numpy as np

y_mean = np.mean(y_actual)
SS_residual = np.sum((np.array(y_actual) - np.array(y_predicted))**2)
SS_total = np.sum((np.array(y_actual) - y_mean)**2)
R_squared = 1 - (SS_residual / SS_total)

print(f"R²: {R_squared:.4f}")


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_actual, y_predicted)
mae = mean_absolute_error(y_actual, y_predicted)
mse = mean_squared_error(y_actual, y_predicted)
rmse = np.sqrt(mse)

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")