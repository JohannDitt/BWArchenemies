import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error
from drop_first_matches import drop_first_matches


data = pd.read_csv("data/bundesliga_ratings.csv")
data = drop_first_matches(data, 6)

y_home_true = data.home_goals.values
y_away_true = data.home_goals.values

y_total_true = y_home_true + y_away_true

y_home_pred = data.home_goals_exp.values
y_away_pred = data.away_goals_exp.values

y_total_pred = y_home_pred + y_away_pred

rmse_home = root_mean_squared_error(y_home_true, y_home_pred)
rmse_away = root_mean_squared_error(y_away_true, y_away_pred)
rmse_total = root_mean_squared_error(y_total_true, y_total_pred)

print(f"RMSE Home: {rmse_home}")
print(f"RMSE Away: {rmse_away}")
print(f"RMSE Total: {rmse_total}")