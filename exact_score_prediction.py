import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

import tensorflow as tf

from drop_first_matches import drop_first_matches

data = pd.read_csv("data/bundesliga_ratings.csv")
data = drop_first_matches(data, 10)

data = data[["home_goals", "away_goals", "home_off", "home_def", "away_off", "away_def"]]

goals = []
attacking = []
defense = []

for idx, row in data.iterrows():
    score_h = row["home_goals"]
    att_h = row["home_off"]
    def_a = row["away_def"]
    
    goals.append(score_h)
    attacking.append(att_h)
    defense.append(def_a)
    
    score_a = row["away_goals"]
    att_a = row["away_off"]
    def_h = row["home_def"]
    
    goals.append(score_a)
    attacking.append(att_a)
    defense.append(def_h)


data_regression = pd.DataFrame({"score": goals, "attack": attacking, "defense": defense})

y_target = data[["home_goals", "away_goals"]].values
y_target = y_target[:, 0] + y_target[:, 1]

X_Features = data.drop(columns=["home_goals", "away_goals"]).values

X_train, X_test, y_train, y_test = train_test_split(X_Features, y_target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)

knn_reg = KNeighborsRegressor(n_neighbors=236)

grid_search = GridSearchCV(knn_reg, param_grid={"n_neighbors": np.arange(2,500)}, cv=3, verbose=10, scoring="neg_root_mean_squared_error")
#knn_reg.fit(X_train, y_train)

grid_search.fit(X_train, y_train)
knn_reg = grid_search.best_estimator_

y_pred_knn = knn_reg.predict(X_test)




#xgboost

xgb_reg = XGBRegressor()

param_grid = {
    "max_depth": np.arange(2,11,1),
    "n_estimators": np.arange(2,501,1)
}

grid_search = GridSearchCV(xgb_reg, param_grid=param_grid, cv=3, verbose=10, scoring="neg_root_mean_squared_error")
grid_search.fit(X_train, y_train)

xgb_reg = grid_search.best_estimator_
print()
print("XGBoost")
print(xgb_reg.get_params())

y_pred_xgb = xgb_reg.predict(X_test)
print(root_mean_squared_error(y_test, y_pred_xgb))

print()
print("knn")
print(knn_reg.get_params())

print(root_mean_squared_error(y_test, y_pred_knn))

print()
print("Linear Regression")
print(lin_reg.coef_, lin_reg.intercept_)
print(root_mean_squared_error(y_test, y_pred_lin))

#ann_regressor = tf.keras.Sequential([
    
#])