import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from drop_first_matches import drop_first_matches

from tensorflow.keras import backend as K
K.clear_session()

df_bundesliga = pd.read_csv("data/bundesliga_ratings.csv")
df_bundesliga = drop_first_matches(df_bundesliga, 10)
df_bundesliga["total_goals"] = df_bundesliga["home_goals"] + df_bundesliga["away_goals"]
df_train, df_test = train_test_split(df_bundesliga, test_size=0.2, random_state=42)

target = ["home_goals", "away_goals"]
features = ["home_off", "home_def", "away_off", "away_def"]

X_train = df_train[features].values
X_test = df_test[features].values

y_train = df_train["total_goals"].values
y_test = df_test["total_goals"].values

#y_train = df_train[target].values
#y_test = df_test[target].values

ann_reg = keras.Sequential([
    layers.Dense(4, activation="gelu"),
    layers.Dense(2, activation="sigmoid"),
    layers.Dense(2, activation="gelu"),
    layers.Dense(1, activation="gelu")
])

ann_reg.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])

training = ann_reg.fit(X_train, y_train, epochs=1000, batch_size=256, validation_split=0.2, verbose=1)

plt.plot(training.history["loss"], label="train", color="blue")
plt.plot(training.history["val_loss"], label="validation", color="red")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend(loc="right")
plt.show()

plt.plot(training.history["mean_absolute_error"], label="train", color="blue")
plt.plot(training.history["val_mean_absolute_error"], label="validation", color="red")
plt.title("MAE")
plt.xlabel("Epochs")
plt.ylabel("mae")
plt.legend(loc="right")
plt.show()

plt.plot(training.history["root_mean_squared_error"], label="train", color="blue")
plt.plot(training.history["val_root_mean_squared_error"], label="validation", color="red")
plt.title("RMSE")
plt.xlabel("Epochs")
plt.ylabel("rmse")
plt.legend(loc="right")
plt.show()

y_pred = ann_reg.predict(X_test)

rmse = np.sqrt(np.mean(np.square(y_pred-y_test)))
mae = np.mean(np.abs(y_pred-y_test))
print(rmse)
print(mae)
ann_reg.save("ann_models/regression_total.keras")

"""
y_diff = np.abs(y_pred - y_test)

n = len(y_diff)

right = 0

nearly_right = 0
for res in y_diff:
    
    game_right = True
    game_nearly = True
    
    for score in res:
        if score > 0.5:
            game_right = False
        if score > 1.5:
            game_nearly = False
    
    if game_right:
        right +=1
        nearly_right += 1
    elif game_nearly:
        nearly_right += 1
    
print(round(right/n, 2), round(nearly_right/n,2))

y_class_test = df_test["winner"].values

y_class_pred = []

score = y_pred[:, 0] - y_pred[:, 1]
 
for res in score:
    if res > 0:
        y_class_pred.append(0)
    elif res == 0:
        y_class_pred.append(1)
    elif res < 0:
        y_class_pred.append(2)

y_class_pred = np.array(y_class_pred)

print(np.sum(y_class_pred))

from sklearn.metrics import classification_report

print(classification_report(y_class_test, y_class_pred))
"""