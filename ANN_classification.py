import pandas as pd
import numpy as np

from Metrics import avg_ranked_probability_score
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt

from drop_first_matches import drop_first_matches
from tensorflow.keras import backend as K
K.clear_session()

import keras
import tensorflow.python.ops as ops

def loss_func_rps(y_true, y_pred):
    
    loss=avg_ranked_probability_score(y_true, y_pred)
    return loss

def get_result(target):
    help_array = np.ones((len(target), 3), int)
    for idx, res in enumerate(target):
        match res:
            case 0:
                help_array[idx] = np.array([1, 0, 0])
            case 1:
                help_array[idx] = np.array([0, 1, 0])
            case 2:
                help_array[idx] = np.array([0, 0, 1])
                 
    target = help_array
    return target

def get_class(y_prob):
    
    y_class = -1
    max_prob = 0
    for idx, y in enumerate(y_prob):
        if y > max_prob:
            max_prob = y
            y_class = idx
    
    return y_class
    
df_data = pd.read_csv("data/bundesliga_ratings.csv")

df_data = drop_first_matches(df_data, 10)
print(len(df_data))

train_data, test_data = train_test_split(df_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)



#print(X_train.shape)

ann_model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation="sigmoid"),
    tf.keras.layers.Dense(3, activation="softmax")
])

opt = tf.keras.optimizers.Adam()

ann_model.compile(optimizer=opt, loss=loss_func_rps, metrics=[tf.keras.metrics.CategoricalAccuracy()]) #tf.keras.losses.CategoricalCrossentropy
print("Compiled!")
X_train = np.array(train_data[["home_off", "home_def", "away_off", "away_def"]])
X_val = np.array(val_data[["home_off", "home_def", "away_off", "away_def"]])
X_test = np.array(test_data[["home_off", "home_def", "away_off", "away_def"]])

y_train = get_result(train_data["winner"].values)
y_val = get_result(val_data["winner"].values)
y_test_class = test_data["winner"].values
y_test = get_result(y_test_class)

training = ann_model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_val, y_val), verbose=1)


y_pred = ann_model.predict(X_test)
print(avg_ranked_probability_score(y_test, y_pred))

y_pred_class = [get_class(y) for y in y_pred]
y_pred_class = np.array(y_pred_class)

print(classification_report(y_test_class, y_pred_class))

ann_model.save("ann_models/classification.keras")


plt.plot(training.history["loss"], label="train", color="blue")
plt.plot(training.history["val_loss"], label="validation", color="red")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("avg_rps")
plt.legend(loc="right")
plt.show()

plt.plot(training.history["categorical_accuracy"], label="train", color="blue")
plt.plot(training.history["val_categorical_accuracy"], label="validation", color="red")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend(loc="right")
plt.show()




#y_proba = ann_model.predict_proba(X_test)
#y_pred = ann_model.predict(X_test)

#print(classification_report(y_test, y_pred))
#print(f"avg RPS ANN: {avg_ranked_probability_score(y_test, y_proba)}")