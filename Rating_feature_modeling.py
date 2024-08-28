import pandas as pd
import numpy as np

from Metrics import avg_ranked_probability_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV

from drop_first_matches import drop_first_matches

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

def split_data(data, target, seed, test_size, *features):
    
    y_target = data[target]
    
    X_features = data[list(features)]
    
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, random_state=seed, test_size=test_size)
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    
    df_rating = pd.read_csv("data/bundesliga_ratings.csv")
    
    df_rating = drop_first_matches(df_rating, 10)
    
    target = "winner"
    
    X_train, y_train, X_test, y_test = split_data(df_rating, target, 42, 0.2, "home_off", "home_def", "away_off", "away_def")
    
    knn_model = KNeighborsClassifier()
    
    param_dict = {"n_neighbors": np.arange(2, 301)}
    
    avg_rps = make_scorer(avg_ranked_probability_score, needs_proba=True, greater_is_better=False)
    
    grid_search_knn = GridSearchCV(knn_model, param_grid=param_dict, cv=3, scoring=avg_rps, verbose=5)
    
    grid_search_knn.fit(X_train, y_train)
    
    knn_model = grid_search_knn.best_estimator_

    knn_model.fit(X_train, y_train)
    
    y_test_proba_knn = knn_model.predict_proba(X_test)
    
    avg_RPS_knn = avg_ranked_probability_score(y_test, y_test_proba_knn)
    
    log_reg = LogisticRegression()
    
    log_reg.fit(X_train, y_train)
    
    y_test_proba_log_reg = log_reg.predict_proba(X_test)
    
    avg_RPS_log_reg = avg_ranked_probability_score(y_test, y_test_proba_log_reg)
    
    
    xgb_model = XGBClassifier()
    
    param_grid_xgb = {
        "max_depth": np.arange(1,6,1),
        "n_estimators": np.arange(1,1001,1),
        "eta": [0.06],
        "subsample": [0.7, 0.8, 0.9, 1]
    }
    
    grid_search_xgb = GridSearchCV(xgb_model, param_grid=param_grid_xgb, cv=3, verbose=10, scoring=avg_rps)
    
    grid_search_xgb.fit(X_train, y_train)
    
    xgb_model = grid_search_xgb.best_estimator_
    
    
    y_proba_xgb = xgb_model.predict_proba(X_test)
    
    print(f"best knn: {knn_model.get_params()}")
    print(f"RPS KNN: {avg_RPS_knn}")
    print(classification_report(y_test, knn_model.predict(X_test)))
    
    print(f"avg_RPS logistic Regression: {avg_RPS_log_reg}")
    print(classification_report(y_test, log_reg.predict(X_test)))
    
    print(f"best random forest: {xgb_model.get_params()}")
    print(f"XGB RPS: {avg_ranked_probability_score(y_test, y_proba_xgb)}")
    print(classification_report(y_test, xgb_model.predict(X_test)))
    