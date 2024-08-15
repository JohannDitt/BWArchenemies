"""
    Function for the goal prediction model based on rating fetures
"""

import numpy as np
import pandas as pd
from pyswarms.single.global_best import GlobalBestPSO

def expected_goals_home(h_att, a_def, alpha_h, beta_h, gamma_h):
    
    arg = 1 + np.exp(-beta_h*(h_att + a_def) - gamma_h)            
    exp_goals_home = alpha_h/arg
    return exp_goals_home

def expected_goals_away(a_att, h_def, alpha_a, beta_a, gamma_h):
    
    arg = 1+np.exp(-beta_a*(a_att+h_def) - gamma_h)
    exp_goals_away = alpha_a/arg
    return exp_goals_away

def update_home_attack(omega_h_att, h_goals_true, h_goals_exp):
    
    return omega_h_att*(h_goals_true-h_goals_exp)
    

def update_home_defense(omega_h_def, a_goals_true, a_goals_exp):
    
    return omega_h_def*(a_goals_true-a_goals_exp)
    
def update_away_attack(omega_a_att, a_goals_true, a_goals_exp):
    
    return omega_a_att*(a_goals_true-a_goals_exp)
    
def update_away_defense(omege_a_def, h_goals_true, h_goals_exp):
    
    return omege_a_def*(h_goals_true-h_goals_exp)
    
def goals_prediction_error(h_goals_true, h_goals_exp, a_goals_true, a_goals_exp):
    
    return 0.5*(np.square(h_goals_true-h_goals_exp) + np.square(a_goals_true-a_goals_exp))


def total_goal_prediction_error(params=(1, 1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2)):
    
    
    alpha_h = 5#params[:,8]
    alpha_a = 5#params[:,9]
    beta_h = params[:,0]
    beta_a = params[:,1]
    gamma_h = params[:,2]
    gamma_a = params[:,3]
    omega_h_att = params[:,4]
    omega_h_def = params[:,5]
    omega_a_att = params[:,6]
    omega_a_def = params[:,7]
    
    ratings = {team: [0, 0, 0, 0] for team in np.unique(df_germany_train[["home_team", "away_team"]].values)}
    
    total_goal_error = 0
    total_matches = 0
    
    #alpha_h = 5#np.max(df_germany_train.home_goals.values)
    #alpha_a = 5#np.max(df_germany_train.away_goals.values)
    
    for index, row in df_germany_train.iterrows():
        
        home_team = row["home_team"]
        away_team = row["away_team"]
        
        home_score = row["home_goals"]
        away_score = row["away_goals"]
        
        goals_h_hat = expected_goals_home(ratings[home_team][0], ratings[away_team][3], alpha_h, beta_h, gamma_h)
        goals_a_hat = expected_goals_away(ratings[away_team][2], ratings[home_team][1], alpha_a, beta_a, gamma_a)
        
        ratings[home_team][0] += update_home_attack(omega_h_att, home_score, goals_h_hat)
        ratings[home_team][1] += update_home_defense(omega_h_def, away_score, goals_a_hat)
        ratings[away_team][2] += update_away_attack(omega_a_att, away_score, goals_a_hat)
        ratings[away_team][3] += update_away_defense(omega_a_def, home_score, goals_h_hat)
        
        total_goal_error += goals_prediction_error(home_score, goals_h_hat, away_score, goals_a_hat)
        total_matches += 1
    
    loss = total_goal_error/total_matches
    return loss

def team_ratings_data(dataframe, params):
    
    alpha = 5
    beta_h = params[0]
    beta_a = params[1]
    gamma_h = params[2]
    gamma_a = params[3]
    omega_h_att = params[4]
    omega_h_def = params[5]
    omega_a_att = params[6]
    omega_a_def = params[7]
    
    ratings = {team: [0, 0, 0, 0] for team in np.unique(dataframe[["home_team", "away_team"]].values)}
    
    for index, row in dataframe.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = row["home_goals"]
        away_score = row["away_goals"]
        
        h_att = ratings[home_team][0]
        h_def = ratings[home_team][1]
        a_att = ratings[away_team][2]
        a_def = ratings[away_team][3]
        
        dataframe.loc[index, "home_off"] = h_att
        dataframe.loc[index, "home_def"] = h_def
        dataframe.loc[index, "away_off"] = a_att
        dataframe.loc[index, "away_def"] = a_def
        
        exp_home_score = expected_goals_home(h_att, a_def, alpha, beta_h, gamma_h)
        exp_away_score = expected_goals_away(a_att, h_def, alpha, beta_a, gamma_a)
        
        dataframe.loc[index, "home_goals_exp"] = exp_home_score
        dataframe.loc[index, "away_goals_exp"] = exp_away_score
        
        h_att += update_home_attack(omega_h_att, home_score, exp_home_score)
        h_def += update_home_defense(omega_h_def, away_score, exp_away_score)
        a_att += update_away_attack(omega_a_att, away_score, exp_away_score)
        a_def += update_away_defense(omega_a_def, home_score, exp_home_score)
        
        ratings[home_team][0] = h_att
        ratings[home_team][1] = h_def
        ratings[away_team][2] = a_att
        ratings[away_team][3] = a_def

        
    return dataframe, ratings


if __name__ == "__main__":
    
    df_germany_train = pd.read_csv("data/bundesliga.csv")
    df_germany_train = df_germany_train.sort_values(["date"], ignore_index=True)
    
    #df_germany_train = df_germany_train.head(5000)
    
    max_goals = np.max(df_germany_train[["home_goals", "away_goals"]].values)
    
    x_max = np.array([5, 5, 5, 5, 1.5, 1.5, 1.5, 1.5])
    x_min = np.array([0, 0, -5, -5, 0, 0, 0, 0])
    bounds = (x_min, x_max)
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    
    optimizer = GlobalBestPSO(n_particles=50, dimensions=8, options=options, bounds=bounds)
    cost, pos = optimizer.optimize(total_goal_prediction_error, 200)
    
    df_germany_train, team_ratings = team_ratings_data(df_germany_train, pos)
    
    print(df_germany_train.tail(10))
    
    df_germany_train.to_csv("data/bundesliga_ratings.csv")