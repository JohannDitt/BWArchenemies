import pandas as pd
import numpy as np

def drop_first_matches(data, n_drop=6):
    
    teams = np.unique(data[["home_team", "away_team"]].values)
    
    team_matches = [drop_first_team_matches(data, team, n_drop) for team in teams]
    team_matches.append(data)
    data_cutted = pd.concat(team_matches, ignore_index=True)
    data_cutted = data_cutted.drop_duplicates(keep=False)
    
    return data_cutted


def drop_first_team_matches(data, team, n_drop=6):
    
    team_matches = data[ (data["home_team"] == team) | (data["away_team"] == team)]

    first_team_matches = team_matches.sort_values(["date"]).iloc[:n_drop , :]
    
    return first_team_matches

if __name__=="__main__":
    
    df = pd.read_csv("data/bundesliga_ratings.csv")
    df = df.drop_duplicates()
    print(df.shape)
    
    team_list = np.unique(df[["home_team", "away_team"]].values)
    df_cutted = drop_first_matches(df, team_list, 10)
    
    print(df_cutted.shape)