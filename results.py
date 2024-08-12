import pandas as pd
import numpy as np
import datetime

import glob

df_germany = pd.read_csv("data/deutschland-master/2010s/2010-11/de.1.csv")

def str_to_datetime(datestring):
    """Function to change the date column intoi datetime format

    Args:
        datestring (string): string containing the date (DayOfWeek Month DayOfMonth Year)

    Returns:
        datetime: date in datetimeformat (yyyy-mm-dd)
    """
    format = "%a %b %d %Y"

    date = datetime.datetime.strptime(datestring, format)
        
    return date

def get_score(results_string, home_team):
    """Function to get the single scored goals 

    Args:
        results_string (string): string containing the result (home-away)
        home_team (bool): boolean whether score for home or away team is desired

    Returns:
        int: individual team score
    """
    try:    
        results_list = results_string.split("-")
    
        if home_team:
            return int(results_list[0])
        else: 
            return int(results_list[-1])
    
    except:
        print(results_string)
        return 0

def clean_dataset(dataset):
    """function to clean a dataframe and bring it to a usable format

    Args:
        dataset (Dataframe): Dataset which should be cleaned

    Returns:
        Dataframe: Cleaned Dataset
    """
    dataset = dataset.dropna()
    dataset.Date = dataset.Date.apply(lambda x: str_to_datetime(x))
    dataset = dataset.rename(columns={"Team 1": "home_team", "Team 2": "away_team", "FT": "final_result", "Date": "date", "Matchday": "matchday"})
    dataset["home_goals"] = dataset.final_result.apply(lambda x: get_score(x, True))
    dataset["away_goals"] = dataset.final_result.apply(lambda x: get_score(x, False))
    
    try:
        dataset = dataset.drop(colums=["HT"])
    except:
        pass
    
    #dataset = dataset["date", "home_team", "home_goals", "away_team", "away_goals"]
    
    return dataset


if __name__=="__main__":
    dataset_list = glob.glob("data/deutschland-master/*/*/*1.csv")

    print(dataset_list)

    datalist = [clean_dataset(pd.read_csv(dataset)) for dataset in dataset_list]

    df_germany = pd.concat(datalist, ignore_index=True)

    print(df_germany.info())

    print(df_germany.Round.unique())