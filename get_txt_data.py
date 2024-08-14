import pandas as pd
import numpy as np
import datetime as dt
import re

import glob

from results import get_score

def check_string(x):

    if "=" in x:
        return "season"
    elif "[" in x and "]" in x:
        return "day"
    elif "matchday" in x.lower() or "spieltag" in x.lower():
        return "matchday"
    else:
        try:
            float(x.strip()[:5])
            return "matchTime"
        except:
            return "match"

def get_information(x):
    
    result = re.search("\d-\d", x).group()
    result_center = result.index("-") + x.index(result)
    res = x[result_center-2:result_center+3].strip()
    
    home = x[:result_center-2].strip()
    
    away = ""
    if ")" in x[result_center:result_center+10]:
       away = x[x.index(")")+1:].strip()
    else:
        away = x[result_center+3:].strip()
    
    return home, away, res

def get_data(filename):
    data_file = open(filename, "r")

    data_lines = []

    data_lines = data_file.read().splitlines()
    data_file.close()

    while "" in data_lines:
        data_lines.remove("")

    years = data_lines[0][-7:].split("/")
    years[0] = int(years[0])
    years[1] = years[0]+1

    print(years)

    data_lines = data_lines[1:]

    matchday = 1
    date = 0
    format = "%a %b/%d %Y"
    format_alt = "%a %d %m %Y"
    first_half = {"Jul", "Aug", "Sep", "Okt", "Oct", "Nov", "Dec", "Dez"}
    home_team = ""
    away_team = ""
    result = ""

    data_home = []
    data_away = []
    data_matchday =[]
    data_date = []
    data_result =[]

    weekday_translater = {
        "Fr": "Fri",
        "Sa": "Sat",
        "So": "Sun",
        "Mo": "Mon",
        "Di": "Tue",
        "Mi": "Wed",
        "Do": "Thu"
    }

    for x in data_lines:
    
        x = x.strip()
        if check_string(x) == "matchday":
            matchday = int(x[-2:].strip())
        elif check_string(x) == "day":
            try:
                datestring = x[1:-1]
       
                if datestring[4:7] in first_half:
                    datestring = datestring + " " + str(years[0])
                else:
                    datestring = datestring + " " + str(years[1])
                date = dt.datetime.strptime(datestring, format)
            except:
                datestring = x[1:-1]
                components = datestring.split(".")
                components.remove("")
                if int(components[-1]) > 6:
                    components.append(str(years[0]))
                else:
                    components.append(str(years[1]))
                
                if len(components) == 4:
                    components[0] = weekday_translater.get(components[0])
                    components = [x.strip().zfill(2) for x in components]
                else:
                    components[1] = components[1].zfill(2)
                    components[0] = components[0].split(" ")
                    components[0][-1] = components[0][-1].zfill(2)
                    components[0][0] = weekday_translater.get(components[0][0])
                    components[0] = " ".join(components[0])
                
                datestring = " ".join(components)
                date = dt.datetime.strptime(datestring, format_alt)

        elif check_string(x) == "matchTime":
            x = x[6:].strip()
            home_team, away_team, result = get_information(x)
            data_matchday.append(matchday)
            data_date.append(date)
            data_home.append(home_team)
            data_away.append(away_team)
            data_result.append(result)  
        elif check_string(x) == "match":
            home_team, away_team, result = get_information(x)
            data_matchday.append(matchday)
            data_date.append(date)
            data_home.append(home_team)
            data_away.append(away_team)
            data_result.append(result)  
    

    dataframe = pd.DataFrame({"matchday": data_matchday, "date": data_date, "home_team": data_home, "final_result": data_result, "away_team": data_away})
    
    return dataframe

def check_winner(score):
    if score == 0:
        return 1 #draw, nobody wins
    if score > 0:
        return 0 #home team wins
    if score < 0:
        return 2 #visiting team wins


files_old = glob.glob("data/deutschland/archive/*/*/1-bundesliga.txt")
files_new = glob.glob("data/deutschland/*/1-bundesliga.txt")
files_first = glob.glob("data/deutschland/*/1-bundesliga-i.txt")
files_second = glob.glob("data/deutschland/*/1-bundesliga-ii.txt")
files = files_old+files_new+files_first+files_second
files.remove("data/deutschland/2024-25/1-bundesliga.txt")

data_bundesliga = [get_data(filename) for filename in files]

data = pd.concat(data_bundesliga, ignore_index=True)
data["home_goals"] = data.final_result.apply(lambda x: get_score(x, True))
data["away_goals"] = data.final_result.apply(lambda x: get_score(x, False))
data["score"] = data.home_goals - data.away_goals
data["winner"] = data.score.apply(lambda x :check_winner(x))
data = data.drop(columns="score")
data = data.sort_values(["date"], ignore_index=True)

teams = {"offenheim": "TSG 1899 Hoffenheim",
        "TSG Hoffenheim": "TSG 1899 Hoffenheim",
        "Bayern München       # Bayern Deutscher Meister 2019/2020": "Bayern München",
        "Blau-Weiß 90 Berlin (-1992)": "Blau-Weiß 90 Berlin",
        "SC Tasmania 1900 Berlin (-1973)": "SC Tasmania 1900 Berlin",
        "Bayer Leverkusen": "Bayer 04 Leverkusen"}
data = data.replace(teams)

print(data.tail(10))

data.to_csv("data/bundesliga.csv")