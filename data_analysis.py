import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

df_bundesliga = pd.read_csv("data/bundesliga.csv")

#total scored goals
n_bins = (df_bundesliga.home_goals+df_bundesliga.away_goals).max()
plt.hist(df_bundesliga.home_goals + df_bundesliga.away_goals, linewidth=2, histtype="step", bins=n_bins, label="Total Goals")
plt.xlabel("goals")
plt.ylabel("counts/goal")
plt.show()

#scored home and away goals
plt.hist(df_bundesliga.home_goals, color="red", histtype="step", label="Home Goals", bins=n_bins, range=(0,n_bins))

plt.hist(df_bundesliga.away_goals, color="blue", histtype="step", label="Visitor Goals", bins=n_bins, range=(0,n_bins))
plt.xlabel("goals")
plt.ylabel("counts/goal")
plt.legend()
plt.show()

#home wins, draws, away wins
df_bundesliga.winner.value_counts().plot(kind="pie", label="")
plt.show()

#matches played by teams
teams_list = list(df_bundesliga.home_team.unique())
games = []

for team in teams_list:
    played = df_bundesliga.home_team.str.contains(team).sum() + df_bundesliga.away_team.str.contains(team).sum()
    games.append(played)

df_teams = pd.DataFrame({"teams": teams_list, "nr_of_matches": games})
df_teams = df_teams.sort_values(["nr_of_matches"], ascending=False).head(20)    

plt.barh(df_teams.teams, df_teams.nr_of_matches)
plt.show()