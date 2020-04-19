import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import helpers

# import the data from the prospect database, take out prospects who have not played in the nfl yet, take out UDFAs.

wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/pahowdy_prospect_database.xlsx", sheet_name = "WR", header=[0,1], na_values=["-"])
print("DEBUG: ###################################")
print(list(wr_data.columns))
print("DEBUG: ###################################")
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!="2021 PROSPECT"]
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!="2020 PROSPECT"]
wr_data = wr_data[wr_data["Unnamed: 4_level_0"]["DP"]!="UDFA"]

# import the data for players final year to get the end date for player's careers

players_final_year = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/player_final_year.xlsx", sheet_name = "main")
print("DEBUG: ###################################")
print(list(players_final_year.columns))
print("DEBUG: ###################################")


"""
Now we create the X and Y matrices for the data
"""

"""
Current X Matrix: [Draft Capital - log(DP) ; Breakout Age_30%, Breakout Age_20%, BEST RecYds/TmPatt, BEST Dominator, PPG Above Conf Expectation, TeamMate Score,
Weight Adjusted Speed Score, BMI]

!!!
TODO: left out: Height Adjusted Speed Score, BEST MS Yards, Total College Dominator, Total Yards Dominator, YPR, Rec/g, AVG PPG, rec, yards
!!!
"""

# Create X and Y


# Start with the "independent variables - model stuff and draft capital"

filtered_cols = wr_data[[('Unnamed: 0_level_0', 'Name'), ('Unnamed: 1_level_0', 'School'), ("Unnamed: 4_level_0",'DP'), ("Breakout Ages", ">20%"), ("Breakout Ages", ">30%"), 
("RecYds/TmPatt", "BEST"), ("Dominator", "BEST"), ("Context Scores", "PPG Above conference expectation (Last Year)"), ("Context Scores", "TeamMate Score"), 
("Combine", "WaSS"), ("Combine", "BMI")]]
draft_capital_only = wr_data[[('Unnamed: 0_level_0', 'Name'), ("Unnamed: 4_level_0",'DP')]]

# Dependent Variable is the score

finish_cols = wr_data[[('Unnamed: 0_level_0', 'Name'), ("Unnamed: 4_level_0",'DP'), ("Unnamed: 6_level_0",'Draft Year'), ("NFL Career Marks since 2000", "# of top 5  finishes"), ("NFL Career Marks since 2000", "# of top  12 finishes"), 
("NFL Career Marks since 2000", "# of top  24 finishes"), ("NFL Career Marks since 2000", "# of top  36 finishes")]]
finish_cols.fillna("0", inplace=True)

# we are going to create a dataframe with the name and score

name_and_score = []

for index,row in finish_cols.iterrows():
	real_top_36 = int(row["NFL Career Marks since 2000"]["# of top  36 finishes"]) - int(row["NFL Career Marks since 2000"]["# of top  24 finishes"])
	real_top_24 = int(row["NFL Career Marks since 2000"]["# of top  24 finishes"]) - int(row["NFL Career Marks since 2000"]["# of top  12 finishes"])
	real_top_12 = int(row["NFL Career Marks since 2000"]["# of top  12 finishes"]) - int(row["NFL Career Marks since 2000"]["# of top 5  finishes"])
	real_top_5 = int(row["NFL Career Marks since 2000"]["# of top 5  finishes"])
	score = real_top_36 + 2*real_top_24 + 4*real_top_12 + 8*real_top_5
	print("DEBUG: score for %s is %s" % (row['Unnamed: 0_level_0']['Name'], score))
	total_years = helpers.get_total_years(row, players_final_year)
	score = score/total_years
	name_and_score.append([row['Unnamed: 0_level_0']['Name'], score])

name_and_score = pd.DataFrame(name_and_score, columns=["Name", "Score"])

# sort by alphabetical names



print("DEBUG: ###################################")
print(name_and_score)
print("DEBUG: ###################################")

# replace the NA values from filtered columns with the mean from the group of data
# sort by alphabetical names
# !!!
# TODO: alter this to "last seen" ?
# !!!

filtered_cols.fillna(filtered_cols.mean(), inplace=True)

 # sort everything by alphabetical name and save to excel files, we will worry about joining manually.
name_and_score.sort_values(by=["Name"], inplace=True)
filtered_cols.sort_values(by=[('Unnamed: 0_level_0', 'Name')], inplace=True)
draft_capital_only.sort_values(by=[('Unnamed: 0_level_0', 'Name')], inplace=True)


name_and_score.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_name_and_score.xlsx")
filtered_cols.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_filtered_columns.xlsx")
draft_capital_only.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_draft_capital_only.xlsx")


print("DEBUG: Successfully created Excel files for WR Model Input and WR Draft Capital Test Input")














