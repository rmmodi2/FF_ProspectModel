import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import helpers

# import the data from the prospect database, take out prospects who have not played in the nfl yet, take out UDFAs, make sure any NA in PPR PPG become 0.

wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/pahowdy_prospect_database.xlsx", sheet_name = "WR", header=[0,1], na_values=["-"])
print("DEBUG: ###################################")
print(list(wr_data.columns))
print("DEBUG: ###################################")
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!="2021 PROSPECT"]
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!="2020 PROSPECT"]
wr_data = wr_data[wr_data["Unnamed: 4_level_0"]["DP"]!="UDFA"]
wr_data[("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3")].fillna(value=0, inplace=True)


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


# Start with the "independent variables - model stuff and draft capital" , last one is the dependent variable - Avg PPG first 3 years in the NFL.

filtered_cols = wr_data[[('Unnamed: 0_level_0', 'Name'), ('Unnamed: 1_level_0', 'School'), ("Unnamed: 4_level_0",'DP'), ("Unnamed: 6_level_0",'Draft Year'), ("Unnamed: 7_level_0",'Age IN DRAFT YEAR (9/1/dy)'), ("Breakout Ages", ">20%"), ("Breakout Ages", ">30%"), 
("RecYds/TmPatt", "BEST"), ("Dominator", "BEST"), ("MS Yards", "BEST"), ("Context Scores", "PPG Above conference expectation (Last Year)"), ("Context Scores", "TeamMate Score"), 
("Context Scores", "Last S/EX"), ("Combine", "WaSS"), ("Combine", "HaSS"), ("Combine", "BMI"), ("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3"), ("Total Counting Stats", "REC/g"), ("Total Counting Stats", "Years Played")]]
draft_capital_only = wr_data[[('Unnamed: 0_level_0', 'Name'), ("Unnamed: 4_level_0",'DP'), ("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3")]]


# replace the NA values from filtered columns with what we want
# sort by alphabetical names

filtered_cols[("Breakout Ages", ">20%")].fillna(value=35, inplace=True)
filtered_cols[("Breakout Ages", ">30%")].fillna(value=35, inplace=True)
filtered_cols.fillna(filtered_cols.mean(), inplace=True)

 # sort everything by alphabetical name and save to excel files, we will worry about joining manually.
filtered_cols.sort_values(by=[('Unnamed: 0_level_0', 'Name')], inplace=True)
draft_capital_only.sort_values(by=[('Unnamed: 0_level_0', 'Name')], inplace=True)


filtered_cols.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_filtered_columns.xlsx")
draft_capital_only.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_draft_capital_only.xlsx")


print("DEBUG: Successfully created Excel files for WR Model Input and WR Draft Capital Test Input")














