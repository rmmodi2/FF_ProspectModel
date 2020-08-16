import pandas as pd
import numpy as np
import math
import scipy
from scipy import stats
from scipy.special import inv_boxcox
from math import sqrt
from sklearn import datasets, linear_model, preprocessing
from sklearn.linear_model import PoissonRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_poisson_deviance, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import normalize, Normalizer, StandardScaler, FunctionTransformer, MinMaxScaler
import helpers
import matplotlib.pyplot as plt

te_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/pahowdy_prospect_database.xlsx", sheet_name = "TE", header=[0,1], na_values=["-", "UDFA", "NA", "#N/A"])

print(te_data.columns.tolist())

te_data = te_data[te_data["Unnamed: 6_level_0"]["Draft Year"]!="2021 PROSPECT"]

te_data[('Unnamed: 3_level_0', 'DR')].fillna(value=8, inplace=True)
te_data[('Unnamed: 4_level_0', 'DP')].fillna(value=257, inplace=True)

te_data[('Unnamed: 3_level_0', 'DR')] = te_data[('Unnamed: 3_level_0', 'DR')].apply(lambda x: 9-x)
te_data[('Unnamed: 4_level_0', 'DP')] = te_data[('Unnamed: 4_level_0', 'DP')].apply(lambda x: 258-x)

te_data[('Unnamed: 7_level_0', 'Age IN DRAFT YEAR (9/1/dy)')] = te_data[('Unnamed: 7_level_0', 'Age IN DRAFT YEAR (9/1/dy)')].apply(lambda x: 27-x)

te_data[('Total Counting Stats', 'College Dominator Rating')] = te_data[('Total Counting Stats', 'College Dominator Rating')].multiply(100)

te_data["Broke Out 15"] = te_data[('BOA', 'BOA (15%)')].isna().apply(lambda x: int((not x)))
te_data["Broke Out 20"] = te_data[('BOA', 'BOA (20%)')].isna().apply(lambda x: int((not x)))

te_data[('BOA', 'BOA (15%)')].fillna(value=24, inplace=True)
te_data[('BOA', 'BOA (20%)')].fillna(value=24, inplace=True)

te_data[('BOA', 'BOA (15%)')] = te_data[('BOA', 'BOA (15%)')].apply(lambda x: 25-x)
te_data[('BOA', 'BOA (20%)')] = te_data[('BOA', 'BOA (20%)')].apply(lambda x: 25-x)

te_data[('Dominator','First')] = te_data[('Dominator','First')].multiply(100)
te_data[('Dominator','Best')] = te_data[('Dominator','Best')].multiply(100)
te_data[('Dominator','Last')] = te_data[('Dominator','Last')].multiply(100)
te_data[('Dominator','AVG')] = te_data[('Dominator','AVG')].multiply(100)

te_data[('MS Yards', 'First')] = te_data[('MS Yards', 'First')].multiply(100)
te_data[('MS Yards', 'Best')] = te_data[('MS Yards', 'Best')].multiply(100)
te_data[('MS Yards', 'Last')] = te_data[('MS Yards', 'Last')].multiply(100)
te_data[('MS Yards', 'AVG')] = te_data[('MS Yards', 'AVG')].multiply(100)

te_data[('Context Scores', 'YOa (yards Over Average)')] = te_data[('Context Scores', 'YOa (yards Over Average)')].add(abs(te_data[('Context Scores', 'YOa (yards Over Average)')].min())+.01)
te_data[('Context Scores', 'YOa (yards Over Average)')] = te_data[('Context Scores', 'YOa (yards Over Average)')].multiply(100)

te_data[('Context Scores', 'DOa (Dom Over Average)')] = te_data[('Context Scores', 'DOa (Dom Over Average)')].add(abs(te_data[('Context Scores', 'DOa (Dom Over Average)')].min())+.01)
te_data[('Context Scores', 'DOa (Dom Over Average)')] = te_data[('Context Scores', 'DOa (Dom Over Average)')].multiply(100)

te_data[('Context Scores', 'S/EX (Yds Share Over Expectatio)')] = te_data[('Context Scores', 'S/EX (Yds Share Over Expectatio)')].add(abs(te_data[('Context Scores', 'S/EX (Yds Share Over Expectatio)')].min())+.01)
te_data[('Context Scores', 'S/EX (Yds Share Over Expectatio)')] = te_data[('Context Scores', 'S/EX (Yds Share Over Expectatio)')].multiply(100)

te_data[('Combine', '40 time')] = te_data[('Combine', '40 time')].apply(lambda x: te_data[('Combine', '40 time')].max() - x)
te_data[('Combine', 'Shuttle')] = te_data[('Combine', 'Shuttle')].apply(lambda x: te_data[('Combine', 'Shuttle')].max() - x)
te_data[('Combine', '3 Cone')] = te_data[('Combine', '3 Cone')].apply(lambda x: te_data[('Combine', '3 Cone')].max() - x)


te_data["Final Year Conference Strength"], te_data["Final Year Conference Defensive Strength"] = helpers.generate_conference_strength(te_data)
te_data["Final Year Conference Strength"] = te_data["Final Year Conference Strength"].add(abs(te_data["Final Year Conference Strength"].min())+.1)
te_data["Final Year Conference Defensive Strength"] = te_data["Final Year Conference Defensive Strength"].add(abs(te_data["Final Year Conference Defensive Strength"].min())+.1)


te_data["Final Year Team Strength"], te_data["Final Year Team Offensive Strength"] = helpers.generate_team_strength(te_data)
te_data["Final Year Team Strength"] = te_data["Final Year Team Strength"].add(abs(te_data["Final Year Team Strength"].min())+.1)
te_data["Final Year Team Offensive Strength"] = te_data["Final Year Team Offensive Strength"].add(abs(te_data["Final Year Team Offensive Strength"].min())+.1)

te_data.fillna(te_data.mean(), inplace=True)

_, _, te_data["broke_10ppg"], te_data["hit"], te_data["hit_within3years"] = helpers.generate_ppr_ppg_by_48(te_data, qb=True)

te_data.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/te_data.xlsx")





