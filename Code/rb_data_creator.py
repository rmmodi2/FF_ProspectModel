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

print(pd.ExcelFile("/Users/ronakmodi/FF_ProspectModel/Data/pahowdy_prospect_database.xlsx").sheet_names)

rb_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/pahowdy_prospect_database.xlsx", sheet_name = 'RB', header=[3,4], na_values=["-", 'UDFA', 'NA', '#N/A'])

print(rb_data.columns.tolist())

rb_data = rb_data[rb_data["Unnamed: 6_level_0"]["Draft Year"]!="2021 PROSPECT"]


rb_data.replace(to_replace="UDFA",value=np.nan,inplace=True)
rb_data.replace(to_replace="<2 Yrs of Data",value=np.nan,inplace=True)
rb_data[('Unnamed: 3_level_0', 'DR')].fillna(value=8, inplace=True)
rb_data[('Unnamed: 4_level_0', 'DP')].fillna(value=257, inplace=True)

rb_data[('Unnamed: 3_level_0', 'DR')] = rb_data[('Unnamed: 3_level_0', 'DR')].apply(lambda x: 9-x)
rb_data[('Unnamed: 4_level_0', 'DP')] = rb_data[('Unnamed: 4_level_0', 'DP')].apply(lambda x: 258-x)

rb_data[('Unnamed: 7_level_0', 'Age IN DRAFT YEAR (9/1/dy)')] = rb_data[('Unnamed: 7_level_0', 'Age IN DRAFT YEAR (9/1/dy)')].apply(lambda x: 26-x)

rb_data[('Total Counting Stats', 'Years Played')] = rb_data[('Total Counting Stats', 'Years Played')].apply(lambda x: 6-x)

rb_data[('REC Yards %', 'First')] = rb_data[('REC Yards %', 'First')].multiply(100)
rb_data[('REC Yards %', 'Best')] = rb_data[('REC Yards %', 'Best')].multiply(100)
rb_data[('REC Yards %', 'Last')] = rb_data[('REC Yards %', 'Last')].multiply(100)
rb_data[('REC Yards %', 'AVG')] = rb_data[('REC Yards %', 'AVG')].multiply(100)

rb_data[('Yards/Team Att (Rec and Rush)', 'Rush College Dominator Rating')] = rb_data[('Yards/Team Att (Rec and Rush)', 'Rush College Dominator Rating')].multiply(100)

rb_data[('Total Dominator (Rush and Rec)', 'First')] = rb_data[('Total Dominator (Rush and Rec)', 'First')].multiply(100)
rb_data[('Total Dominator (Rush and Rec)', 'Best')] = rb_data[('Total Dominator (Rush and Rec)', 'Best')].multiply(100)
rb_data[('Total Dominator (Rush and Rec)', 'Last')] = rb_data[('Total Dominator (Rush and Rec)', 'Last')].multiply(100)
rb_data[('Total Dominator (Rush and Rec)', 'AVG')] = rb_data[('Total Dominator (Rush and Rec)', 'AVG')].multiply(100)

rb_data[('TDOa (Total Dominator over average)', 'First')] = rb_data[('TDOa (Total Dominator over average)', 'First')].add(abs(rb_data[('TDOa (Total Dominator over average)', 'First')].min())+.01)
rb_data[('TDOa (Total Dominator over average)', 'First')] = rb_data[('TDOa (Total Dominator over average)', 'First')].multiply(100)
rb_data[('TDOa (Total Dominator over average)', 'Best')] = rb_data[('TDOa (Total Dominator over average)', 'Best')].add(abs(rb_data[('TDOa (Total Dominator over average)', 'Best')].min())+.01)
rb_data[('TDOa (Total Dominator over average)', 'Best')] = rb_data[('TDOa (Total Dominator over average)', 'Best')].multiply(100)
rb_data[('TDOa (Total Dominator over average)', 'Last')] = rb_data[('TDOa (Total Dominator over average)', 'Last')].add(abs(rb_data[('TDOa (Total Dominator over average)', 'Last')].min())+.01)
rb_data[('TDOa (Total Dominator over average)', 'Last')] = rb_data[('TDOa (Total Dominator over average)', 'Last')].multiply(100)
rb_data[('TDOa (Total Dominator over average)', 'AVG')] = rb_data[('TDOa (Total Dominator over average)', 'AVG')].add(abs(rb_data[('TDOa (Total Dominator over average)', 'AVG')].min())+.01)
rb_data[('TDOa (Total Dominator over average)', 'AVG')] = rb_data[('TDOa (Total Dominator over average)', 'AVG')].multiply(100)


rb_data[('Combine', '40 time')] = rb_data[('Combine', '40 time')].apply(lambda x: rb_data[('Combine', '40 time')].max() - x)
rb_data[('Combine', 'Shuttle')] = rb_data[('Combine', 'Shuttle')].apply(lambda x: rb_data[('Combine', 'Shuttle')].max() - x)
rb_data[('Combine', '3 Cone')] = rb_data[('Combine', '3 Cone')].apply(lambda x: rb_data[('Combine', '3 Cone')].max() - x)


rb_data.fillna(rb_data.mean(), inplace=True)

rb_data["Final Year Conference Strength"], rb_data["Final Year Conference Defensive Strength"] = helpers.generate_conference_strength(rb_data)
rb_data["Final Year Conference Strength"] = rb_data["Final Year Conference Strength"].add(abs(rb_data["Final Year Conference Strength"].min())+.1)
rb_data["Final Year Conference Defensive Strength"] = rb_data["Final Year Conference Defensive Strength"].add(abs(rb_data["Final Year Conference Defensive Strength"].min())+.1)


rb_data["Final Year Team Strength"], rb_data["Final Year Team Offensive Strength"] = helpers.generate_team_strength(rb_data)
rb_data["Final Year Team Strength"] = rb_data["Final Year Team Strength"].add(abs(rb_data["Final Year Team Strength"].min())+.1)
rb_data["Final Year Team Offensive Strength"] = rb_data["Final Year Team Offensive Strength"].add(abs(rb_data["Final Year Team Offensive Strength"].min())+.1)

for _, (k, v) in enumerate(helpers.generate_return_data(rb_data).items()):
	rb_data[k] = v

rb_data[("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3")].fillna(value=0, inplace=True)
rb_data[("NFL Career Marks since 2000","# of top  24 finishes")].fillna(value=0, inplace=True)
_, _, _, _, rb_data["hit_within3years"] = helpers.generate_ppr_ppg_by_48(rb_data)

rb_data.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/rb_data.xlsx")





