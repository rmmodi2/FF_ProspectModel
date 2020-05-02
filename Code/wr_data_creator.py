import pandas as pd
import numpy as np
import math
from math import sqrt
from sklearn import datasets, linear_model, preprocessing
from sklearn.linear_model import PoissonRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_poisson_deviance, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import normalize, Normalizer, StandardScaler, FunctionTransformer, MinMaxScaler
import helpers
import matplotlib.pyplot as plt

wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/pahowdy_prospect_database.xlsx", sheet_name = "WR", header=[0,1], na_values=["-"])

# show us all the columns that are available

print(wr_data.columns.tolist())

# first remove the 2021 prospects

wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!="2021 PROSPECT"]

# now we need to make the data a little cleaner

# make udfa an 8th draft round

wr_data.replace(to_replace="UDFA",value=8,inplace=True)

# make non-cfb conferences NA

wr_data.replace(to_replace="Non-CFB",value=np.nan,inplace=True)
wr_data["Final Year Conference Strength"] = np.array(helpers.generate_conference_strength(wr_data))
wr_data["Final Year Team Strength"] = np.array(helpers.generate_team_strength(wr_data))

# now make <2 yrs of data all nans

wr_data.replace(to_replace="<2 Yrs of Data",value=np.nan,inplace=True)

# make ms yards avg & rec yds/tm att nans their last ms yards

wr_data[('MS Yards', 'AVG')] = np.where(wr_data[('MS Yards', 'AVG')].isna, wr_data[('MS Yards', 'LAST')], wr_data[('MS Yards', 'AVG')])

# make dominator avg same as dominator last if it is nan

wr_data[('Dominator', 'AVG')] = np.where(wr_data[('Dominator', 'AVG')].isna, wr_data[('Dominator', 'LAST')], wr_data[('Dominator', 'AVG')])

# all nans for ppr pts is 0

wr_data[("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3")].fillna(value=0, inplace=True)

# fill no breakout age with a high number, carry over 20% number for people with no 30% -- this could be adjusted tbh

wr_data[('Breakout Ages', '>20%')].fillna(value=35, inplace=True)
wr_data[('Breakout Ages', '>30%')] = np.where(wr_data[('Breakout Ages', '>30%')].isna, wr_data[('Breakout Ages', '>20%')], wr_data[('Breakout Ages', '>30%')])

# take log of draft round

wr_data["logDR"] = np.log(wr_data[[('Unnamed: 3_level_0', 'DR')]]) 

# fill remaining empty columns with the mean

wr_data.fillna(wr_data.mean(), inplace=True)

# sort by name PRIOR to calculating true points

wr_data.sort_values(by=[('Unnamed: 0_level_0', 'Name')], inplace=True)
wr_data["true_points"] = np.array(helpers.generate_ppr_ppg_by_48(wr_data))

# convert to excel, save.

wr_data.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx")