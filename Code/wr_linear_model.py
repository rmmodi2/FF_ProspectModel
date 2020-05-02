"""
THIS IS OUR BEST MODEL FOR WIDE RECEIVERS
"""

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

# first we import the excel files with our nice data

wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx", header=[0,1])
wr_data_with_prospects = wr_data
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!=2020]
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!=2018]
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!=2019]

print("DEBUG: Number of players in our sample: %s" % wr_data.shape[0])
print(wr_data.columns.tolist())

# Initialize empty values for the Y^ model, Y^ baseline, and actual Y.
model_predicted = np.array([])
baseline_predicted = np.array([])
actual_values = np.array([])

# Years we are going to iterate through. All where receivers have played out the 3 years already.
yrs = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]


# Iterate through each year
for yr in yrs:
	train = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"] != yr]
	test = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"] == yr]

	train_columns = train[[('logDR', 'Unnamed: 164_level_1'), ('RecYds/TmPatt', 'AVG'), ('DOa (Dom Over Average)', 'Doa (AVG)')]]
	X_train = np.array(train_columns.values.tolist())
	dc_train = np.array(train[[('Unnamed: 3_level_0', 'DR')]].values.tolist())
	test_columns = test[[('logDR', 'Unnamed: 164_level_1'), ('RecYds/TmPatt', 'AVG'), ('DOa (Dom Over Average)', 'Doa (AVG)')]]
	X_test = np.array(test_columns.values.tolist())
	dc_test = np.array(test[[('Unnamed: 3_level_0', 'DR')]].values.tolist())

	Y_train = np.array(train[[('true_points', 'Unnamed: 165_level_1')]].values.tolist())
	Y_test = np.array(test[[('true_points', 'Unnamed: 165_level_1')]].values.tolist())


	#  Normalize model feature inputs (does it row-wise.. wierd.)
	# normalize(X_train,copy=False)
	# normalize(X_test,copy=False)

	#standardize

	scaler = StandardScaler()
	X_train=scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)
	dc_train = scaler.fit_transform(dc_train)
	dc_test = scaler.fit_transform(dc_test)


	# Initialize Linear Regression Models
	model = linear_model.LinearRegression()
	draftcap = linear_model.LinearRegression()

	# Fit the model and baseline on training year and generate predictions
	model.fit(X_train, Y_train)
	draftcap.fit(dc_train,Y_train)
	Y_pred = model.predict(X_test)
	dc_pred = draftcap.predict(dc_test)

	# Append predictions
	model_predicted = np.append(model_predicted, Y_pred)
	baseline_predicted = np.append(baseline_predicted, dc_pred)
	actual_values = np.append(actual_values, Y_test)

# Calculate total r^2 and total RMSE on overall predictions

mean = np.array([np.mean(actual_values) for i in range(actual_values.shape[0])])

dc_r_2 = r2_score(actual_values, baseline_predicted)
dc_rmse = mean_squared_error(actual_values, baseline_predicted, squared=False)
model_r_2 = 1-(1-r2_score(actual_values, model_predicted))*((wr_data.shape[0]-1)/(wr_data.shape[0]-X_train.shape[1]-1))
model_rmse = mean_squared_error(actual_values, model_predicted, squared=False)
mean_r_2 = r2_score(actual_values, mean)
mean_rmse = mean_squared_error(actual_values, mean, squared=False)

print("DEBUG: SIMPLE MEAN R^2 for every draft class in our data %s" % mean_r_2)
print("DEBUG: SIMPLE MEAN RMSE for every draft class in our data %s" % mean_rmse)
print("DEBUG: LINEAR DRAFTCAPITAL R^2 for every draft class in our data %s" % dc_r_2)
print("DEBUG: LINEAR DRAFTCAPITAL RMSE for every draft class in our data %s" % dc_rmse)
print("DEBUG: LINEAR MODEL R^2 for every draft class in our data %s" % model_r_2)
print("DEBUG: LINEAR MODEL RMSE for every draft class in our data %s" % model_rmse)

# let's see what coefficients our model came up with

print("Features %s" % list(train_columns))
print("Weights %s" % list(model.coef_))

# result = pd.DataFrame({"Name": test[(('Unnamed: 0_level_0', 'Name'))], "Predicted PPR PPG Years 1-3": Y_pred.ravel(), "Baseline Pred PPR PPG Years 1-3": dc_pred.ravel(), "Actual PPR ...": Y_test.ravel()})
# result.sort_values(by="Actual PPR ...",inplace=True, ascending=False)
# print(result)


# let's graph the model / draft capital / actual for a random train/test split

# model_x_train, model_x_test, model_draft_train, model_draft_test, model_y_train, model_y_test = train_test_split(X_model, X_draftcap, Y, test_size=.1, shuffle=True)

# normalize(model_x_train,copy=False)
# normalize(model_x_test,copy=False)

# model.fit(model_x_train, model_y_train.ravel())
# draft_capital_only.fit(model_draft_train, model_y_train)

# model_pred = model.predict(model_x_test)
# draft_cap_pred = draft_capital_only.predict(model_draft_test)

# plt.plot(range(len(model_pred)), model_pred, "g--", range(len(model_pred)), model_y_test, "r^", range(len(model_pred)), draft_cap_pred, "bo")
# plt.xlabel("index of observation")
# plt.ylabel("PPG - model in green, draft capital in blue, actual in red")
# plt.title("Predicting PPR PPG of Wide Receiver Prospects (Years 1-3)")
# plt.grid(True)
# plt.show()

model = linear_model.LinearRegression()

test_yr = 2018

train = wr_data
test = wr_data_with_prospects[wr_data_with_prospects["Unnamed: 6_level_0"]["Draft Year"] == test_yr]

train_columns = train[[('logDR', 'Unnamed: 164_level_1'), ('RecYds/TmPatt', 'AVG'), ('DOa (Dom Over Average)', 'Doa (AVG)')]]
X_train = np.array(train_columns.values.tolist())
test_columns = test[[('logDR', 'Unnamed: 164_level_1'), ('RecYds/TmPatt', 'AVG'), ('DOa (Dom Over Average)', 'Doa (AVG)')]]
X_test = np.array(test_columns.values.tolist())

Y_train = np.array(train[[('true_points', 'Unnamed: 165_level_1')]].values.tolist())

model.fit(X_train, Y_train.ravel())
Y_pred = model.predict(X_test)

result = pd.DataFrame({"Name": test[(('Unnamed: 0_level_0', 'Name'))], "Predicted PPR PPG Years 1-3": Y_pred.ravel()})
result.sort_values(by="Predicted PPR PPG Years 1-3",inplace=True, ascending=False)
print(result)














	

