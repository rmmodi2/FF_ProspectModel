"""
THIS FILE IS DEPRECATED. NO LONGER USED.
"""



import pandas as pd
import numpy as np
import math
from math import sqrt
from math import log
from sklearn import datasets, linear_model, preprocessing
from sklearn.linear_model import PoissonRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_poisson_deviance, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import normalize, Normalizer, StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
import helpers
import matplotlib.pyplot as plt


wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx", header=[0,1])
wr_data_with_prospects = wr_data
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!=2020]
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!=2018]
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!=2019]

print("DEBUG: Number of players in our sample: %s" % wr_data.shape[0])
print(wr_data.columns.tolist())


# get rid of columns we don't want in our model

model_columns = wr_data[[('logDR', 'Unnamed: 164_level_1'), ('RecYds/TmPatt', 'AVG'), ('DOa (Dom Over Average)', 'Doa (AVG)'), ('Total Counting Stats', 'AVG PPG'), ('Final Year Team Strength', 'Unnamed: 163_level_1'), ("Combine", "BMI"), 
("Final Year Conference Strength", "Unnamed: 162_level_1")]]

# now we have to split it into an X array and a Y array

X_model = np.array(model_columns.values.tolist())
Y = np.array(wr_data[[('true_points', 'Unnamed: 165_level_1')]].values.tolist())


# can try out conferences later , order by rank and then try.

# conferences = ["ACC","American","Big 12", "Big East", "Big Ten", "CUSA", "Ind", "MAC", "MWC", "Pac-12", "SEC", "Sun Belt", "WAC"]

# # transform conference into an ordinal

# transformer = ColumnTransformer(
# 	[("encode_conferences", OrdinalEncoder(categories=[conferences]), [0])],
# 	remainder="passthrough"
# 	)

# X_model = transformer.fit_transform(X_model)

# Hyperparameter Tuning

model = RandomForestRegressor()

# Randomized Search to lower overhead time

# n_estimators = [10,25,50,100,200,300,400,500,600,700,800,900,1000]
# max_features = ['auto','sqrt','log2']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10, 20, 40, 75, 100]
# min_samples_leaf = [1, 2, 4, 8, 16, 32, 64]
# bootstrap = [True]

# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 500, cv = 5, verbose=2, scoring="neg_root_mean_squared_error")


# We can do a grid search with the parameters we found worked the best from random search

# n_estimators = [10,25,50,100,200,300,400,500,600]
# max_features = ['auto','sqrt','log2']
# max_depth = [10,20,30,40,50,60,70,80]
# max_depth.append(None)
# min_samples_split = [2, 5, 10, 20, 30, 40]
# min_samples_leaf = [2, 4, 8, 16, 32, 64]
# bootstrap = [True]

# param_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}


# rf_random = GridSearchCV(estimator = model, param_grid=param_grid, cv=5, verbose=2, scoring="neg_root_mean_squared_error")

# rf_random.fit(X_model,Y.ravel())

# print("DEBUG: Best Parameters for this run: %s" % rf_random.best_params_)

# # feature importances
# feature_weight = pd.DataFrame({"Feature": model_columns.columns.tolist(), "Importance": rf_random.best_estimator_.feature_importances_})
# print(feature_weight)


# # now let's test on our sample of single draft classes

# model = rf_random.best_estimator_
# # model = RandomForestRegressor(n_estimators=10, min_samples_split=10, min_samples_leaf=16, max_features='auto', max_depth=40, bootstrap=True)

# real_r_2 = 0
# real_rmse = 0
# for i in range(50):
# 	model_predicted = np.array([])
# 	actual_values = np.array([])
# 	yrs = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
# 	for yr in yrs:
# 		train = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"] != yr]
# 		test = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"] == yr]

# 		train_columns = train[[('logDR', 'Unnamed: 164_level_1'), ('RecYds/TmPatt', 'AVG'), ('DOa (Dom Over Average)', 'Doa (AVG)'), ('Total Counting Stats', 'AVG PPG'), ('Final Year Team Strength', 'Unnamed: 163_level_1'), ("Combine", "BMI"), 
# 		("Final Year Conference Strength", "Unnamed: 162_level_1")]]
# 		X_train = np.array(train_columns.values.tolist())
# 		test_columns = test[[('logDR', 'Unnamed: 164_level_1'), ('RecYds/TmPatt', 'AVG'), ('DOa (Dom Over Average)', 'Doa (AVG)'), ('Total Counting Stats', 'AVG PPG'), ('Final Year Team Strength', 'Unnamed: 163_level_1'), ("Combine", "BMI"), 
# 		("Final Year Conference Strength", "Unnamed: 162_level_1")]]
# 		X_test = np.array(test_columns.values.tolist())

# 		Y_train = np.array(train[[('true_points', 'Unnamed: 165_level_1')]].values.tolist())
# 		Y_test = np.array(test[[('true_points', 'Unnamed: 165_level_1')]].values.tolist())

# 		model.fit(X_train, Y_train.ravel())
# 		Y_pred = model.predict(X_test)
# 		model_predicted = np.append(model_predicted, Y_pred)
# 		actual_values = np.append(actual_values, Y_test)

# 	model_r_2 = 1-(1-r2_score(actual_values, model_predicted))*((wr_data.shape[0]-1)/(wr_data.shape[0]-X_train.shape[1]-1))
# 	model_rmse = mean_squared_error(actual_values, model_predicted, squared=False)
# 	real_r_2 += model_r_2
# 	real_rmse += model_rmse

# print("DEBUG: ENSEMBLE MODEL Average R^2 for every draft class in our data %s" % (real_r_2/50))
# print("DEBUG: ENSEMBLE MODEL Average RMSE for every draft class in our data %s" % (real_rmse/50))

model = RandomForestRegressor(n_estimators=100, min_samples_split=10, min_samples_leaf=2, max_features='log2', max_depth=30, bootstrap=True)

test_yr = 2019

train = wr_data
test = wr_data_with_prospects[wr_data_with_prospects["Unnamed: 6_level_0"]["Draft Year"] == test_yr]

train_columns = train[[('logDR', 'Unnamed: 164_level_1'), ('RecYds/TmPatt', 'AVG'), ('DOa (Dom Over Average)', 'Doa (AVG)'), ('Total Counting Stats', 'AVG PPG'), ('Final Year Team Strength', 'Unnamed: 163_level_1'), ("Combine", "BMI"), 
("Final Year Conference Strength", "Unnamed: 162_level_1")]]
X_train = np.array(train_columns.values.tolist())
test_columns = test[[('logDR', 'Unnamed: 164_level_1'), ('RecYds/TmPatt', 'AVG'), ('DOa (Dom Over Average)', 'Doa (AVG)'), ('Total Counting Stats', 'AVG PPG'), ('Final Year Team Strength', 'Unnamed: 163_level_1'), ("Combine", "BMI"), 
("Final Year Conference Strength", "Unnamed: 162_level_1")]]
X_test = np.array(test_columns.values.tolist())

Y_train = np.array(train[[('true_points', 'Unnamed: 165_level_1')]].values.tolist())

model.fit(X_train, Y_train.ravel())
Y_pred = model.predict(X_test)

result = pd.DataFrame({"Name": test[(('Unnamed: 0_level_0', 'Name'))], "Predicted PPR PPG Years 1-3": Y_pred.ravel()})
result.sort_values(by="Predicted PPR PPG Years 1-3",inplace=True, ascending=False)
print(result)








