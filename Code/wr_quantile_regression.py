import pandas as pd
import numpy as np
import math
import scipy
import sys
import collections
from scipy import stats
from scipy.special import inv_boxcox
from math import sqrt
from sklearn import datasets, linear_model, preprocessing
from sklearn.linear_model import PoissonRegressor, HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_poisson_deviance, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, ShuffleSplit
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, GridSearchCV, ShuffleSplit
import helpers
import matplotlib.pyplot as plt
import statsmodels.api as sm

# first we import the excel files with our nice data

wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx", header=[0])
wr_data = wr_data[wr_data["Draft Year"]!=2020]
wr_data = wr_data[wr_data["Draft Year"]!=2018]
wr_data = wr_data[wr_data["Draft Year"]!=2019]

print("DEBUG: Number of players in our sample: %s" % wr_data.shape[0])

print(wr_data.columns.tolist())

ffeatures = ['DR', 'DP', 'Age IN DRAFT YEAR', 'Years Played', 'G', 'AVG PPG', 'rec', 'YARDS', 'YPR', 'REC/g', 'FINAL MS YARDS RK', 'College Dominator Rating', 'Yards Dominator', 'Breakout Age >20%', 'Breakout Age >30%', 
'RecYds/TmPatt First', 'RecYds/TmPatt Best', 'RecYds/TmPatt Last', 'RecYds/TmPatt AVG', 'RecYds/TmPatt Above Team AVG First', 'RecYds/TmPatt Above Team AVG Best', 'RecYds/TmPatt Above Team AVG Last', 'RecYds/TmPatt Above Team AVG AVG', 'Dominator First', 
'Dominator Best', 'Dominator Last', 'Dominator AVG', 'DOa (Dom Over Average) First', 'DOa (Dom Over Average) Best', 'DOa (Dom Over Average) Last', 'DOa (Dom Over Average) AVG', 'MS Yards First', 'MS Yards Best', 'MS Yards Last', 'MS Yards AVG', 
'YOa (Yards Over Age Average) First', 'YOa (Yards Over Age Average) Best', 'YOa (Yards Over Age Average) Last', 'YOa (Yards Over Age Average) AVG', 'PPG Above conference expectation (Last Year)', 'AVG S/EX (Yds Share Over Expectation)', 
'Last S/EX (Yds Share Over Expectation)', 'TeamMate Score (TeamMate Over Expected)', 'BMI', '40 time', 'height', 'weight', 'Bench', 'Verticle', 'Broad', 'Shuttle', '3 Cone', 'WaSS', 'HaSS', 'Hand Size', 'Arm Length', 'Final Year Conference Strength', 
'Final Year Conference Defensive Strength', 'Final Year Team Strength', 'Final Year Team Offensive Strength', 'Broke Out 20', 'Broke Out 30', 'Yards Leader Final Year', 'punt_returns', 'kick_returns', 'kick_return_yards', 'punt_return_yards', 'kick_return_avg', 
'punt_return_avg', 'kick_return_td', 'punt_return_td', 'hc_tenure', 'oc_tenure', 'hc_retained', 'oc_retained', 'vacated_tgt_pct', 'vacated_rec_pct', 'vacated_yds_pct', 'Att', 'Cmp%', 'Yds', 'Y/A', 'AY/A', 'Y/C', 'Rate', 'NY/A', 'ANY/A', 'EXP']

relation = ['DP', 'G', 'AVG PPG', 'rec', 'YARDS', 'YPR', 'REC/g', 'FINAL MS YARDS RK', 'College Dominator Rating', 'Yards Dominator', 'RecYds/TmPatt Best', 'RecYds/TmPatt Last', 'RecYds/TmPatt AVG', 
'RecYds/TmPatt Above Team AVG Best', 'RecYds/TmPatt Above Team AVG Last', 'RecYds/TmPatt Above Team AVG AVG', 'Dominator Best', 'Dominator Last', 'Dominator AVG', 'DOa (Dom Over Average) Best', 
'DOa (Dom Over Average) Last', 'DOa (Dom Over Average) AVG', 'MS Yards AVG', 'YOa (Yards Over Age Average) Best', 'YOa (Yards Over Age Average) Last', 'YOa (Yards Over Age Average) AVG', 
'PPG Above conference expectation (Last Year)', 'AVG S/EX (Yds Share Over Expectation)', 'TeamMate Score (TeamMate Over Expected)', '40 time', 'height', 'weight',
'Bench', 'Broad', 'Shuttle', '3 Cone', 'WaSS', 'HaSS', 'Hand Size', 'Arm Length', 'punt_returns', 'hc_tenure', 'vacated_tgt_pct', 'vacated_rec_pct', 'vacated_yds_pct',
'Final Year Conference Defensive Strength', 'Final Year Team Strength', 'Final Year Team Offensive Strength', 'Final Year Conference Strength','DR', 'Age IN DRAFT YEAR', 'Breakout Age >20%', 'Breakout Age >30%', 'Broke Out 20', 'Broke Out 30', 
'Yards Leader Final Year',]

fwd_sel = ['DR', 'Breakout Age >30%', 'RecYds/TmPatt AVG']

# Hyperparameter tuning first

# hyperparam = {}
# hyperparam['best_rmse'] = 23943240324932
# hyperparam['best_r2'] = None
# hyperparam['best_quantile'] = None
# hyperparam['best_iter'] = None
# hyperparam['best_threshold'] = None
# for q in [.5768]:
# 	for iterations in [20]:
# 		for threshold in [5e-1]:
# 			rmse, r2, _ = helpers.cross_validation(best, wr_data, model=None,quantile=q, max_iter=iterations, p_tol=threshold)
# 			if rmse < hyperparam['best_rmse']:
# 				hyperparam['best_rmse'] = rmse
# 				hyperparam['best_r2'] = r2
# 				hyperparam['best_iter'] = iterations
# 				hyperparam['best_quantile'] = q
# 				hyperparam['best_threshold'] = threshold
# for k in hyperparam.keys():
# 	print("%s: %s" % (k, hyperparam[k]))

# Forward selection

# best_features, rmse, adjr2 = helpers.forward_stepwise_selection(p_value_initial, wr_data, quantile=.5768, max_iter=20, p_tol=5e-1)
# print(best_features)
# print(rmse)
# print(adjr2)

# Backward selection

# best_features, rmse, adjr2 = helpers.backwards_stepwise_selection(initial_features, wr_data, quantile=.5768, max_iter=20, p_tol=5e-1)
# print(best_features)
# print(rmse)
# print(adjr2)

# for f in transformations:

# 	x = wr_data[trim_fwd+[f]]
# 	y = wr_data["true_points"]

# 	rmse, r2, results, statsmodels_res = helpers.cross_validation(trim_fwd+[f], wr_data, "true_points", quantile=.5768, max_iter=20, p_tol=5e-1)

# 	# 	# features, rmse, r2 = helpers.forward_stepwise_selection(best_fwd_sel, model, wr_data)

# 	# 	# print("Best Features: %s" % features)

# 	print("RMSE %s" % rmse)
# 	print("Adj R2: %s" % r2)
# 	# result_df.to_excel("/Users/ronakmodi/FF_ProspectModel/Results/wide_receiver_robusthuber_residuals.xlsx")

# 	print(statsmodels_res.summary())
# 	input("press enter to continue\n")

# rmse, r2, results, res = helpers.cross_validation(relation, wr_data, "true_points", quantile=.5768, max_iter=20, p_tol=5e-1)
features, rmse, r2 = helpers.backwards_stepwise_selection(relation, wr_data, "true_points", quantile=.5768, max_iter=20, p_tol=5e-1)

print("Best features: {}".format(features))
print("RMSE: {}".format(rmse))
print("R2: {}".format(r2))
# print(res.summary())





