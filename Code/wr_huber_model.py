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
from supersmoother import SuperSmoother

# first we import the excel files with our nice data

wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx", header=[0])
wr_data = wr_data[wr_data["Draft Year"]!=2020]
wr_data = wr_data[wr_data["Draft Year"]!=2018]
wr_data = wr_data[wr_data["Draft Year"]!=2019]

print("DEBUG: Number of players in our sample: %s" % wr_data.shape[0])
print(wr_data.columns.tolist())

features = ['DR', 'DP', 'Age IN DRAFT YEAR', 'Years Played', 'G', 'AVG PPG', 'rec', 'YARDS', 'YPR', 'REC/g', 'FINAL MS YARDS RK', 'College Dominator Rating', 'Yards Dominator', 'Breakout Age >20%', 'Breakout Age >30%', 
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

fwd_sel = ['DP_times_YOa (Yards Over Age Average) AVG', 'vacated_yds_pct_times_PPG Above conference expectation (Last Year)', 'Breakout Age >30%', 'PPG Above conference expectation (Last Year)_times_punt_return_avg']

to_log = ['DP', 'YOa (Yards Over Age Average) AVG', 'vacated_yds_pct', 'PPG Above conference expectation (Last Year)', 'Final Year Team Strength']

transformations=[]
for f in to_log:
	name = "LOG "+f
	sq = f+" SQUARED"
	wr_data[name] = np.log(wr_data[f])
	wr_data[sq] = np.square(wr_data[f])
	transformations.append(name)
	transformations.append(sq)
	# fwd_sel.append(name)
	# fwd_sel.append(sq)

to_interact = {
'DP':['YOa (Yards Over Age Average) AVG', 'punt_return_avg'],
'YOa (Yards Over Age Average) AVG': ['punt_return_avg'],
'vacated_yds_pct': ['PPG Above conference expectation (Last Year)', 'punt_return_avg'],
'PPG Above conference expectation (Last Year)': ['punt_return_avg']
}

interactions=[]
for k in to_interact.keys():
	for f in to_interact[k]:
		name = k+"_times_"+f
		wr_data[name] = np.multiply(wr_data[k],wr_data[f])
		# interactions.append(name)
		# fwd_sel.append(name)

# param_grid = {
# 				'epsilon':epsilon,
# 				'max_iter': max_iter,
# 				'alpha': alpha,
# 				'tol': tol
# 				}

# model = linear_model.HuberRegressor()

# rf_random = GridSearchCV(estimator=model, param_grid=param_grid, cv=ShuffleSplit(), verbose=2, scoring="r2", n_jobs=-1)

# rf_random.fit(wr_data[fwd_sel],wr_data['true_points'])

# print("Best params: {}".format(rf_random.best_params_))

# model = rf_random.best_estimator_

# for f in interactions:

# 	if f in fwd_sel:
# 		continue
# 	# x = wr_data[copy_for_p_value+[f]]
# 	# y = wr_data[('true_points', 'Unnamed: 179_level_1')]

# 	# huber = sm.RLM(y,x,M=sm.robust.norms.HuberT(3.32))
# 	# results = huber.fit(maxiter=200,tol=.7)
# 	# print(results.summary())

# 	model = linear_model.HuberRegressor(alpha=.001,epsilon=3.32,max_iter=200,tol=.7)

# 	rmse, r2, result_df, _ = helpers.cross_validation(fwd_sel+[f], wr_data, 'true_points', model=model)

# 	# 	# features, rmse, r2 = helpers.forward_stepwise_selection(best_fwd_sel, model, wr_data)

# 	# 	# print("Best Features: %s" % features)

# 	print("RMSE %s" % rmse)
# 	print("Adj R2: %s" % r2)
# 	# result_df.to_excel("/Users/ronakmodi/FF_ProspectModel/Results/wide_receiver_robusthuber_residuals.xlsx")

# 	print(pd.DataFrame.from_dict(zip(fwd_sel+[f], model.coef_), orient='columns'))
# 	# print(wr_data[best+[f]].corr())
# 	input("press enter to continue\n")

# x = wr_data[relation]
# y = wr_data['true_points']
# huber = sm.RLM(y,x,M=sm.robust.norms.HuberT(3.32))
# results = huber.fit(maxiter=200,tol=.7)
# print(results.summary())

# model = linear_model.HuberRegressor(alpha=.001,epsilon=3.32,max_iter=200,tol=.7)

# rmse, r2, result_df, _ = helpers.cross_validation(fwd_sel, wr_data, 'true_points', model=model)
# features, rmse, r2 = helpers.forward_stepwise_selection(fwd_sel, wr_data, 'true_points', model)

# print("RMSE %s" % rmse)
# print("Adj R2: %s" % r2)
# print("Features: {}".format(features))
# print(wr_data[fwd_sel].corr())
# print(pd.DataFrame.from_dict(zip(fwd_sel, model.coef_), orient='columns'))
# result_df.to_excel("/Users/ronakmodi/FF_ProspectModel/Results/wide_receiver_robusthuber_residuals.xlsx")

# for f in features:
# 	y = np.abs(result_df['Residual']).to_numpy().flatten()
# 	x = wr_data[[f]]
# 	plt.xlabel(str(f))
# 	plt.ylabel("Absolute Residuals")
# 	plt.plot(x,y,'o')
# 	lowess_est = helpers.lowess(x,y)
# 	plt.plot(lowess_est[:,0],lowess_est[:,1],'-',linewidth=2)
# 	plt.show()


epsilon = [i/100 for i in np.arange(100,600).tolist()]
max_iter = np.arange(10,510,10).tolist()
alpha = [.00001,.0001,.001,.01,.1,.5,.05,.005,.0005,.00005]
tol = [.00001,.0001,.001,.01,.1,.4,.7,.5,.25,.025,.0025]

best_rmse = 92347239473209472
best_r2 = None
best_params = []
for e in epsilon:
	for m in max_iter:
		for a in alpha:
			for t in tol:
				print("Attempting hyperparameters epsilon: {}, max_iter: {}, alpha: {}, tol: {}".format(e,m,a,t))
				model = linear_model.HuberRegressor(alpha=a,epsilon=e,max_iter=m,tol=t)
				try:
					rmse, r2, _, _ = helpers.cross_validation(fwd_sel, wr_data, 'true_points', model=model)
					if rmse < best_rmse:
						best_params = [e,m,a,t]
						best_rmse = rmse
						best_r2 = r2
				except:
					continue
print("RMSE: {}".format(best_rmse))
print("R2: {}".format(best_r2))
print("Best Params: epsilon: {}, max_iter: {}, alpha: {}, tol: {}".format(*best_params))











