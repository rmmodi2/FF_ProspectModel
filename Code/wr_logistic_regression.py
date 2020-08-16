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
from sklearn.linear_model import PoissonRegressor, HuberRegressor, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_poisson_deviance, mean_absolute_error, confusion_matrix
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, ShuffleSplit
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, GridSearchCV, ShuffleSplit
import helpers
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from supersmoother import SuperSmoother

wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx", header=[0])
wr_data = wr_data[wr_data["Draft Year"]!=2020]
wr_data = wr_data[(wr_data["Draft Year"]!=2018) | (wr_data["hit_within3years"] == 1)]
wr_data = wr_data[(wr_data["Draft Year"]!=2019) | (wr_data["hit_within3years"] == 1)]

print("DEBUG: Number of players in our sample: %s" % wr_data.shape[0])

hits = wr_data[wr_data["hit_within3years"]==1]

print("DEBUG: Number of hits in our sample: %s" % hits.shape[0])

print("DEBUG: Percent of hits in our sample: %s" % (hits.shape[0]/wr_data.shape[0]))

print(wr_data.columns.tolist())

draft_capital_only = ['DR', 'DP']

features = ['Age IN DRAFT YEAR', 'Years Played', 'G', 'AVG PPG', 'rec', 'YARDS', 'YPR', 'REC/g', 'FINAL MS YARDS RK', 'College Dominator Rating', 'Yards Dominator', 'Breakout Age >20%', 'Breakout Age >30%', 
'RecYds/TmPatt First', 'RecYds/TmPatt Best', 'RecYds/TmPatt Last', 'RecYds/TmPatt AVG', 'RecYds/TmPatt Above Team AVG First', 'RecYds/TmPatt Above Team AVG Best', 'RecYds/TmPatt Above Team AVG Last', 'RecYds/TmPatt Above Team AVG AVG', 'Dominator First', 
'Dominator Best', 'Dominator Last', 'Dominator AVG', 'DOa (Dom Over Average) First', 'DOa (Dom Over Average) Best', 'DOa (Dom Over Average) Last', 'DOa (Dom Over Average) AVG', 'MS Yards First', 'MS Yards Best', 'MS Yards Last', 'MS Yards AVG', 
'YOa (Yards Over Age Average) First', 'YOa (Yards Over Age Average) Best', 'YOa (Yards Over Age Average) Last', 'YOa (Yards Over Age Average) AVG', 'PPG Above conference expectation (Last Year)', 'AVG S/EX (Yds Share Over Expectation)', 
'Last S/EX (Yds Share Over Expectation)', 'TeamMate Score (TeamMate Over Expected)', 'BMI', '40 time', 'height', 'weight', 'Bench', 'Verticle', 'Broad', 'Shuttle', '3 Cone', 'WaSS', 'HaSS', 'Hand Size', 'Arm Length', 'Final Year Conference Strength', 
'Final Year Conference Defensive Strength', 'Final Year Team Strength', 'Final Year Team Offensive Strength', 'Broke Out 20', 'Broke Out 30', 'Yards Leader Final Year', 'punt_returns', 'kick_returns', 'kick_return_yards', 'punt_return_yards', 'kick_return_avg', 
'punt_return_avg', 'kick_return_td', 'punt_return_td', 'hc_tenure', 'oc_tenure', 'hc_retained', 'oc_retained', 'vacated_tgt_pct', 'vacated_rec_pct', 'vacated_yds_pct', 'Att', 'Cmp%', 'Yds', 'Y/A', 'AY/A', 'Y/C', 'Rate', 'NY/A', 'ANY/A', 'EXP']

using = ['DP', 'YOa (Yards Over Age Average) AVG', 'PPG Above conference expectation (Last Year)', 'hc_tenure', 'Breakout Age >20%', 'Last S/EX (Yds Share Over Expectation)']

interaction = {
	'DP':['YOa (Yards Over Age Average) AVG', 'PPG Above conference expectation (Last Year)', 'hc_tenure', 'Breakout Age >20%', 'Last S/EX (Yds Share Over Expectation)',],
	'YOa (Yards Over Age Average) AVG': ['PPG Above conference expectation (Last Year)', 'hc_tenure', 'Breakout Age >20%', 'Last S/EX (Yds Share Over Expectation)',],
	'PPG Above conference expectation (Last Year)': ['hc_tenure', 'Breakout Age >20%', 'Last S/EX (Yds Share Over Expectation)',],
	'Breakout Age >20%': ['Last S/EX (Yds Share Over Expectation)',]
}

candidates = []

for k in interaction.keys():
	for v in interaction[k]:
		name = k+"_times_"+v
		wr_data[name] = np.multiply(wr_data[k],wr_data[v])
		candidates.append(name)

model = LogisticRegression()

# for f in candidates+trying+using:
# 	if f in trying:
# 		continue
# 	print("Trying out feature {}".format(f))
# 	f1_score, log_loss, results, _ = helpers.cross_validation(trying+[f], wr_data, "hit_within3years", model=model, logistic=True)
# 	print("f1_score: {}".format(f1_score))
# 	print("log loss: {}".format(log_loss))
# 	print(pd.DataFrame.from_dict(zip(trying+[f], model.coef_[0]), orient='columns'))
# 	ax = plt.subplot()
# 	sns.heatmap(confusion_matrix(results["Actual"], results["Model_Class"]), ax=ax, annot=True, fmt='g')
# 	ax.set_xlabel("Predicted Labels")
# 	ax.set_ylabel("True Labels")
# 	ax.xaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# 	ax.yaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# 	plt.show()

# features, f1_score, log_loss = helpers.forward_stepwise_selection(using+candidates, wr_data, "hit_within3years", model=model, logistic=True)
# features, f1_score, log_loss = helpers.backwards_stepwise_selection(using, wr_data, "hit_within3years", model=model, logistic=True)
f1_score, log_loss, results, _ = helpers.cross_validation(using, wr_data, "hit_within3years", model=model, logistic=True)

# print("Features: {}".format(features))
# print("f1_score: {}".format(f1_score))
# print("log loss: {}".format(log_loss))
# print(pd.DataFrame.from_dict(zip(using, model.coef_[0]), orient='columns'))
# ax = plt.subplot()
# sns.heatmap(confusion_matrix(results["Actual"], results["Model_Class"]), ax=ax, annot=True, fmt='g')
# ax.set_xlabel("Predicted Labels")
# ax.set_ylabel("True Labels")
# ax.xaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# ax.yaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# plt.show()

# results.to_excel("/Users/ronakmodi/FF_ProspectModel/Results/wide_receiver_logistic_regression.xlsx", index=False)

print(results.columns.values)

new_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx", header=[0])
new_data = new_data[(new_data["Draft Year"] > 2017) & (new_data["hit_within3years"] == 0)] 

model.fit(wr_data[using], wr_data["hit_within3years"])
model_proba = model.predict_proba(new_data[using])
model_proba = [sample[1] for sample in model_proba]
res = new_data[['Name','Draft Year']+using]
res.insert(8,"Model",model_proba,True)
res.insert(9,"Actual",np.repeat("UNK",120),True)

print(res.columns.values)

results = results.append(res,ignore_index=True)

# results.to_excel("/Users/ronakmodi/FF_ProspectModel/Results/wide_receiver_results.xlsx",index=False)






