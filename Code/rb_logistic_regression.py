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

rb_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/rb_data.xlsx", header=[0])

rb_data = rb_data[rb_data["Draft Year"]!=2020]
rb_data = rb_data[(rb_data["Draft Year"]!=2018) | (rb_data["hit_within3years"] == 1)]
rb_data = rb_data[(rb_data["Draft Year"]!=2019) | (rb_data["hit_within3years"] == 1)]

print("DEBUG: Number of players in our sample: %s" % rb_data.shape[0])

hits = rb_data[rb_data["hit_within3years"]==1]

print("DEBUG: Number of hits in our sample: %s" % hits.shape[0])

print("DEBUG: Percent of hits in our sample: %s" % (hits.shape[0]/rb_data.shape[0]))

rb_data["RUSH% AVG"] = rb_data["RUSH% AVG"].apply(lambda x: x+.01)

print("DEBUG: Number of players in our sample: %s" % rb_data.shape[0])
print(rb_data.columns.tolist())

draft_capital = ['DR', 'DP',]

features = ['Age IN DRAFT YEAR', 'Years Played', 'rec', 'Touches', 'All Yards', 'YPTch', 'G', 'Years >= 20 Recs', 'REC/g', 'REC Yards % First', 'REC Yards % Best', 'REC Yards % Last', 'REC Yards % AVG', 'Yards/Team Att First', 'Yards/Team Att Best', 
'Yards/Team Att Last', 'Yards/Team Att AVG', 'Rush College Dominator Rating', 'Total Dominator First', 'Total Dominator Last', 'Total Dominator Best', 'Total Dominator AVG', 'TDOa (Total Dominator over average) First', 'TDOa (Total Dominator over average) Last', 
'TDOa (Total Dominator over average) Best', 'TDOa (Total Dominator over average) AVG', 'PPR Pts First', 'PPR Pts Last', 'PPR Pts Best', 'PPR Pts AVG', 'PPR PPG First', 'PPR PPG Last', 'PPR PPG Best', 'PPR PPG AVG', 'BMI', '40 time', 'height', 'weight', 'Bench', 
'Verticle', 'Broad', 'Shuttle', '3 Cone', 'WaSS', 'HaSS','Final Year Conference Strength', 'Final Year Conference Defensive Strength', 'Final Year Team Strength', 'Final Year Team Offensive Strength', 'punt_returns', 'kick_returns', 'kick_return_yards', 
'punt_return_yards', 'kick_return_avg', 'punt_return_avg', 'kick_return_td', 'punt_return_td',]

fwd_sel = ['DP', 'Total Dominator AVG', 'WaSS', 'Final Year Conference Strength']


model = LogisticRegression()

# for f in interactions:
# 	if f in fwd_sel:
# 		continue
# 	print("Trying out feature {}".format(f))
# 	f1_score, log_loss, results, _ = helpers.cross_validation(fwd_sel+[f], rb_data, "hit_within3years", model=model, logistic=True)
# 	results["Model_Class"] = results["Model"].apply(lambda x: x >= .5)
# 	print("f1_score: {}".format(f1_score))
# 	print("log loss: {}".format(log_loss))
# 	print(pd.DataFrame.from_dict(zip(fwd_sel+[f], model.coef_[0]), orient='columns'))
# 	ax = plt.subplot()
# 	sns.heatmap(confusion_matrix(results["Actual"], results["Model_Class"]), ax=ax, annot=True, fmt='g')
# 	ax.set_xlabel("Predicted Labels")
# 	ax.set_ylabel("True Labels")
# 	ax.xaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# 	ax.yaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# 	plt.show()

# features, f1_score, log_loss = helpers.forward_stepwise_selection(features+draft_capital, rb_data, "hit_within3years", model=model, logistic=True)
f1_score, log_loss, results, _ = helpers.cross_validation(fwd_sel, rb_data, "hit_within3years", model=model, logistic=True)
# results["Model_Class"] = results["Model"].apply(lambda x: x >= .5)

# print("Features: {}".format(features))
print("f1_score: {}".format(f1_score))
print("log loss: {}".format(log_loss))
print(pd.DataFrame.from_dict(zip(fwd_sel, model.coef_[0]), orient='columns'))
# ax = plt.subplot()
# sns.heatmap(confusion_matrix(results["Actual"], results["Model_Class"]), ax=ax, annot=True, fmt='g')
# ax.set_xlabel("Predicted Labels")
# ax.set_ylabel("True Labels")
# ax.xaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# ax.yaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# plt.show()

print(results.columns.values)

new_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/rb_data.xlsx", header=[0])
new_data = new_data[(new_data["Draft Year"] > 2017) & (new_data["hit_within3years"] == 0)] 

model.fit(rb_data[fwd_sel], rb_data["hit_within3years"])
model_proba = model.predict_proba(new_data[fwd_sel])
model_proba = [sample[1] for sample in model_proba]
res = new_data[['Name','Draft Year']+fwd_sel]
res.insert(6,"Model",model_proba,True)
res.insert(7,"Actual",np.repeat("UNK",len(model_proba)),True)

print(res.columns.values)

results = results.append(res,ignore_index=True)

results.to_excel("/Users/ronakmodi/FF_ProspectModel/Results/running_back_results.xlsx",index=False)

