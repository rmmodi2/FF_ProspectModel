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
wr_data = wr_data[wr_data["Draft Year"]!=2018]
wr_data = wr_data[wr_data["Draft Year"]!=2019]

print("DEBUG: Number of players in our sample: %s" % wr_data.shape[0])
print(wr_data.columns.tolist())

draft_capital_only = ['DR', 'DP']

features = ['Age IN DRAFT YEAR', 'Years Played', 'G', 'AVG PPG', 'rec', 'YARDS', 'YPR', 'REC/g', 'FINAL MS YARDS RK', 'College Dominator Rating', 'Yards Dominator', 'Breakout Age >20%', 'Breakout Age >30%', 
'RecYds/TmPatt First', 'RecYds/TmPatt Best', 'RecYds/TmPatt Last', 'RecYds/TmPatt AVG', 'RecYds/TmPatt Above Team AVG First', 'RecYds/TmPatt Above Team AVG Best', 'RecYds/TmPatt Above Team AVG Last', 'RecYds/TmPatt Above Team AVG AVG', 'Dominator First', 
'Dominator Best', 'Dominator Last', 'Dominator AVG', 'DOa (Dom Over Average) First', 'DOa (Dom Over Average) Best', 'DOa (Dom Over Average) Last', 'DOa (Dom Over Average) AVG', 'MS Yards First', 'MS Yards Best', 'MS Yards Last', 'MS Yards AVG', 
'YOa (Yards Over Age Average) First', 'YOa (Yards Over Age Average) Best', 'YOa (Yards Over Age Average) Last', 'YOa (Yards Over Age Average) AVG', 'PPG Above conference expectation (Last Year)', 'AVG S/EX (Yds Share Over Expectation)', 
'Last S/EX (Yds Share Over Expectation)', 'TeamMate Score (TeamMate Over Expected)', 'BMI', '40 time', 'height', 'weight', 'Bench', 'Verticle', 'Broad', 'Shuttle', '3 Cone', 'WaSS', 'HaSS', 'Hand Size', 'Arm Length', 'Final Year Conference Strength', 
'Final Year Conference Defensive Strength', 'Final Year Team Strength', 'Final Year Team Offensive Strength', 'Broke Out 20', 'Broke Out 30', 'Yards Leader Final Year', 'punt_returns', 'kick_returns', 'kick_return_yards', 'punt_return_yards', 'kick_return_avg', 
'punt_return_avg', 'kick_return_td', 'punt_return_td', 'hc_tenure', 'oc_tenure', 'hc_retained', 'oc_retained', 'vacated_tgt_pct', 'vacated_rec_pct', 'vacated_yds_pct', 'Att', 'Cmp%', 'Yds', 'Y/A', 'AY/A', 'Y/C', 'Rate', 'NY/A', 'ANY/A', 'EXP']

after_fwd_sel = ['RecYds/TmPatt AVG', 'Final Year Team Strength', 'PPG Above conference expectation (Last Year)', 'Breakout Age >30%', 'Y/C']
fwd_sel_add_dc = after_fwd_sel + ['DR']
after_full_fwd_sel = ['DR', 'Breakout Age >30%', 'MS Yards Last', 'DOa (Dom Over Average) Best', 'MS Yards Best', 'Y/A']

# for f in features+draft_capital_only:
# 	if f in after_full_fwd_sel:
# 		continue
# 	model = linear_model.LogisticRegression()
# 	recall_score, balanced_accuracy, _, _ = helpers.cross_validation(after_full_fwd_sel+[f], wr_data, "broke_160pts", model=model, logistic=True)
# 	print("Recall Score (Sensitivity): {}".format(recall_score))
# 	print("Balanced Accuracy: {}".format(balanced_accuracy)) 
# 	print(pd.DataFrame({"Features":after_full_fwd_sel+[f], 'Coefficient': model.coef_[0]}))
# 	input("Press any key to continue\n")


# x = wr_data[features]
# y = wr_data['broke_160pts']
# logistic = sm.Logit(y,x)
# results = logistic.fit()
# print(results.summary())
 

model = linear_model.LogisticRegression()

recall_score, balanced_accuracy, result_df, _ = helpers.cross_validation(after_full_fwd_sel, wr_data, "broke_160pts", model=model, logistic=True)
# features, recall_score, balanced_accuracy = helpers.forward_stepwise_selection(features+draft_capital_only, wr_data, "broke_160pts", model=model, logistic=True)

print("Recall Score (Sensitivity): {}".format(recall_score))
print("Balanced Accuracy: {}".format(balanced_accuracy)) 
print(pd.DataFrame({"Features":after_full_fwd_sel, 'Coefficient': model.coef_[0]}))
# # print("Features: {}".format(features))
ax = plt.subplot()
sns.heatmap(confusion_matrix(result_df["Actual"], result_df["Model_Class"]), ax=ax, annot=True,fmt='g')
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.xaxis.set_ticklabels(["No", ">=160 ppr points"])
ax.yaxis.set_ticklabels(["No", ">=160 ppr points"])
plt.show()

result_df.to_excel("/Users/ronakmodi/FF_ProspectModel/Results/wide_receiver_logistic_regression.xlsx")

#hyperparameter training

# w = 'balanced'

# for m in max_iter:
# 	print("Max_iter: {}".format(m))
# 	model = linear_model.LogisticRegression(class_weight=w)
# 	recall_score, balanced_accuracy, result_df, _ = helpers.cross_validation(after_full_fwd_sel, wr_data, "broke_160pts", model=model, logistic=True)
# 	print("Recall Score (Sensitivity): {}".format(recall_score))
# 	print("Balanced Accuracy: {}".format(balanced_accuracy)) 
# 	print(pd.DataFrame({"Features":after_full_fwd_sel, 'Coefficient': model.coef_[0]}))
# 	ax = plt.subplot()
# 	sns.heatmap(confusion_matrix(result_df["Actual"], result_df["Model_Class"]), ax=ax, annot=True)
# 	ax.set_xlabel("Predicted Labels")
# 	ax.set_ylabel("True Labels")
# 	ax.xaxis.set_ticklabels(["No", ">=160 ppr points"])
# 	ax.yaxis.set_ticklabels(["No", ">=160 ppr points"])
# 	plt.show()

