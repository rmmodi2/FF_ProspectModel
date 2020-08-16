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

te_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/te_data.xlsx", header=[0])
te_data = te_data[te_data["Draft Year"]!=2020]
te_data = te_data[(te_data["Draft Year"]!=2018) | (te_data["hit_within3years"] == 1)]
te_data = te_data[(te_data["Draft Year"]!=2019) | (te_data["hit_within3years"] == 1)]

print("DEBUG: Number of players in our sample: %s" % te_data.shape[0])

hits = te_data[te_data["hit_within3years"]==1]

print("DEBUG: Number of hits in our sample: %s" % hits.shape[0])

print("DEBUG: Percent of hits in our sample: %s" % (hits.shape[0]/te_data.shape[0]))

print(te_data.columns.tolist())

draft_capital_only = ['DR', 'DP']

features = ['Age IN DRAFT YEAR (9/1/dy)', 'rec', 'YARDS', 'Ypr', 'G', 'REC/g', '# Of Years as MS Yards #1', 'College Dominator Rating', 'BOA (15%)', 'BOA (20%)', 'Dominator First', 'Dominator Best', 'Dominator Last', 'Dominator AVG', 'MS Yards First', 
'MS Yards Best', 'MS Yards Last', 'MS Yards AVG', 'RecYds/TmPatt First', 'RecYds/TmPatt Best', 'RecYds/TmPatt Last', 'RecYds/TmPatt AVG', 'YOa (yards Over Average)', 'DOa (Dom Over Average)', 'S/EX (Yds Share Over Expectatio)', 'BMI', '40 time', 'height', 
'weight', 'Bench', 'Verticle', 'Broad', 'WaSS', 'HaSS', 'Broke Out 15', 'Broke Out 20', 'Final Year Conference Strength', 'Final Year Conference Defensive Strength', 'Final Year Team Strength', 
'Final Year Team Offensive Strength',]

using = ['DP', 'REC/g', 'WaSS']

model = LogisticRegression()

# for f in interactions:
# 	if f in using:
# 		continue
# 	print("Trying out feature {}".format(f))
# 	f1_score, log_loss, results, _ = helpers.cross_validation(using+[f], te_data, "hit_within3years", model=model, logistic=True)
# 	results["Model_Class"] = results["Model"].apply(lambda x: x >= .5)
# 	print("f1_score: {}".format(f1_score))
# 	print("log loss: {}".format(log_loss))
# 	print(pd.DataFrame.from_dict(zip(using+[f], model.coef_[0]), orient='columns'))
# 	ax = plt.subplot()
# 	sns.heatmap(confusion_matrix(results["Actual"], results["Model_Class"]), ax=ax, annot=True, fmt='g')
# 	ax.set_xlabel("Predicted Labels")
# 	ax.set_ylabel("True Labels")
# 	ax.xaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# 	ax.yaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# 	plt.show()

# features, f1_score, log_loss = helpers.forward_stepwise_selection(draft_capital_only+features, te_data, "hit_within3years", model=model, logistic=True)
# features, f1_score, log_loss = helpers.backwards_stepwise_selection(using, te_data, "hit_within3years", model=model, logistic=True)
f1_score, log_loss, results, _ = helpers.cross_validation(using, te_data, "hit_within3years", model=model, logistic=True)
# results["Model_Class"] = results["Model"].apply(lambda x: x >= .5)

# print("Features: {}".format(features))
print("f1_score: {}".format(f1_score))
print("log loss: {}".format(log_loss))
# print(pd.DataFrame.from_dict(zip(using, model.coef_[0]), orient='columns'))
# ax = plt.subplot()
# sns.heatmap(confusion_matrix(results["Actual"], results["Model_Class"]), ax=ax, annot=True, fmt='g')
# ax.set_xlabel("Predicted Labels")
# ax.set_ylabel("True Labels")
# ax.xaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# ax.yaxis.set_ticklabels(["No", "Top 24 Finish (f3)"])
# plt.show()

# results.to_excel("/Users/ronakmodi/FF_ProspectModel/Results/te_results.xlsx", index=False)

print(results.columns.values)

new_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/te_data.xlsx", header=[0])
new_data = new_data[(new_data["Draft Year"] > 2017) & (new_data["hit_within3years"] == 0)] 

model.fit(te_data[using], te_data["hit_within3years"])
model_proba = model.predict_proba(new_data[using])
model_proba = [sample[1] for sample in model_proba]
res = new_data[['Name','Draft Year']+using]
res.insert(5,"Model",model_proba,True)
res.insert(6,"Actual",np.repeat("UNK",len(model_proba)),True)

print(res.columns.values)

results = results.append(res,ignore_index=True)

results.to_excel("/Users/ronakmodi/FF_ProspectModel/Results/tight_end_results.xlsx",index=False)



