
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
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import normalize, Normalizer, StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.feature_selection import RFECV
from supersmoother import SuperSmoother
import helpers
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

# first we import the excel files with our nice data

wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx", header=[0,1])
wr_data_with_prospects = wr_data
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!=2020]
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!=2018]
wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!=2019]

print("DEBUG: Number of players in our sample: %s" % wr_data.shape[0])

# value = input("give me a lambda:\n")
# value = float(value)

# feature selection

features = [
('Unnamed: 3_level_0', 'DR'), ('Unnamed: 4_level_0', 'DP'), ('Unnamed: 7_level_0', 'Age IN DRAFT YEAR'), ('Total Counting Stats', 'Years Played'), ('Total Counting Stats', 'G'), ('Total Counting Stats', 'AVG PPG'), 
('Total Counting Stats', 'rec'), ('Total Counting Stats', 'YARDS'), ('Total Counting Stats', 'YPR'), ('Total Counting Stats', 'REC/g'), ('Total Counting Stats', 'FINAL MS YARDS RK'), ('Total Counting Stats', 'College Dominator Rating'), 
('Total Counting Stats', 'Yards Dominator'), ('Breakout Ages', '>20%'), ('Breakout Ages', '>30%'), ('RecYds/TmPatt', 'First'), ('RecYds/TmPatt', 'Best'), ('RecYds/TmPatt', 'Last'), ('RecYds/TmPatt', 'AVG'), ('RecYds/TmPatt Above Team AVG', 'First'), 
('RecYds/TmPatt Above Team AVG', 'Best'), ('RecYds/TmPatt Above Team AVG', 'Last'), ('RecYds/TmPatt Above Team AVG', 'AVG'), ('Dominator', 'First'), ('Dominator', 'Best'), ('Dominator', 'Last'), ('Dominator', 'AVG'), ('DOa (Dom Over Average)', 'First'), 
('DOa (Dom Over Average)', 'Best'), ('DOa (Dom Over Average)', 'Last'), ('DOa (Dom Over Average)', 'AVG'), ('MS Yards', 'First'), ('MS Yards', 'Best'), ('MS Yards', 'Last'), ('MS Yards', 'AVG'), ('YOa (Yards Over Age Average)', 'First'), 
('YOa (Yards Over Age Average)', 'Best'), ('YOa (Yards Over Age Average)', 'Last'), ('YOa (Yards Over Age Average)', 'AVG'), ('Context Scores', 'PPG Above conference expectation (Last Year)'), ('Context Scores', 'AVG S/EX (Yds Share Over Expectation)'), 
('Context Scores', 'Last S/EX (Yds Share Over Expectation)'), ('Context Scores', 'TeamMate Score (TeamMate Over Expected)'), ('Combine', 'BMI'), ('Combine', '40 time'), ('Combine', 'height'), ('Combine', 'weight'), ('Combine', 'Bench'), ('Combine', 'Verticle'), 
('Combine', 'Broad'), ('Combine', 'Shuttle'), ('Combine', '3 Cone'), ('Combine', 'WaSS'), ('Combine', 'HaSS'), ('Combine', 'Hand Size'), ('Combine', 'Arm Length'), ('Final Year Conference Strength', 'Unnamed: 169_level_1'), 
('Final Year Team Strength', 'Unnamed: 172_level_1')
]

# create feature transformations / interactions

wr_data["AVGPPG SQUARED"] = np.square(wr_data[[('Total Counting Stats', 'AVG PPG')]])
wr_data["REC SQUARED"] = np.square(wr_data[[('Total Counting Stats', 'rec')]])
wr_data["YARDS SQUARED"] = np.square(wr_data[[('Total Counting Stats', 'YARDS')]])
wr_data["YPR SQUARED"] = np.square(wr_data[[(('Total Counting Stats', 'YPR'))]])
wr_data["REC/G SQUARED"] = np.square(wr_data[[('Total Counting Stats', 'REC/g')]])
wr_data["YARDS DOMINATOR SQUARED"] = np.square(wr_data[[('Total Counting Stats', 'Yards Dominator')]])
small_value = wr_data[wr_data['RecYds/TmPatt']["AVG"] != 0]
small_value = small_value[[('RecYds/TmPatt', 'AVG')]].min()/2
wr_data["LOG RECYDS/TMPATT AVERAGE"] = np.log(np.add(wr_data[[('RecYds/TmPatt', 'AVG')]],small_value))
wr_data["RECYDS/TMPATT ABOVE TEAM AVERAGE (FIRST) SQUARED"] = np.square(wr_data[[('RecYds/TmPatt', 'First')]])
small_value = wr_data[wr_data['Dominator']["AVG"] != 0]
small_value = small_value[[('Dominator', 'AVG')]].min()/2
wr_data["LOG DOMINATOR AVG"] = np.log(np.add(wr_data[[('Dominator', 'AVG')]],small_value))
wr_data["E^X DOA BEST"] = np.exp(wr_data[[('DOa (Dom Over Average)', 'Best')]])
wr_data["MS YARDS AVG SQUARED"] = np.square(wr_data[[('MS Yards', 'AVG')]])
wr_data["YOA BEST SQUARED"] = np.square(wr_data[[('YOa (Yards Over Age Average)', 'Best')]])
wr_data["PPG ABOVE CONFERENCE EXPECTATION (LAST) SQUARED"] = np.square(wr_data[[('Context Scores', 'PPG Above conference expectation (Last Year)')]])
wr_data["AVERAGE SHARE ABOVE EXPECTATION SQUARED"] = np.square(wr_data[[('Context Scores', 'AVG S/EX (Yds Share Over Expectation)')]])
wr_data["TEAMMATE SCORE SQUARED"] = np.square(wr_data[[('Context Scores', 'TeamMate Score (TeamMate Over Expected)')]])
wr_data["COMBINE SHUTTLE SQUARED"] = np.square(wr_data[[('Combine', 'Shuttle')]])
wr_data["COMBINE WASS SQAURED"] = np.square(wr_data[[('Combine', 'WaSS')]])
wr_data["LOG DRAFT POSITION"] = np.log(wr_data[[('Unnamed: 4_level_0', 'DP')]])
wr_data["FINAL YEAR TEAM STRENGTH SQUARED"] = np.square(wr_data[[('Final Year Team Strength', 'Unnamed: 172_level_1')]])
wr_data["DominatorBest_Times_CombineShuttle"] = np.multiply(wr_data[[(('Dominator', 'Best'))]], wr_data[[(('Combine', 'Shuttle'))]])
wr_data["CombineShuttle_Times_DraftPosition"] = np.multiply(wr_data[[(('Combine', 'Shuttle'))]], wr_data[[('Unnamed: 4_level_0', 'DP')]])
wr_data["LOG CombineShuttle_Times_DraftPosition"] = np.log(wr_data[[("CombineShuttle_Times_DraftPosition","")]])
wr_data["DraftPositions_Times_BOA30"] = np.multiply(wr_data[[('Unnamed: 4_level_0', 'DP')]], wr_data[[('Breakout Ages', '>30%')]])
wr_data["DominatorBest_Times_BrokeOut30"] = np.multiply(wr_data[[(('Dominator', 'Best'))]], wr_data[[('Broke Out 30', 'Unnamed: 172_level_1')]])
wr_data["DraftPosition_Times_BrokeOut30"] = np.multiply(wr_data[[('Unnamed: 4_level_0', 'DP')]], wr_data[[('Broke Out 30', 'Unnamed: 172_level_1')]])
wr_data["COMBINE HEIGHT SQUARED"] = np.square(wr_data[[('Combine', 'height')]])
wr_data["DominatorBest_Times_CombineHeight"] = np.multiply(wr_data[[(('Dominator', 'Best'))]], wr_data[[("Combine","height")]])
wr_data["DraftPosition_Times_CombineHeight"] = np.multiply(wr_data[[('Unnamed: 4_level_0', 'DP')]], wr_data[[("Combine","height")]])
wr_data["LOG DraftPosition_Times_CombineHeight"] = np.log(wr_data[[("DraftPosition_Times_CombineHeight","")]])
wr_data["DOA AVG SQUARED"] = np.square(wr_data[[('DOa (Dom Over Average)', 'AVG')]])
wr_data["DominatorBest_Times_DoaAvg"] = np.multiply(wr_data[[(('Dominator', 'Best'))]], wr_data[[('DOa (Dom Over Average)', 'AVG')]])
wr_data["DominatorBest_Times_DoaAvg SQUARED"] = np.square(wr_data[[("DominatorBest_Times_DoaAvg","")]])
wr_data["CombineShuttle_Times_DoaAvg"] = np.multiply(wr_data[[(('Combine', 'Shuttle'))]], wr_data[[('DOa (Dom Over Average)', 'AVG')]])
wr_data["CombineShuttle_Times_DoaAvg SQUARED"] = np.square(wr_data[[("CombineShuttle_Times_DoaAvg","")]])
wr_data["CombineHeight_TimesDoaAvg"] = np.multiply(wr_data[[('Combine', 'height')]], wr_data[[('DOa (Dom Over Average)', 'AVG')]])
wr_data["CombineHeight_TimesDoaAvg SQUARED"] = np.square(wr_data[[("CombineHeight_TimesDoaAvg","")]])
wr_data["BOA30_Times_DoaAvg"] = np.multiply(wr_data[[('Breakout Ages', '>30%')]], wr_data[[('DOa (Dom Over Average)', 'AVG')]])
wr_data["BrokeOut30_Times_DoaAvg"] = np.multiply(wr_data[[('Broke Out 30', 'Unnamed: 172_level_1')]], wr_data[[('DOa (Dom Over Average)', 'AVG')]])

# print(wr_data.columns.tolist())

features_subset_transformation = [
('Total Counting Stats', 'AVG PPG'), ("AVGPPG SQUARED",""), ('Total Counting Stats', 'rec'), ("REC SQUARED",""), ('Total Counting Stats', 'YARDS'), ("YARDS SQUARED",""), ('Total Counting Stats', 'YPR'), ("YPR SQUARED",""), 
('Total Counting Stats', 'REC/g'), ("REC/G SQUARED",""), ('Total Counting Stats', 'Yards Dominator'), ("YARDS DOMINATOR SQUARED",""), ('RecYds/TmPatt', 'AVG'), ("LOG RECYDS/TMPATT AVERAGE",""), ('RecYds/TmPatt Above Team AVG', 'First'), 
("RECYDS/TMPATT ABOVE TEAM AVERAGE (FIRST) SQUARED",""), ('RecYds/TmPatt Above Team AVG', 'Best'), ('Dominator', 'Best'), ('Dominator', 'AVG'), ("LOG DOMINATOR AVG",""), ('DOa (Dom Over Average)', 'Best'), ("E^X DOA BEST",""), 
('MS Yards', 'Last'), ('MS Yards', 'AVG'), ("MS YARDS AVG SQUARED", ""), ('YOa (Yards Over Age Average)', 'Best'), ("YOA BEST SQUARED",""), ('Context Scores', 'PPG Above conference expectation (Last Year)'), ("PPG ABOVE CONFERENCE EXPECTATION (LAST) SQUARED",""),
('Context Scores', 'AVG S/EX (Yds Share Over Expectation)'), ("AVERAGE SHARE ABOVE EXPECTATION SQUARED",""), ('Context Scores', 'TeamMate Score (TeamMate Over Expected)'), ("TEAMMATE SCORE SQUARED",""), ('Combine', 'weight'), ('Combine', 'Shuttle'), 
("COMBINE SHUTTLE SQUARED",""), ('Combine', 'WaSS'), ("COMBINE WASS SQAURED",""), ('Unnamed: 4_level_0', 'DP'), ("LOG DRAFT POSITION",""), ('Final Year Team Strength', 'Unnamed: 170_level_1'), ("FINAL YEAR TEAM STRENGTH SQUARED",""), 
('Unnamed: 7_level_0', 'Age IN DRAFT YEAR'), ('Total Counting Stats', 'FINAL MS YARDS RK'), ('Breakout Ages', '>20%'), ('Breakout Ages', '>30%'), ('Unnamed: 3_level_0', 'DR'), ('Broke Out 30', 'Unnamed: 172_level_1'), ('Broke Out 20', 'Unnamed: 171_level_1')
]

features_trim_subset_transformation = [
('Dominator', 'Best'), ('DOa (Dom Over Average)', 'Best'), ("E^X DOA BEST",""), ('Combine', 'Shuttle'), ("COMBINE SHUTTLE SQUARED",""), ('Unnamed: 4_level_0', 'DP'), ("LOG DRAFT POSITION",""), ('Breakout Ages', '>30%'), ('Broke Out 30', 'Unnamed: 172_level_1')
]

features_remove_bad_coeffs = [
('Dominator', 'Best'), ('Combine', 'Shuttle'), ("COMBINE SHUTTLE SQUARED",""), ('Unnamed: 4_level_0', 'DP'), ("LOG DRAFT POSITION",""), ('Breakout Ages', '>30%'), ('Broke Out 30', 'Unnamed: 172_level_1')
]

features_best_plus_interactions = [
('Dominator', 'Best'), ('Combine', 'Shuttle'), ("LOG DRAFT POSITION",""), ('Breakout Ages', '>30%'), ("DominatorBest_Times_CombineShuttle",""),
("LOG CombineShuttle_Times_DraftPosition",""), ("DraftPosition_Times_BrokeOut30",""), ('Unnamed: 4_level_0', 'DP'), ("CombineShuttle_Times_DraftPosition",""), ('Broke Out 30', 'Unnamed: 172_level_1'),
("DominatorBest_Times_BrokeOut30",""), ("DraftPositions_Times_BOA30",""), ("COMBINE SHUTTLE SQUARED","") 
]

features_best_after_backwards_elimination = [
('Dominator', 'Best'), ('Combine', 'Shuttle'), ("LOG DRAFT POSITION",""), ('Breakout Ages', '>30%'), ("DominatorBest_Times_CombineShuttle",""),
("LOG CombineShuttle_Times_DraftPosition",""), ("DraftPosition_Times_BrokeOut30","")
]

features_best_plus_sdr_features = [
('Dominator', 'Best'), ('Combine', 'Shuttle'), ("LOG DRAFT POSITION",""), ('Breakout Ages', '>30%'), ("DominatorBest_Times_CombineShuttle",""), ("LOG CombineShuttle_Times_DraftPosition",""), ("DraftPosition_Times_BrokeOut30",""),  
('DOa (Dom Over Average)', 'AVG'),
("CombineShuttle_Times_DoaAvg SQUARED", ""), ("DominatorBest_Times_DoaAvg SQUARED",""), ("DraftPosition_Times_CombineHeight",""), ("DominatorBest_Times_DoaAvg",""), ("DOA AVG SQUARED", ""), ("CombineHeight_TimesDoaAvg SQUARED",""), 
("CombineShuttle_Times_DoaAvg",""), ("BrokeOut30_Times_DoaAvg",""), ("BOA30_Times_DoaAvg",""), ("DominatorBest_Times_CombineHeight", ""), ("COMBINE HEIGHT SQUARED",""), ('Combine', 'height'), ("LOG DraftPosition_Times_CombineHeight",""), 
("CombineHeight_TimesDoaAvg",""),
]

features_best = [
('Dominator', 'Best'), ('Combine', 'Shuttle'), ("LOG DRAFT POSITION",""), ('Breakout Ages', '>30%'), ("DominatorBest_Times_CombineShuttle",""),
("LOG CombineShuttle_Times_DraftPosition",""), ("DraftPosition_Times_BrokeOut30",""), ('DOa (Dom Over Average)', 'AVG'),
]

x = wr_data[features_best]
y = wr_data[('true_points', 'Unnamed: 176_level_1')]

results = sm.OLS(y,x).fit()
print(results.summary())

model = linear_model.LinearRegression(fit_intercept=False)
model.fit(x,y)
print(model.score(x,y))

# max_r2 = -289237589475947892739
# feature_to_eliminate = None
# for feature_left_out in features_best_plus_sdr_features:

	# print("we are leaving out %s" % str(feature_left_out))
# Years we are going to iterate through. All where receivers have played out the 3 years already.
yrs = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
result_df = pd.DataFrame()

# Initialize empty values for the Y^ model, Y^ baseline, and actual Y.

model_predicted = np.array([])
baseline_predicted = np.array([])
actual_values = np.array([])

# Iterate through each year
for yr in yrs:
	train = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"] != yr]
	test = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"] == yr]

	train_columns = train[features_best]

	X_train = np.array(train_columns.values.tolist())
	dc_train = np.array(train[[('logDR', 'Unnamed: 173_level_1')]].values.tolist())

	test_columns = test[features_best]

	X_test = np.array(test_columns.values.tolist())
	dc_test = np.array(test[[('logDR', 'Unnamed: 175_level_1')]].values.tolist())

	Y_train = np.array(train[[('true_points', 'Unnamed: 178_level_1')]].values.tolist())
	Y_test = np.array(test[[('true_points', 'Unnamed: 178_level_1')]].values.tolist())

	#standardize

	# scaler = StandardScaler()
	# X_train=scaler.fit_transform(X_train)
	# X_test = scaler.fit_transform(X_test)
	# dc_train = scaler.fit_transform(dc_train)
	# dc_test = scaler.fit_transform(dc_test)


	# Initialize Linear Regression Models
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

	# Append to our result dataframe to export later
	test = test[[('Unnamed: 0_level_0', 'Name'), ("Unnamed: 6_level_0", "Draft Year")]+features_best]
	# test["Model Projected Average PPR Points Per First 48 Games"] = inv_boxcox(Y_pred, value)
	test["Model Projected Average PPR Points Per First 48 Games"] = Y_pred
	test["Residual"] = np.abs(np.subtract(Y_pred, Y_test))
	result_df = result_df.append(test)

# Calculate total r^2 and total RMSE on overall predictions

mean = np.array([np.mean(actual_values) for i in range(actual_values.shape[0])])

dc_r_2 = r2_score(actual_values, baseline_predicted)
dc_rmse = mean_squared_error(actual_values, baseline_predicted, squared=False)
model_r_2 = 1-(1-r2_score(actual_values, model_predicted))*((len(model_predicted)-1)/(len(model_predicted)-X_train.shape[1]-1))
model_rmse = mean_squared_error(actual_values, model_predicted, squared=False)
mean_r_2 = r2_score(actual_values, mean)
mean_rmse = mean_squared_error(actual_values, mean, squared=False)

print("DEBUG: LINEAR MODEL R^2 for every draft class in our data %s" % model_r_2)
print("DEBUG: LINEAR MODEL RMSE for every draft class in our data %s" % model_rmse)
# print("DEBUG: SIMPLE MEAN R^2 for every draft class in our data %s" % mean_r_2)
# print("DEBUG: SIMPLE MEAN RMSE for every draft class in our data %s" % mean_rmse)
print("DEBUG: LINEAR DRAFTCAPITAL R^2 for every draft class in our data %s" % dc_r_2)
print("DEBUG: LINEAR DRAFTCAPITAL RMSE for every draft class in our data %s" % dc_rmse)

# 	if model_r_2 > max_r2:
# 		max_r2=model_r_2
# 		feature_to_eliminate = feature_left_out

# print("Best feature we found to leave out was %s with adj_r2 value of %s" % (str(feature_to_eliminate), str(max_r2)))



# # let's see what coefficients our model came up with

print("Features %s" % list(train_columns))
print("Weights %s" % list(model.coef_))
result_df.to_excel("/Users/ronakmodi/FF_ProspectModel/Results/wide_receiver_OLS_residuals.xlsx")


# graph residuals

# residuals = np.subtract(actual_values, model_predicted)

# plot = sns.distplot(residuals, hist=True, bins=50, kde=True, kde_kws={'linewidth':3})
# plt.title("residual graph for standard ols model")

# x=actual_values
# y=np.abs(residuals)

# plt.figure()
# plt.plot(x,y,'bo')
# x,indices = np.unique(x,return_index=True)
# y = [y[i] for i in indices]
# y, indices = np.unique(y,return_index=True)
# x = [x[i] for i in indices]
# model = SuperSmoother()
# try:
# 	model.fit(x,y,[i + .000001 for i in x])
# 	tfit = np.linspace(min(x),max(x),1000)
# 	yfit = model.predict(tfit)
# 	plt.plot(tfit,yfit,'-g',linewidth=3)
# except Exception as e:
# 	print(e)
# plt.title("absolute residual vs dependent variable - standard OLS")

# x=model_predicted
# y=np.abs(residuals)

# plt.figure()
# plt.plot(x,y,'bo')
# x,indices = np.unique(x,return_index=True)
# y = [y[i] for i in indices]
# y, indices = np.unique(y,return_index=True)
# x = [x[i] for i in indices]
# model = SuperSmoother()
# try:
# 	model.fit(x,y,[i + .000001 for i in x])
# 	tfit = np.linspace(min(x),max(x),1000)
# 	yfit = model.predict(tfit)
# 	plt.plot(tfit,yfit,'-g',linewidth=3)
# except Exception as e:
# 	print(e)
# plt.title("absolute residual vs predicted - standard OLS")
# plt.show()



# for yr in [2018,2019,2020]:
# 	train = wr_data
# 	test = wr_data_with_prospects[wr_data_with_prospects["Unnamed: 6_level_0"]["Draft Year"] == yr]
# 	train_columns = train[features_best]
# 	X_train = np.array(train_columns.values.tolist())
# 	test_columns = test[features_best]
# 	X_test = np.array(test_columns.values.tolist())

# 	Y_train = np.array(train[[('true_points', 'Unnamed: 177_level_1')]].values.tolist())

# 	scaler = StandardScaler()
# 	X_train=scaler.fit_transform(X_train)
# 	X_test = scaler.fit_transform(X_test)

# 	model.fit(X_train, Y_train)
# 	Y_pred = model.predict(X_test)

# 	test = test[[('Unnamed: 0_level_0', 'Name'), ("Unnamed: 6_level_0", "Draft Year"), ("Unnamed: 3_level_0", "DR"),  ('RecYds/TmPatt', 'AVG'), ('DOa (Dom Over Average)', 'AVG')]]
# 	test["Model Projected Average PPR Points Per First 48 Games"] = inv_boxcox(Y_pred, value)
# 	result_df = result_df.append(test)
















	

