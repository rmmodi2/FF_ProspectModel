import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import helpers
import seaborn as sns
import statsmodels
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from supersmoother import SuperSmoother
from scipy.stats import norm
from scipy.signal import savgol_filter

wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx", header=[0])
wr_data = wr_data[wr_data["Draft Year"]!=2020]
wr_data = wr_data[wr_data["Draft Year"]!=2018]
wr_data = wr_data[wr_data["Draft Year"]!=2019]

true_points = wr_data['true_points'].to_numpy()


print("Median: %s" % np.median(true_points))
print("Standard Deviation: %s" % np.std(true_points))
print("Mean: %s" % np.mean(true_points))

# for f in to_be_squared:
# 	if f[1].split(":")[0] == "Unnamed":
# 		name = (str(f[0]+"_SQUARED"), "")
# 	if f[0].split(":")[0] == "Unnamed":
# 		name = (str(f[1]+"_SQUARED"), "")
# 	else:
# 		name = (str(f[0]+"_"+f[1]+"_SQUARED"), "")
# 	wr_data[name] = np.square(wr_data[[f]])

# for f in to_be_log:
# 	if f[1].split(":")[0] == "Unnamed":
# 		name = (str("LOG_"+f[0]), "")
# 	if f[0].split(":")[0] == "Unnamed":
# 		name = (str("LOG_"+f[1]), "")
# 	else:
# 		name = (str("LOG_"+f[0]+"_"+f[1]), "")
# 	wr_data[name] = np.log(wr_data[[f]])

print(wr_data.columns.tolist())

features = ['DR', 'DP', 'Age IN DRAFT YEAR', 'Years Played', 'G', 'AVG PPG', 'rec', 'YARDS', 'YPR', 'REC/g', 'FINAL MS YARDS RK', 'College Dominator Rating', 'Yards Dominator', 'Breakout Age >20%', 'Breakout Age >30%', 
'RecYds/TmPatt First', 'RecYds/TmPatt Best', 'RecYds/TmPatt Last', 'RecYds/TmPatt AVG', 'RecYds/TmPatt Above Team AVG First', 'RecYds/TmPatt Above Team AVG Best', 'RecYds/TmPatt Above Team AVG Last', 'RecYds/TmPatt Above Team AVG AVG', 'Dominator First', 
'Dominator Best', 'Dominator Last', 'Dominator AVG', 'DOa (Dom Over Average) First', 'DOa (Dom Over Average) Best', 'DOa (Dom Over Average) Last', 'DOa (Dom Over Average) AVG', 'MS Yards First', 'MS Yards Best', 'MS Yards Last', 'MS Yards AVG', 
'YOa (Yards Over Age Average) First', 'YOa (Yards Over Age Average) Best', 'YOa (Yards Over Age Average) Last', 'YOa (Yards Over Age Average) AVG', 'PPG Above conference expectation (Last Year)', 'AVG S/EX (Yds Share Over Expectation)', 
'Last S/EX (Yds Share Over Expectation)', 'TeamMate Score (TeamMate Over Expected)', 'BMI', '40 time', 'height', 'weight', 'Bench', 'Verticle', 'Broad', 'Shuttle', '3 Cone', 'WaSS', 'HaSS', 'Hand Size', 'Arm Length', 'Final Year Conference Strength', 
'Final Year Conference Defensive Strength', 'Final Year Team Strength', 'Final Year Team Offensive Strength', 'Broke Out 20', 'Broke Out 30', 'Yards Leader Final Year', 'punt_returns', 'kick_returns', 'kick_return_yards', 'punt_return_yards', 'kick_return_avg', 
'punt_return_avg', 'kick_return_td', 'punt_return_td', 'hc_tenure', 'oc_tenure', 'hc_retained', 'oc_retained', 'vacated_tgt_pct', 'vacated_rec_pct', 'vacated_yds_pct', 'Att', 'Cmp%', 'Yds', 'Y/A', 'AY/A', 'Y/C', 'Rate', 'NY/A', 'ANY/A', 'EXP']

fwd_sel_cont = ['DP', 'YOa (Yards Over Age Average) AVG', 'vacated_yds_pct', 'PPG Above conference expectation (Last Year)', 'punt_return_avg']

fwd_sel_cat = ['Breakout Age >30%']

using = ['DP', 'YOa (Yards Over Age Average) AVG', 'PPG Above conference expectation (Last Year)', 'hc_tenure', 'Breakout Age >20%', 'Last S/EX (Yds Share Over Expectation)', ]

# CDF
# sns.set_style("whitegrid")
# plot = sns.distplot(true_points, hist=False, kde=True, kde_kws={'linewidth':2, 'cumulative': True})
# plt.yticks(np.arange(0,1.05,.05))
# plt.xticks(np.arange(true_points.min(),true_points.max()+1,1))
# plt.show()

# # PDF
# sns.set_style("whitegrid")
# plot = sns.distplot(y, hist=False, kde=True, kde_kws={'linewidth':3})
# plt.yticks(np.arange(0,.2,.01))
# plt.xticks(np.arange(y.min(),y.max()+1,1))
# plt.show()

# results = sm.OLS(y,x).fit()

# Plot a curve of best fit / dot plot of each continuous indepdenent variable vs the dependent variable

# for feature in fwd_sel_cont:
# 	x = wr_data[[feature]]
# 	x = x.to_numpy().flatten()
# 	y = true_points.flatten()
# 	plt.plot(x,y,'o')
# 	lowess_est = helpers.lowess(x,y)
# 	plt.plot(lowess_est[:,0],lowess_est[:,1],'-',linewidth=3)
# 	plt.title(str(feature))
# 	plt.show()

# # Plot a seperate boxplot for each category in all categorical indepedent variables vs the dependent variable

# for feature in fwd_sel_cat:
# 	melted = pd.melt(wr_data,id_vars=[feature],value_vars=['true_points'])
# 	sns.boxplot(x=feature,y='value',data=melted)
# 	plt.show()

# Plot an interaction plot for variables that need it

# for i,f in enumerate(relation_trim_fwd_cont):
# 	for f2 in relation_trim_fwd_cat:
# 		x1 = wr_data[[f]].to_numpy().flatten()
# 		trace = wr_data[[f2]].to_numpy().flatten()
# 		y = true_points.flatten()
# 		fig = interaction_plot(x1,trace,y,xlabel=str(f),ylabel="PPR Points / 48 games",legendtitle=str(f2))
# 		plt.show()

# Plot x1*x2 effect for continuous variables that need it


# scaler=StandardScaler()
# tempdf = wr_data[fwd_sel_cont]
# tempdf = pd.DataFrame(data=scaler.fit_transform(tempdf), columns=fwd_sel_cont)
# for i,feature in enumerate(fwd_sel_cont):
# 	for j,feature2 in enumerate(fwd_sel_cont[i+1:]):
# 		x1 = wr_data[[feature]]
# 		x2 = wr_data[[feature2]]
# 		x = np.multiply(x1,x2).to_numpy().flatten()
# 		y = wr_data[['true_points']].to_numpy().flatten()
# 		plt.plot(x,y,'o')
# 		lowess_est = helpers.lowess(x,y)
# 		plt.plot(lowess_est[:,0],lowess_est[:,1],'-',linewidth=2)
# 		plt.title("points vs %s * %s" % (str(feature), str(feature2)))
# 		plt.show()

# Plot boxplot of continuous variables at each level of categorical variables for interaction

# for categorical in fwd_sel_cat:
# 	for feature in fwd_sel_cont:
# 		sns.boxplot(y=feature, x=categorical, data=wr_data)
# 		plt.show()


# Plot studentized residuals vs features that were eliminated

# ols_influence = OLSInfluence(results)
# studentized_deleted_residuals = ols_influence.get_resid_studentized_external().to_numpy().flatten()
# for deleted_feature in [f for f in features if f not in features_best_and_interpretable]:
# 	x = wr_data[[deleted_feature]].to_numpy().flatten()
# 	y = studentized_deleted_residuals
# 	plt.plot(x,y,'ro')
# 	plt.xlabel(str(deleted_feature))
# 	plt.ylabel("Studentized Deleted Residuals")
# 	x,indices = np.unique(x,return_index=True)
# 	y = [y[i] for i in indices]
# 	y, indices = np.unique(y,return_index=True)
# 	x = [x[i] for i in indices]
# 	model = SuperSmoother()
# 	try:
# 		model.fit(x,y,[i + .000001 for i in x])
# 		tfit = np.linspace(min(x),max(x),1000)
# 		yfit = model.predict(tfit)
# 		plt.plot(tfit,yfit,'-g',linewidth=3)
# 	except Exception as e:
# 		print(e)
# 	plt.show()


# LOGISTIC REGRESSION

# Plot p(success) vs bin 

# for i,f in enumerate(using):
# 	for f2 in using[i+1:]:
# 		print(f)
# 		print(f2)
# 		wr_data[f+f2] = np.multiply(wr_data[f],wr_data[f2])
# 		sns.regplot(x=f+f2, y='hit_within3years', data=wr_data, logistic=True, n_boot=448)
# 		plt.show()


# sns.regplot(x='hc_tenure', y='hit_within3years', data=wr_data, logistic=True, n_boot=448)
# plt.show()

"""
draft position + log(DP)
G
AVG PPG
rec + rec^2
yards
YPR + YPR^2
rec/g + rec/g^2
college dominator 
yards dominator
recyds/tmpatt first + x^2
recyds/tmpatt best + x^2
recyds/tmpatt last 
recyds/tmpatt avg 
recyds/tmpatt above tm avg best
recyds/tmpatt above tm avg last
recyds/tmpatt above tm avg avg
dominator first
dominator best 
dominator last
dominator avg
doa first
doa best + x^2 + e^x
doa last
doa avg + x^2
ms yards best
ms yards last
yoa first + yoa first x^2
yoa best
yoa last + e^x
yoa avg + x^2
context scores ppg above ex (last) + x^2 
context scores avg share above ex + x^2
context scores teammate score + x^2
combine BMI
combine weight 
combine shuttle + x^2
combine WASS + x^2
combine HASS
final year team strength + log(final year team strength)
DR
Age in Draft Year
Years Played
BOA 20
BOA 30
Final Year Conference Strength
Broke Out 20
Broke Out 30
"""






