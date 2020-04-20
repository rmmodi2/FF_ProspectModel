import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, preprocessing
from sklearn.linear_model import PoissonRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_poisson_deviance, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import normalize, Normalizer, StandardScaler
import helpers
import matplotlib.pyplot as plt

# first we import the excel files with our nice data

filtered_columns = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_filtered_columns.xlsx", header=[0,1])
draft_capital_only = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_draft_capital_only.xlsx", header=[0,1])

print("DEBUG: Number of players in our model sample: %s" % filtered_columns.shape[0])
print("DEBUG: Number of players in our draft capital sample: %s" % draft_capital_only.shape[0])

# drop the columns we don't want in our model, preprocess filtered_columns

model_columns = filtered_columns.drop(columns=[('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), ('Unnamed: 0_level_0', 'Name'), ('Unnamed: 1_level_0', 'School'), ("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3"), ('Breakout Ages', '>20%'), 
	('Dominator', 'BEST'), ('Context Scores', 'PPG Above conference expectation (Last Year)'), ('Context Scores', 'TeamMate Score'), ('MS Yards', 'BEST'), ('Combine', 'HaSS'), ("Unnamed: 6_level_0",'Draft Year'), ('Context Scores', 'Last S/EX'), 
	('Total Counting Stats', 'REC/g'), ('Total Counting Stats', 'Years Played')])

# now we have to split it into an X array and a Y array

X_model = np.array(model_columns.values.tolist())
X_draftcap = np.array(draft_capital_only[[('Unnamed: 4_level_0', 'DP')]].values.tolist())
Y = np.array(draft_capital_only[[("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3")]].values.tolist())

normalize(X_model,copy=False)

n_splits = 10
kf = KFold(n_splits=n_splits,shuffle=True)

# run our best models for just draft capital and our model.
model = linear_model.LinearRegression()

model_mae = 0
model_mse = 0
model_r2 = 0
max_r2 = -19238409327423
best_coef = []
i=0
for train, test in kf.split(X_model, Y):
	model.fit(X_model[train], Y[train])
	y_pred = model.predict(X_model[test])
	model_mse += mean_squared_error(Y[test], y_pred)
	model_mae += mean_absolute_error(Y[test], y_pred)
	tmp_r2 = r2_score(Y[test], y_pred)
	if tmp_r2 > max_r2:
		max_r2=tmp_r2
		best_coef=model.coef_
	model_r2+=tmp_r2
	i+=1

print("DEBUG: LINEAR Average R2 value for the model was %s" % (model_r2/i))
print("DEBUG: LINEAR BEST R2 value for the model was %s" % max_r2)
# print("DEBUG: LINEAR Average MAE value for the model was %s" % (model_mae/i))
# print("DEBUG: LINEAR Average MSE value for the model was %s\n" % (model_mse/i))

draft_capital_only = linear_model.LinearRegression()

draft_cap_mse = 0
draft_cap_mae = 0
draft_cap_r2 = 0
i=0
for train, test in kf.split(X_draftcap, Y):
	draft_capital_only.fit(X_draftcap[train], Y[train])
	y_pred = draft_capital_only.predict(X_draftcap[test])
	draft_cap_mse += mean_squared_error(Y[test], y_pred)
	draft_cap_mae += mean_absolute_error(Y[test], y_pred)
	draft_cap_r2 += r2_score(Y[test], y_pred)
	i+=1

print("DEBUG: LINEAR Average R2 value for just draft capital was %s" % (draft_cap_r2/i))
# print("DEBUG: LINEAR Average MAE value for just draft capital %s" % (draft_cap_mae/i))
# print("DEBUG: LINEAR Average MSE value for just draft capital %s\n" % (draft_cap_mse/i))

# let's see what coefficients our model came up with

print("Features %s" % list(model_columns))
print("Weights %s" % list(best_coef))

# now let's see how we did on one class itself

train = filtered_columns[filtered_columns["Unnamed: 6_level_0"]["Draft Year"] != 2014]
test = filtered_columns[filtered_columns["Unnamed: 6_level_0"]["Draft Year"] == 2014]

train_columns = train.drop(columns=[('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), ('Unnamed: 0_level_0', 'Name'), ('Unnamed: 1_level_0', 'School'), ("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3"),('Breakout Ages', '>20%'), 
	('Dominator', 'BEST'), ('Context Scores', 'PPG Above conference expectation (Last Year)'), ('Context Scores', 'TeamMate Score'), ('MS Yards', 'BEST'), ('Combine', 'HaSS'), ("Unnamed: 6_level_0",'Draft Year'), ('Context Scores', 'Last S/EX'), 
	('Total Counting Stats', 'REC/g'), ('Total Counting Stats', 'Years Played')])
X_train = np.array(train_columns.values.tolist())
test_columns = test.drop(columns=[('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), ('Unnamed: 0_level_0', 'Name'), ('Unnamed: 1_level_0', 'School'), ("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3"),('Breakout Ages', '>20%'), 
	('Dominator', 'BEST'), ('Context Scores', 'PPG Above conference expectation (Last Year)'), ('Context Scores', 'TeamMate Score'), ('MS Yards', 'BEST'), ('Combine', 'HaSS'), ("Unnamed: 6_level_0",'Draft Year'), ('Context Scores', 'Last S/EX'), 
	('Total Counting Stats', 'REC/g'), ('Total Counting Stats', 'Years Played')])
X_test = np.array(test_columns.values.tolist())

Y_train = np.array(train[[("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3")]].values.tolist())
Y_test = np.array(test[[("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3")]].values.tolist())

normalize(X_train,copy=False)
normalize(X_test,copy=False)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("DEBUG: R^2 for one draft class is %s" % r2_score(Y_test,Y_pred))

result = pd.DataFrame({"Name": test[(('Unnamed: 0_level_0', 'Name'))], "Predicted PPR PPG Years 1-3": Y_pred.ravel(), "Actual PPR ...": Y_test.ravel()})
result.sort_values(by="Predicted PPR PPG Years 1-3",inplace=True,ascending=False)
print(result)


# let's graph the model / draft capital / actual for a random train/test split

model_x_train, model_x_test, model_draft_train, model_draft_test, model_y_train, model_y_test = train_test_split(X_model, X_draftcap, Y, test_size=.1, shuffle=True)

normalize(model_x_train,copy=False)
normalize(model_x_test,copy=False)

model.fit(model_x_train, model_y_train.ravel())
draft_capital_only.fit(model_draft_train, model_y_train)

model_pred = model.predict(model_x_test)
draft_cap_pred = draft_capital_only.predict(model_draft_test)

plt.plot(range(len(model_pred)), model_pred, "g--", range(len(model_pred)), model_y_test, "r^", range(len(model_pred)), draft_cap_pred, "bo")
plt.xlabel("index of observation")
plt.ylabel("PPG - model in green, draft capital in blue, actual in red")
plt.title("Predicting PPR PPG of Wide Receiver Prospects (Years 1-3)")
plt.grid(True)
plt.show()














	

