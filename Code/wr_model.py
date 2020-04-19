import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, preprocessing
from sklearn.linear_model import PoissonRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_poisson_deviance, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import normalize
import helpers
import matplotlib.pyplot as plt

# first we import the excel files with our nice data

filtered_columns = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_filtered_columns.xlsx", header=[0,1])
draft_capital_only = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_draft_capital_only.xlsx", header=[0,1])

print("DEBUG: Number of players in our model sample: %s" % filtered_columns.shape[0])
print("DEBUG: Number of players in our draft capital sample: %s" % draft_capital_only.shape[0])

# take the log of draft capital

filtered_columns.loc[:, ("Unnamed: 4_level_0", 'DP')] = np.log(filtered_columns["Unnamed: 4_level_0"]['DP'])
draft_capital_only.loc[:, ("Unnamed: 4_level_0", 'DP')] = np.log(draft_capital_only["Unnamed: 4_level_0"]['DP'])

#drop the columns we don't want in our model

filtered_columns = filtered_columns.drop(columns=[('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), ('Unnamed: 0_level_0', 'Name'), ('Unnamed: 1_level_0', 'School'), ('score_hdr', 'Score')])

# now we have to split it into an X array and a Y array

X_model = np.array(filtered_columns.values.tolist())
X_draftcap = np.array(draft_capital_only[[('Unnamed: 4_level_0', 'DP')]].values.tolist())
Y = np.array(draft_capital_only[[('score_hdr','Score')]].values.tolist())

kf = KFold(n_splits=10,shuffle=True)

# run our best models for just draft capital and our model.

draft_capital_only = linear_model.LinearRegression()

draft_cap_mse = 0
draft_cap_mae = 0
draft_cap_r2 = 0
draft_cap_d2 = 0
i=0
for train, test in kf.split(X_draftcap, Y):
	draft_capital_only.fit(X_draftcap[train], Y[train])
	y_pred = draft_capital_only.predict(X_draftcap[test])
	draft_cap_mse += mean_squared_error(Y[test], y_pred)
	draft_cap_mae += mean_absolute_error(Y[test], y_pred)
	draft_cap_r2 += r2_score(Y[test], y_pred)
	mask = y_pred > 0
	draft_cap_d2 += mean_poisson_deviance(Y[test][mask], y_pred[mask])
	i+=1

print("DEBUG: LINEAR Average R2 value for just draft capital was %s" % (draft_cap_r2/i))
print("DEBUG: LINEAR Average D2 value for just draft capital was %s" % (draft_cap_d2/i))
print("DEBUG: LINEAR Average MSE value for just draft capital %s" % (draft_cap_mse/i))
print("DEBUG: LINEAR Average MAE value for just draft capital %s" % (draft_cap_mae/i))

model = PoissonRegressor(alpha=1/(.75*X_model.shape[0]), max_iter=1000)

model_mse = 0
model_mae = 0
model_d2 = 0
i=0
for train, test in kf.split(X_model, Y):
	model.fit(X_model[train], Y[train].ravel())
	y_pred = model.predict(X_model[test])
	model_mse += mean_squared_error(Y[test], y_pred)
	model_mae += mean_absolute_error(Y[test], y_pred)
	model_d2 += mean_poisson_deviance(Y[test], y_pred)
	i+=1

print("DEBUG: POISSON Average D2 value for the model was %s" % (model_d2/i))
print("DEBUG: POISSON Average MSE value for the model was %s" % (model_mse/i))
print("DEBUG: POISSON Average MAE value for the model was %s" % (model_mae/i))

# let's graph it

model_x_train, model_x_test, model_draft_train, model_draft_test, model_y_train, model_y_test = train_test_split(X_model, X_draftcap, Y, test_size=.1, shuffle=True)

model.fit(model_x_train, model_y_train.ravel())
draft_capital_only.fit(model_draft_train, model_y_train)

model_pred = model.predict(model_x_test)
draft_cap_pred = draft_capital_only.predict(model_draft_test)

plt.plot(range(len(model_pred)), model_pred, "g--", range(len(model_pred)), model_y_test, "r^", range(len(model_pred)), draft_cap_pred, "b--")
plt.xlabel("index of observation")
plt.ylabel("score - model in green, draft capital in blue, actual in red")
plt.title("Predicting scores of Wide Receiver Prospects")
plt.grid(True)
plt.show()

# let's see what coefficients our model came up with
feature_name_weight = dict(zip(filtered_columns.columns, model.coef_))
print("DEBUG: weights for all of our features: %s" % feature_name_weight)














	

