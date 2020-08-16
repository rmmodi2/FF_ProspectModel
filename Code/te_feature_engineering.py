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

te_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/te_data.xlsx", header=[0])

te_data = te_data[te_data["Draft Year"] >= 2010]

te_data = te_data[te_data["Draft Year"]!=2020]
te_data = te_data[(te_data["Draft Year"]!=2018) | (te_data["hit_within3years"] == 1)]
te_data = te_data[(te_data["Draft Year"]!=2019) | (te_data["hit_within3years"] == 1)]

using = ['DP', 'REC/g', 'WaSS']

# for f in using:
# 	sns.regplot(x=f, y='hit_within3years', data=te_data, logistic=True, n_boot=258)
# 	plt.show()

for i,f in enumerate(using):
	for f2 in using[(i+1):]:
		te_data[f+f2] = np.multiply(te_data[f],te_data[f2])
		sns.regplot(x=f+f2, y="hit_within3years", data=te_data, logistic=True, n_boot=228)
		plt.show()