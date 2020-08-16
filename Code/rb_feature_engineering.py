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

rb_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/rb_data.xlsx", header=[0])

rb_data = rb_data[rb_data["Draft Year"] >= 2010]

rb_data = rb_data[rb_data["Draft Year"]!=2020]
rb_data = rb_data[(rb_data["Draft Year"]!=2018) | (rb_data["hit_within3years"] == 1)]
rb_data = rb_data[(rb_data["Draft Year"]!=2019) | (rb_data["hit_within3years"] == 1)]

fwd_sel = ['DP', 'Final Year Team Offensive Strength', 'WaSS']


# find any transformations for features

# for f in fwd_sel:
# 	sns.regplot(x=f, y='hit_within3years', data=rb_data, logistic=True, n_boot=228)
# 	plt.show()

# find any interactions for features

for i,f in enumerate(fwd_sel):
	for f2 in fwd_sel[(i+1):]:
		rb_data[f+f2] = np.multiply(rb_data[f],rb_data[f2])
		sns.regplot(x=f+f2, y="hit_within3years", data=rb_data, logistic=True, n_boot=228)
		plt.show()