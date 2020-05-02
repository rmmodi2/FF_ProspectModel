import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import helpers
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import normalize, Normalizer, StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer

wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx", header=[0,1])

print(wr_data.dtypes)

print("Total Columns: %s" % wr_data.shape[1])

correlation_matrix = wr_data.corr()

correlation_matrix.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_correlation_matrix.xlsx")


"""

"""