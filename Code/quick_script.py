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
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import normalize, Normalizer, StandardScaler, FunctionTransformer, MinMaxScaler
import helpers
import matplotlib.pyplot as plt


wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/pahowdy_prospect_database.xlsx", sheet_name = "WR", header=[0,1], na_values=["-"])


f = [('Unnamed: 0_level_0', 'Name'), ('Dominator (Combined REC Yards Touchdown MS)', 18), ('Dominator (Combined REC Yards Touchdown MS)', 19), ('Dominator (Combined REC Yards Touchdown MS)', 20), ('Dominator (Combined REC Yards Touchdown MS)', 21), 
('Dominator (Combined REC Yards Touchdown MS)', 22), ('Dominator (Combined REC Yards Touchdown MS)', 23), ('MS REC Yards', 18), ('MS REC Yards', 19), ('MS REC Yards', 20), ('MS REC Yards', 21), ('MS REC Yards', 22), ('MS REC Yards', 23), 
('NFL Career Marks since 2000', '# of top  24 finishes')]

mod = wr_data[f]

mod[mod['NFL Career Marks since 2000']['# of top  24 finishes'] > 0].to_excel("age_adjusted_help_sheet.xlsx")