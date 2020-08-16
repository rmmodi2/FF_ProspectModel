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

# show us all the columns that are available

print(wr_data.columns.tolist())

# first remove the 2021 prospects

wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!="2021 PROSPECT"]

# now we need to make the data a little cleaner

# make udfa an 8th draft round

wr_data.replace(to_replace="UDFA",value=np.nan,inplace=True)
wr_data[('Unnamed: 3_level_0', 'DR')].fillna(value=8, inplace=True)
wr_data[('Unnamed: 4_level_0', 'DP')].fillna(value=257, inplace=True)

wr_data[('Unnamed: 3_level_0', 'DR')] = wr_data[('Unnamed: 3_level_0', 'DR')].apply(lambda x: 9-x)
wr_data[('Unnamed: 4_level_0', 'DP')] = wr_data[('Unnamed: 4_level_0', 'DP')].apply(lambda x: 258-x)

wr_data[('Unnamed: 7_level_0', 'Age IN DRAFT YEAR')] = wr_data[('Unnamed: 7_level_0', 'Age IN DRAFT YEAR')].apply(lambda x: 29-x)
wr_data[('Total Counting Stats', 'Years Played')] = wr_data[('Total Counting Stats', 'Years Played')].apply(lambda x: 6-x)
wr_data[('Total Counting Stats', 'FINAL MS YARDS RK')] = wr_data[('Total Counting Stats', 'FINAL MS YARDS RK')].apply(lambda x: 19-x)
wr_data[('Total Counting Stats', 'College Dominator Rating')] = wr_data[('Total Counting Stats', 'College Dominator Rating')].multiply(100)
wr_data[('Total Counting Stats', 'Yards Dominator')] = wr_data[('Total Counting Stats', 'Yards Dominator')].multiply(100)
wr_data[('RecYds/TmPatt Above Team AVG', 'First')] = wr_data[('RecYds/TmPatt Above Team AVG', 'First')].add(abs(wr_data[('RecYds/TmPatt Above Team AVG', 'First')].min())+.01)
wr_data[('RecYds/TmPatt Above Team AVG', 'Best')] = wr_data[('RecYds/TmPatt Above Team AVG', 'Best')].add(abs(wr_data[('RecYds/TmPatt Above Team AVG', 'Best')].min())+.01)
wr_data[('RecYds/TmPatt Above Team AVG', 'Last')] = wr_data[('RecYds/TmPatt Above Team AVG', 'Last')].add(abs(wr_data[('RecYds/TmPatt Above Team AVG', 'Last')].min())+.01)
wr_data[('RecYds/TmPatt Above Team AVG', 'AVG')] = wr_data[('RecYds/TmPatt Above Team AVG', 'AVG')].add(abs(wr_data[('RecYds/TmPatt Above Team AVG', 'AVG')].min())+.01)
wr_data[('DOa (Dom Over Average)', 'First')] = wr_data[('DOa (Dom Over Average)', 'First')].add(abs(wr_data[('DOa (Dom Over Average)', 'First')].min())+.01)
wr_data[('DOa (Dom Over Average)', 'First')] = wr_data[('DOa (Dom Over Average)', 'First')].multiply(100)
wr_data[('DOa (Dom Over Average)', 'Best')] = wr_data[('DOa (Dom Over Average)', 'Best')].add(abs(wr_data[('DOa (Dom Over Average)', 'Best')].min())+.01)
wr_data[('DOa (Dom Over Average)', 'Best')] = wr_data[('DOa (Dom Over Average)', 'Best')].multiply(100)
wr_data[('DOa (Dom Over Average)', 'Last')] = wr_data[('DOa (Dom Over Average)', 'Last')].add(abs(wr_data[('DOa (Dom Over Average)', 'Last')].min())+.01)
wr_data[('DOa (Dom Over Average)', 'Last')] = wr_data[('DOa (Dom Over Average)', 'Last')].multiply(100)
wr_data[('DOa (Dom Over Average)', 'AVG')] = wr_data[('DOa (Dom Over Average)', 'AVG')].add(abs(wr_data[('DOa (Dom Over Average)', 'AVG')].min())+.01)
wr_data[('DOa (Dom Over Average)', 'AVG')] = wr_data[('DOa (Dom Over Average)', 'AVG')].multiply(100)
wr_data[('YOa (Yards Over Age Average)', 'First')] = wr_data[('YOa (Yards Over Age Average)', 'First')].add(abs(wr_data[('YOa (Yards Over Age Average)', 'First')].min())+.01)
wr_data[('YOa (Yards Over Age Average)', 'First')] = wr_data[('YOa (Yards Over Age Average)', 'First')].multiply(100)
wr_data[('YOa (Yards Over Age Average)', 'Best')] = wr_data[('YOa (Yards Over Age Average)', 'Best')].add(abs(wr_data[('YOa (Yards Over Age Average)', 'Best')].min())+.01)
wr_data[('YOa (Yards Over Age Average)', 'Best')] = wr_data[('YOa (Yards Over Age Average)', 'Best')].multiply(100)
wr_data[('YOa (Yards Over Age Average)', 'Last')] = wr_data[('YOa (Yards Over Age Average)', 'Last')].add(abs(wr_data[('YOa (Yards Over Age Average)', 'Last')].min())+.01)
wr_data[('YOa (Yards Over Age Average)', 'Last')] = wr_data[('YOa (Yards Over Age Average)', 'Last')].multiply(100)
wr_data[('YOa (Yards Over Age Average)', 'AVG')] = wr_data[('YOa (Yards Over Age Average)', 'AVG')].add(abs(wr_data[('YOa (Yards Over Age Average)', 'AVG')].min())+.01)
wr_data[('YOa (Yards Over Age Average)', 'AVG')] = wr_data[('YOa (Yards Over Age Average)', 'AVG')].multiply(100)
wr_data[('Context Scores', 'PPG Above conference expectation (Last Year)')] = wr_data[('Context Scores', 'PPG Above conference expectation (Last Year)')].add(abs(wr_data[('Context Scores', 'PPG Above conference expectation (Last Year)')].min())+.01)
wr_data[('Context Scores', 'AVG S/EX (Yds Share Over Expectation)')] = wr_data[('Context Scores', 'AVG S/EX (Yds Share Over Expectation)')].add(abs(wr_data[('Context Scores', 'AVG S/EX (Yds Share Over Expectation)')].min())+.01)
wr_data[('Context Scores', 'AVG S/EX (Yds Share Over Expectation)')] = wr_data[('Context Scores', 'AVG S/EX (Yds Share Over Expectation)')].multiply(100)
wr_data[('Context Scores', 'Last S/EX (Yds Share Over Expectation)')] = wr_data[('Context Scores', 'Last S/EX (Yds Share Over Expectation)')].add(abs(wr_data[('Context Scores', 'Last S/EX (Yds Share Over Expectation)')].min())+.01)
wr_data[('Context Scores', 'Last S/EX (Yds Share Over Expectation)')] = wr_data[('Context Scores', 'Last S/EX (Yds Share Over Expectation)')].multiply(100)
wr_data[('Context Scores', 'TeamMate Score (TeamMate Over Expected)')] = wr_data[('Context Scores', 'TeamMate Score (TeamMate Over Expected)')].add(abs(wr_data[('Context Scores', 'TeamMate Score (TeamMate Over Expected)')].min())+.01)
wr_data[('Context Scores', 'TeamMate Score (TeamMate Over Expected)')] = wr_data[('Context Scores', 'TeamMate Score (TeamMate Over Expected)')].multiply(100)
wr_data[('Combine', '40 time')] = wr_data[('Combine', '40 time')].apply(lambda x: wr_data[('Combine', '40 time')].max() - x)
wr_data[('Combine', 'Shuttle')] = wr_data[('Combine', 'Shuttle')].apply(lambda x: wr_data[('Combine', 'Shuttle')].max() - x)
wr_data[('Combine', '3 Cone')] = wr_data[('Combine', '3 Cone')].apply(lambda x: wr_data[('Combine', '3 Cone')].max() - x)

# make non-cfb conferences NA

wr_data.replace(to_replace="Non-CFB",value=np.nan,inplace=True)
wr_data.replace(to_replace="UNK",value=np.nan,inplace=True)
wr_data["Final Year Conference Strength"], wr_data["Final Year Conference Defensive Strength"] = helpers.generate_conference_strength(wr_data)
wr_data["Final Year Conference Strength"] = wr_data["Final Year Conference Strength"].add(abs(wr_data["Final Year Conference Strength"].min())+.1)
wr_data["Final Year Conference Defensive Strength"] = wr_data["Final Year Conference Defensive Strength"].add(abs(wr_data["Final Year Conference Defensive Strength"].min())+.1)


wr_data["Final Year Team Strength"], wr_data["Final Year Team Offensive Strength"] = helpers.generate_team_strength(wr_data)
wr_data["Final Year Team Strength"] = wr_data["Final Year Team Strength"].add(abs(wr_data["Final Year Team Strength"].min())+.1)
wr_data["Final Year Team Offensive Strength"] = wr_data["Final Year Team Offensive Strength"].add(abs(wr_data["Final Year Team Offensive Strength"].min())+.1)

# now make <2 yrs of data all nans

wr_data.replace(to_replace="<2 Yrs of Data",value=np.nan,inplace=True)

# make ms yards avg & rec yds/tm att nans their last ms yards

wr_data[('MS Yards', 'AVG')].fillna(wr_data[('MS Yards', 'Last')], inplace=True)
wr_data[('MS Yards', 'First')] = wr_data[('MS Yards', 'First')].multiply(100)
wr_data[('MS Yards', 'Best')] = wr_data[('MS Yards', 'Best')].multiply(100)
wr_data[('MS Yards', 'Last')] = wr_data[('MS Yards', 'Last')].multiply(100)
wr_data[('MS Yards', 'AVG')] = wr_data[('MS Yards', 'AVG')].multiply(100)

# make dominator avg same as dominator last if it is nan

wr_data[('Dominator', 'AVG')].fillna(wr_data[('Dominator', 'Last')], inplace=True)
wr_data[('Dominator','First')] = wr_data[('Dominator','First')].multiply(100)
wr_data[('Dominator','Best')] = wr_data[('Dominator','Best')].multiply(100)
wr_data[('Dominator','Last')] = wr_data[('Dominator','Last')].multiply(100)
wr_data[('Dominator','AVG')] = wr_data[('Dominator','AVG')].multiply(100)

# all nans for ppr pts is 0

wr_data[("NFL Career Marks since 2000","AVG PPG (PPR) Season 1-3")].fillna(value=0, inplace=True)

# fill no breakout age with a high number, create log scaled breakout age

wr_data["Broke Out 20"] = wr_data[('Breakout Ages', '>20%')].isna().apply(lambda x: int((not x)))
wr_data["Broke Out 30"] = wr_data[('Breakout Ages', '>30%')].isna().apply(lambda x: int((not x)))

wr_data["Yards Leader Final Year"] = wr_data[('Total Counting Stats', 'FINAL MS YARDS RK')].apply(lambda x: int(int(x) == 1))

wr_data[('Breakout Ages', '>20%')].fillna(value=25, inplace=True)
wr_data[('Breakout Ages', '>30%')].fillna(value=25, inplace=True)
wr_data[('Breakout Ages', '>20%')] = wr_data[('Breakout Ages', '>20%')].apply(lambda x: 26-x)
wr_data[('Breakout Ages', '>30%')] = wr_data[('Breakout Ages', '>30%')].apply(lambda x: 26-x)

# wr_data["recruiting_stars"], wr_data["recruiting_rating"] = helpers.generate_recruiting_data(wr_data)

# fill remaining empty columns with the mean

wr_data.fillna(wr_data.mean(), inplace=True)

# sort by name PRIOR to calculating true points

wr_data.sort_values(by=[('Unnamed: 0_level_0', 'Name')], inplace=True)
wr_data["true_points"], wr_data["broke_100pts"], wr_data["broke_10ppg"], wr_data["hit"], wr_data["hit_within3years"] = helpers.generate_ppr_ppg_by_48(wr_data)

for _, (k, v) in enumerate(helpers.generate_return_data(wr_data).items()):
	wr_data[k] = v

for _, (k,v) in enumerate(helpers.generate_coaching_data(wr_data).items()):
	wr_data[k] = v

wr_data["vacated_tgt_pct"], wr_data["vacated_rec_pct"], wr_data["vacated_yds_pct"] = helpers.generate_vacated_stats(wr_data)
wr_data["vacated_tgt_pct"] = wr_data["vacated_tgt_pct"].add(abs(wr_data["vacated_tgt_pct"].min())+.01)
wr_data["vacated_tgt_pct"] = wr_data["vacated_tgt_pct"].multiply(100)
wr_data["vacated_rec_pct"] = wr_data["vacated_rec_pct"].add(abs(wr_data["vacated_rec_pct"].min())+.01)
wr_data["vacated_rec_pct"] = wr_data["vacated_rec_pct"].multiply(100)
wr_data["vacated_yds_pct"] = wr_data["vacated_yds_pct"].add(abs(wr_data["vacated_yds_pct"].min())+.01)
wr_data["vacated_yds_pct"] = wr_data["vacated_yds_pct"].multiply(100)


ret_dict = helpers.generate_passing_offense_info(wr_data)
for k in ret_dict.keys():
	wr_data[k] = np.array(ret_dict[k])

wr_data["EXP"] = wr_data["EXP"].add(abs(wr_data["EXP"].min())+1)

# convert to excel, save.

wr_data.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/wr_data_truepts.xlsx")