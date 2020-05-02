import pandas as pd
import numpy as np
import math
from math import log
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import re

regex = re.compile('[^a-zA-Z]')

# This file simply is used for a helper functions that will regenerate the real PPR PPG to replace pahowdy's ppr ppg data with, and other stuff.

# First create a dict mapping the year to its fantasy finishes

years_avail = [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
fantasy_finishes_by_year = {}
for year in years_avail:
	df = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/%s_fantasy_finishes.xlsx" % year, header=[0,1])
	df.fillna(value=0,inplace=True)
	df[('Unnamed: 1_level_0', 'Player')] = df[('Unnamed: 1_level_0', 'Player')].map(lambda x: x.split("*")[0])
	fantasy_finishes_by_year[year] = df

years_avail = [2000,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
school_strength_by_year = {}
conf_strength_by_year = {}
for year in years_avail:
	df_conf = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/%s_cfb_conferences.xlsx" % year, header=[0,1])
	df_conf.replace(to_replace="Independent", value="Ind", inplace=True)
	df_conf.replace(to_replace="Southeastern Conference", value="SEC", inplace=True)
	df_conf.replace(to_replace="Big Ten Conference", value="Big Ten", inplace=True)
	df_conf.replace(to_replace="Big East Conference", value="Big East", inplace=True)
	df_conf.replace(to_replace="Big 12 Conference", value="Big 12", inplace=True)
	df_conf.replace(to_replace="Pacific-12 Conference", value="Pac-12", inplace=True)
	df_conf.replace(to_replace="Pacific-10 Conference", value="Pac-10", inplace=True)
	df_conf.replace(to_replace="American Athletic Conference", value="American", inplace=True)
	df_conf.replace(to_replace="Atlantic Coast Conference", value="ACC", inplace=True)
	df_conf.replace(to_replace="Western Athletic Conference", value="WAC", inplace=True)
	df_conf.replace(to_replace="Mountain West Conference", value="MWC", inplace=True)
	df_conf.replace(to_replace="Sun Belt Conference", value="Sun Belt", inplace=True)
	df_conf.replace(to_replace="Conference USA", value="CUSA", inplace=True)
	df_conf.replace(to_replace="Mid-American Conference", value="MAC", inplace=True)
	df_conf[('Unnamed: 1_level_0', 'Conference')] = df_conf[('Unnamed: 1_level_0', 'Conference')].apply(lambda x: regex.sub('',x).lower())
	df_school = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/%s_cfb_schools.xlsx" % year, header=[0,1])
	df_school.replace(to_replace="BYU", value="Brigham Young", inplace=True)
	df_school[('Unnamed: 1_level_0', 'School')] = df_school[('Unnamed: 1_level_0', 'School')].apply(lambda x: regex.sub('',x).lower())
	school_strength_by_year[year] = df_school
	conf_strength_by_year[year] = df_conf

# This function returns a single column vector that matches the wr_data exactly

def generate_ppr_ppg_by_48(wr_data):
	column = []
	for idx, row in wr_data.iterrows():
		name = regex.sub('',row[('Unnamed: 0_level_0', 'Name')]).lower()
		pos = row[('Unnamed: 5_level_0', 'NFL POS')]
		draft_year = row[('Unnamed: 6_level_0', 'Draft Year')]
		draft_year = int(draft_year)
		points=0
		for years in [draft_year, draft_year+1, draft_year+2]:
			if years > 2019:
				break
			df = fantasy_finishes_by_year[years]
			df[('Unnamed: 1_level_0', 'Player')] = df[('Unnamed: 1_level_0', 'Player')].apply(lambda x: regex.sub('',x).lower())
			row = df[df['Unnamed: 1_level_0']['Player'] == name]
			row = row[row['Unnamed: 3_level_0']['FantPos'] == pos]
			if len(row) == 0:
				continue
			elif len(row) > 1:
				print("DEBUG: There were multiple matching rows for %s, drafted in year %s" % (name, draft_year))
				print("DEBUG: These are the matching rows:")
				print(row)
				value = input("Tell me which index to keep: (None, 0, 1, 2 etc.)\n")
				if value == "None":
					print("DEBUG: None of the matching rows were actually of the player")
					continue
				value = int(value)
				row = row.iloc[[value]]
			points+=float(row[("Fantasy", "PPR")])
		column.append((points/48))
	return column	

def generate_conference_strength(wr_data):
	column = []
	for idx, row in wr_data.iterrows():
		conference = row[('Unnamed: 2_level_0', "Conference")]
		conference = regex.sub('',str(conference)).lower()
		draft_year = int(row[('Unnamed: 6_level_0', "Draft Year")])
		df_conferences = conf_strength_by_year[draft_year-1]
		match = df_conferences[df_conferences['Unnamed: 1_level_0']["Conference"] == conference]
		if len(match) == 0:
			worst_rk = df_conferences[('Unnamed: 0_level_0', "Rk")].max()
			column.append(log(worst_rk+1))
			continue
			match = df_conferences[df_conferences['Unnamed: 1_level_0']["Conference"] == value]
		column.append(log(match[('Unnamed: 0_level_0', "Rk")]))
	return column

def generate_team_strength(wr_data):
	column = []
	for idx, row in wr_data.iterrows():
		team = row[('Unnamed: 1_level_0', "School")]
		team = regex.sub('',str(team).lower())
		team_year = int(row[('Unnamed: 6_level_0', "Draft Year")]) - 1
		df_school = school_strength_by_year[team_year]
		match = df_school[df_school['Unnamed: 1_level_0']["School"] == team]
		if len(match) == 0:
			worst_rk = df_school[('Unnamed: 0_level_0', "Rk")].max()
			column.append(log(worst_rk+1))
			continue
			match = df_school[df_school['Unnamed: 1_level_0']["School"] == value]
		column.append(log(match[('Unnamed: 0_level_0', "Rk")]))
	return column

# This function returns a single column vector that matches the wr_data exactly

def generate_yds_att_oaa(wr_data):
	ages = [18,19,20,21,22,23]
	column = []
	mean = {}
	for age in ages:
		mean[age] = wr_data[("Y/TmPatt (Rec Yards Per Team Pass Attempt)", age)].mean()
	for idx, row in wr_data.iterrows():
		avg = 0
		matches = 0
		for age in ages:
			yds_per_att = float(row[("Y/TmPatt (Rec Yards Per Team Pass Attempt)",age)])
			if np.isnan(yds_per_att):
				continue
			else:
				matches+=1
				avg+=(yds_per_att-mean[age])
		if matches > 0:
			column.append((avg/matches))
		else:
			column.append(float("nan"))
	return column

def generate_msyds_above_avg_successful(wr_data):
	truepts_90percentile = wr_data[('true_points')].quantile(q=.9,interpolation='lower')
	successful_wrs = wr_data[wr_data['true_points'] >= truepts_90percentile]
	successful_wrs_avg_msyards = successful_wrs[('MS Yards', 'AVG')].mean()
	column = []
	for idx,row in wr_data.iterrows():
		avg_msyards = float(row[('MS Yards', 'AVG')])
		column.append(avg_msyards-successful_wrs_avg_msyards)
	return column

def generate_yds_att_above_avg_successful(wr_data):
	truepts_90percentile = wr_data[('true_points')].quantile(q=.9,interpolation='lower')
	successful_wrs = wr_data[wr_data['true_points'] >= truepts_90percentile]
	successful_wrs_avg_recyards_att = successful_wrs[('RecYds/TmPatt', 'AVG')].mean()
	column = []
	for idx,row in wr_data.iterrows():
		avg_recyds_att = float(row[('RecYds/TmPatt', 'AVG')])
		column.append(avg_recyds_att-successful_wrs_avg_recyards_att)
	return column

def generate_doa_above_avg_successful(wr_data):
	truepts_90percentile = wr_data[('true_points')].quantile(q=.90,interpolation='lower')
	successful_wrs = wr_data[wr_data['true_points'] >= truepts_90percentile]
	successful_wrs_avg_recyards_att = successful_wrs[('DOa (Dom Over Average)', 'Doa (AVG)')].mean()
	column = []
	for idx,row in wr_data.iterrows():
		avg_recyds_att = float(row[('DOa (Dom Over Average)', 'Doa (AVG)')])
		column.append(avg_recyds_att-successful_wrs_avg_recyards_att)
	return column































