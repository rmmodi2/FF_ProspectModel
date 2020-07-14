import sys

import pandas as pd
import numpy as np
import math
from math import log
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, balanced_accuracy_score, recall_score, log_loss
from sklearn.model_selection import KFold
import re
import statsmodels.api as sm

regex = re.compile('[^a-zA-Z]')
outside_d1 = [regex.sub('',"kentucky wesleyan".lower()), regex.sub('',"Hillsdale".lower()), regex.sub('',"Malone".lower())]

# This file simply is used for a helper functions that will regenerate the real PPR PPG to replace pahowdy's ppr ppg data with, and other stuff.

# First create a dict mapping the year to its fantasy finishes

fantasy_finishes_by_year = {}
school_strength_by_year = {}
conf_strength_by_year = {}
nfl_passing_offense = {}

def create_fantasy_finishes_by_year():
	years_avail = [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
	for year in years_avail:
		df = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/%s_fantasy_finishes.xlsx" % year, header=[0,1])
		df.fillna(value=0,inplace=True)
		df[('Unnamed: 1_level_0', 'Player')] = df[('Unnamed: 1_level_0', 'Player')].map(lambda x: x.split("*")[0])
		fantasy_finishes_by_year[year] = df

def create_school_and_conf_info():
	years_avail = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
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
		def remove_conf_div(x):
			if len(x.split("(")) > 1:
				return x.split("(")[0]
			else:
				return x
		df_school[('Unnamed: 2_level_0', 'Conf')] = df_school[('Unnamed: 2_level_0', 'Conf')].apply(remove_conf_div)
		df_school[('Unnamed: 2_level_0', 'Conf')] = df_school[('Unnamed: 2_level_0', 'Conf')].apply(lambda x: regex.sub('',x).lower())
		school_strength_by_year[year] = df_school
		conf_strength_by_year[year] = df_conf

def create_passing_offense_info():
	years_avail = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
	replace = {
	'Indianapolis Colts': 'IND', 'New England Patriots': 'NWE', 'Green Bay Packers': 'GNB', 'New Orleans Saints': 'NOR', 'Los Angeles Rams': 'LAR', 'New York Giants': 'NYG', 'Cleveland Browns':'CLE', 'Philadelphia Eagles':'PHI', 
	'Buffalo Bills':'BUF', 'Jacksonville Jaguars':'JAX', 'St. Louis Rams':'STL', 'Seattle Seahawks':'SEA', 'Tennessee Titans':'TEN', 'Arizona Cardinals':'ARI', 'Chicago Bears':'CHI', 'Detroit Lions':'DET', 'Houston Texans':'HOU', 'Dallas Cowboys':'DAL', 
	'Kansas City Chiefs':'KAN', 'Miami Dolphins':'MIA', 'Pittsburgh Steelers':'PIT', 'Washington Redskins':'WAS', 'Baltimore Ravens':'BAL', 'New York Jets':'NYJ', 'Minnesota Vikings':'MIN', 'Denver Broncos':'DEN', 'San Francisco 49ers':'SFO', 
	'Los Angeles Chargers':'LAC', 'San Diego Chargers':'SDG', 'Oakland Raiders':'OAK', 'Tampa Bay Buccaneers':'TAM', 'Atlanta Falcons':'ATL', 'Carolina Panthers':'CAR', 'Cincinnati Bengals':'CIN'
	}
	for year in years_avail:
		df = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/%s_nfl_passing_offense.xlsx" % year, header=[0])
		df=df.replace(replace)
		nfl_passing_offense[year] = df

def create_landing_spot_info():
	global landing_spot
	landing_spot = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/landing_spot.xlsx", header=[0])
	landing_spot["Name"] = landing_spot["Name"].apply(lambda x: regex.sub('',x).lower())

def create_recruiting_grades():
	global recruiting_grades_data
	recruiting_grades_data = pd.read_csv("/Users/ronakmodi/FF_ProspectModel/Data/recruiting_rankings.csv", header=[0])

def create_sp_team_data():
	global sp_team_data
	sp_team_data = pd.read_csv("/Users/ronakmodi/FF_ProspectModel/Data/cfb_team_SPRankings.csv", header=[0])
	sp_team_data.replace(to_replace="BYU", value="Brigham Young", inplace=True)
	sp_team_data["team"] = sp_team_data["team"].apply(lambda x: regex.sub('',x).lower())

def create_sp_conf_data():
	global sp_conf_data
	sp_conf_data = pd.read_csv("/Users/ronakmodi/FF_ProspectModel/Data/cfb_conf_SPRankings.csv", header=[0])
	sp_conf_data.replace(to_replace="FBS Independents", value="Ind", inplace=True)
	sp_conf_data.replace(to_replace="American Athletic", value="American", inplace=True)
	sp_conf_data.replace(to_replace="Conference USA", value="CUSA", inplace=True)
	sp_conf_data.replace(to_replace="Mid-American", value="MAC", inplace=True)
	sp_conf_data.replace(to_replace="Mountain West", value="MWC", inplace=True)
	sp_conf_data["conference"] = sp_conf_data["conference"].apply(lambda x: regex.sub('',x).lower())

def create_vacated_stats():
	global receiving_data_00_19
	receiving_data_00_19 = pd.read_csv("/Users/ronakmodi/FF_ProspectModel/Data/receiving_data_00_19.csv", header=[0])
	receiving_data_00_19["Player"] = receiving_data_00_19["Player"].apply(lambda x: regex.sub('',x).lower())

def create_return_data():
	global punt_return_data_00_19
	punt_return_data_00_19 = pd.read_csv("/Users/ronakmodi/FF_ProspectModel/Data/college_punt_return_data_00_19.csv", header=[0])
	punt_return_data_00_19["Player"] = punt_return_data_00_19["Player"].apply(lambda x: regex.sub('',x).lower())
	punt_return_data_00_19["School"] = punt_return_data_00_19["School"].apply(lambda x: regex.sub('',x).lower())
	global kick_return_data_00_19
	kick_return_data_00_19 = pd.read_csv("/Users/ronakmodi/FF_ProspectModel/Data/college_kick_return_data_00_19.csv", header=[0])
	kick_return_data_00_19["Player"] = kick_return_data_00_19["Player"].apply(lambda x: regex.sub('',x).lower())
	kick_return_data_00_19["School"] = kick_return_data_00_19["School"].apply(lambda x: regex.sub('',x).lower())

def create_coaching_history():
	global coaching_history
	coaching_history = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/coaching_history.xlsx")

def generate_ppr_ppg_by_48(wr_data):
	create_fantasy_finishes_by_year()
	column = []
	success_1 = []
	success_2 = []
	hit_col = []
	for idx, row in wr_data.iterrows():
		success_total = False
		success_pg = False
		name = regex.sub('',row[('Unnamed: 0_level_0', 'Name')]).lower()
		pos = row[('Unnamed: 5_level_0', 'NFL POS')]
		draft_year = row[('Unnamed: 6_level_0', 'Draft Year')]
		draft_year = int(draft_year)
		hit_col.append(int(int(row[("NFL Career Marks since 2000", "# of top  24 finishes")]) > 0))
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
				print("DEBUG: There were multiple matching rows for %s in year %s who was drafted in year %s" % (name, years, draft_year))
				print("DEBUG: These are the matching rows:")
				print(row)
				value = input("Tell me which index to keep: (None, 0, 1, 2 etc.)\n")
				if value == "None":
					print("DEBUG: None of the matching rows were actually of the player")
					continue
				value = int(value)
				row = row.iloc[[value]]
			season_pts = float(row[("Fantasy", "PPR")])
			games = float(row[("Games", "G")])
			posrank = int(row[("Fantasy", "PosRank")])
			points+=season_pts
			success_total = bool(success_total or season_pts>=160)
			success_pg = bool(success_total or (season_pts/games)>=10)
		column.append((points/48))
		success_1.append(int(success_total))
		success_2.append(int(success_pg))
	return column, success_1, success_2, hit_col	

def generate_conference_strength(wr_data):
	create_school_and_conf_info()
	create_sp_conf_data()
	column = []
	conf_def_str = []
	for idx, row in wr_data.iterrows():
		conference = row[('Unnamed: 2_level_0', "Conference")]
		conference = regex.sub('',str(conference)).lower()
		team = row[('Unnamed: 1_level_0', "School")]
		team = regex.sub('',str(team).lower())
		draft_year = int(row[('Unnamed: 6_level_0', "Draft Year")])
		df_conferences = conf_strength_by_year[draft_year-1]
		match = df_conferences[df_conferences['Unnamed: 1_level_0']["Conference"] == conference]
		if len(match) == 0:
			worst_rk = df_conferences[('SRS', "SRS")].min()
			if team in outside_d1:
				column.append(float(worst_rk)*3)
			else:
				column.append(float(worst_rk)*2)
		else:
			column.append(float(match[('SRS', "SRS")]))
		df_school = school_strength_by_year[draft_year-1]
		worst_rk = df_school[('SRS','DSRS')].min()
		match = df_school[df_school['Unnamed: 2_level_0']["Conf"] == conference]
		if len(match) == 0:
			if team in outside_d1:
				conf_def_str.append(float(worst_rk)*2)
			else:
				conf_def_str.append(float(worst_rk))
		else:
			conf_def_str.append(float(match[('SRS','DSRS')].mean()))
	return np.array(column), np.array(conf_def_str)

def generate_team_strength(wr_data):
	create_school_and_conf_info()
	create_sp_team_data()
	column = []
	offense = []
	for idx, row in wr_data.iterrows():
		team = row[('Unnamed: 1_level_0', "School")]
		team = regex.sub('',str(team).lower())
		team_year = int(row[('Unnamed: 6_level_0', "Draft Year")]) - 1
		years_played = int(row[('Total Counting Stats', 'Years Played')])
		df_school = school_strength_by_year[team_year]
		match = df_school[df_school['Unnamed: 1_level_0']["School"] == team]
		if len(match) == 0:
			worst_rk = df_school[('SRS', "SRS")].min()
			worst_off_rk = df_school[('SRS', "OSRS")].min()
			if team in outside_d1:
				column.append(float(worst_rk)*3)
				offense.append(float(worst_off_rk)*3)
			else:
				column.append(float(worst_rk)*2)
				offense.append(float(worst_off_rk)*2)
		else:
			column.append(float(match[('SRS', "SRS")]))
			offense.append(float(match[('SRS', "OSRS")]))
	return np.array(column), np.array(offense)

def generate_landing_spot():
	create_fantasy_finishes_by_year()
	wr_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/pahowdy_prospect_database.xlsx", sheet_name = "WR", header=[0,1], na_values=["-"])
	wr_data = wr_data[wr_data["Unnamed: 6_level_0"]["Draft Year"]!="2021 PROSPECT"]
	draft_data = pd.read_excel("/Users/ronakmodi/FF_ProspectModel/Data/drafted_players.xlsx", header=[0,1])
	draft_data[('Unnamed: 3_level_0', 'Player')] = draft_data[('Unnamed: 3_level_0', 'Player')].apply(lambda x: regex.sub('',x).lower())
	name_arr = []
	team_arr = []
	for idx, row in wr_data.iterrows():
		draft_year = int(row[('Unnamed: 6_level_0', "Draft Year")])
		if draft_year < 2020:
			fantasy_finish = fantasy_finishes_by_year[draft_year]
		fantasy_finish[('Unnamed: 1_level_0', 'Player')] = fantasy_finish[('Unnamed: 1_level_0', 'Player')].apply(lambda x: regex.sub('',x).lower())
		name_arr.append(str(row[('Unnamed: 0_level_0', 'Name')]))
		name = regex.sub('',row[('Unnamed: 0_level_0', 'Name')]).lower()
		pos = str(row[('Unnamed: 5_level_0', 'NFL POS')])
		match = draft_data[draft_data['Unnamed: 3_level_0']['Player'] == name]
		match = match[match['Unnamed: 4_level_0']['Pos'] == pos]
		match = match[match['Unnamed: 0_level_0']['Year'] == draft_year]
		if len(match) == 0:
			if draft_year < 2020:
				match = fantasy_finish[fantasy_finish['Unnamed: 1_level_0']['Player'] == name]
				match = match[match['Unnamed: 3_level_0']['FantPos'] == pos]
			if len(match) == 0:
				value = input("Which team did %s who was drafted in year %s was drafted by?\n" % (name, draft_year))
				team_arr.append(str(value))
				continue
			elif len(match) > 1:
				print("DEBUG: There were multiple matching rows for %s who was drafted in year %s" % (name, draft_year))
				print("DEBUG: These are the matching rows:")
				print(match)
				value = input("Tell me which index to keep: (None, 0, 1, 2 etc.)\n")
				if value == "None":
					value = input("DEBUG: None of the matching rows were actually of the player, who was they drafted by?\n")
					team_arr.append(value)
					continue
				value = int(value)
				match = match.iloc[[value]]
			tm = match[('Unnamed: 2_level_0', 'Tm')].values[0]
			team_arr.append(str(tm))
			continue
		elif len(match) > 1:
			print("DEBUG: There were multiple matching rows for %s who was drafted in year %s" % (name, draft_year))
			print("DEBUG: These are the matching rows:")
			print(match)
			value = input("Tell me which index to keep: (None, 0, 1, 2 etc.)\n")
			if value == "None":
				value = input("DEBUG: None of the matching rows were actually of the player, who was they drafted by?\n")
				team_arr.append(value)
				continue
			value = int(value)
			match = match.iloc[[value]]
		tm = match[('Unnamed: 6_level_0', 'Tm')].values[0]
		team_arr.append(str(tm))
	print(name_arr)
	print(team_arr)
	ret = pd.DataFrame({"Name":name_arr, "Team":team_arr})
	ret.to_excel("/Users/ronakmodi/FF_ProspectModel/Data/landing_spot.xlsx")

def generate_passing_offense_info(wr_data):
	create_passing_offense_info()
	create_landing_spot_info()
	ret_dict = {}
	for idx, row in wr_data.iterrows():
		name = str(row[('Unnamed: 0_level_0', 'Name')])
		name = regex.sub('',row[('Unnamed: 0_level_0', 'Name')]).lower()
		draft_year = int(row[('Unnamed: 6_level_0', "Draft Year")])
		passing_offense_df = nfl_passing_offense[draft_year-1]
		match_team = landing_spot[landing_spot["Name"] == name]
		if len(match_team) > 1:
			print("there are two options for %s who was drafted in year %s" % (name, draft_year))
			print(match_team)
			value = input("tell me which index to keep\n")
			value = int(value)
			match_team = match_team.iloc[[value]]
		try:
			match_team = str(match_team["Team"].values[0])
		except Exception as e:
			print(name)
		match_pass_offense = passing_offense_df[passing_offense_df["Tm"] == match_team]
		if len(match_pass_offense) == 0:
			value = input("There was no match for %s in the year %s, please give me one\n" % (match_team, draft_year-1))
			match_pass_offense = passing_offense_df[passing_offense_df["Tm"] == str(value)]
		for stat in ["Att", "Cmp%", "Yds", "Y/A", "AY/A", "Y/C", "Rate", "NY/A", "ANY/A", "EXP"]:
			print(float(match_pass_offense[stat].values[0]))
			ret_dict.setdefault(stat, []).append(match_pass_offense[stat].values[0])
	for k in ret_dict.keys():
		print(len(ret_dict[k]))
	return ret_dict

def generate_recruiting_data(wr_data):
	create_recruiting_grades()
	recruiting_grades_data["name"] = recruiting_grades_data["name"].apply(lambda x: regex.sub('',x).lower())
	column_stars = []
	column_rating = []
	for idx,row in wr_data.iterrows():
		name = str(row[('Unnamed: 0_level_0', 'Name')])
		name = regex.sub('',name).lower()
		draft_year = int(row[('Unnamed: 6_level_0', "Draft Year")])
		school = str(row[('Unnamed: 1_level_0', 'School')])
		position = str(row[('Unnamed: 5_level_0', 'NFL POS')])
		matching_recruiting = recruiting_grades_data[recruiting_grades_data["name"]==name]
		matching_recruiting = matching_recruiting[matching_recruiting["year"]<draft_year]
		if len(matching_recruiting) == 0:
			column_stars.append(None)
			column_rating.append(None)
			continue
			# value = input("there were no matches for %s in the recruiting rankings, please type a value or just press enter\n" % name)
			# if value:
			# 	value = value.split(",")
			# 	column_stars.append(int(value[0]))
			# 	column_rating.append(float(value[1]))
			# else:
			# 	column_stars.append(None)
			# 	column_rating.append(None)
			# continue
		if len(matching_recruiting) > 1:
			print(matching_recruiting)
			value = input("There were multiple matches for %s, drafted in %s, went to school %s.\n" % (name, draft_year, school))
			if not value:
				column_stars.append(None)
				column_rating.append(None)
				continue
			if str(value) == "None":
				value = input("Please give me his stars and rank:\n")
				value = value.split(",")
				column_stars.append(int(value[0]))
				column_rating.append(float(value[1]))
				continue
			index = int(value)
			matching_recruiting = matching_recruiting.iloc[[index]]
		column_stars.append(int(matching_recruiting["stars"]))
		column_rating.append(float(matching_recruiting["rating"]))
	return column_stars, column_rating


def generate_vacated_stats(wr_data):
	create_landing_spot_info()
	create_passing_offense_info()
	create_vacated_stats()
	v_tgt_pct=[]
	v_rec_pct=[]
	v_yds_pct=[]
	for idx, row in wr_data.iterrows():
		name = str(row[('Unnamed: 0_level_0', 'Name')])
		name = regex.sub('',name).lower()
		draft_year = int(row[('Unnamed: 6_level_0', "Draft Year")])
		if draft_year == 2020:
			v_tgt_pct.append(None)
			v_rec_pct.append(None)
			v_yds_pct.append(None)
			continue
		landing_spot_match = landing_spot[landing_spot["Name"] == name]
		if len(landing_spot_match) > 1:
			print(landing_spot_match)
			v = input("Pick one of the rows from above, name: {}, drafted {}\n".format(name,draft_year))
			landing_spot_match = landing_spot_match.iloc[[v]]
		team = landing_spot_match.get("Team").item()
		prev_year = receiving_data_00_19[receiving_data_00_19["Tm"] == team]
		prev_year = prev_year[prev_year["Year"] == draft_year-1]
		if len(prev_year) == 0:
			newteam = input("Give the correct abbrev for team {} in year {}\n".format(team,draft_year-1))
			prev_year = receiving_data_00_19[receiving_data_00_19["Tm"] == newteam]
			prev_year = prev_year[prev_year["Year"] == draft_year-1]
		next_year = receiving_data_00_19[receiving_data_00_19["Tm"] == team]
		next_year = next_year[next_year["Year"] == draft_year]
		if len(next_year) == 0:
			nteam = input("Give the correct abbrev for team {} in year {}\n".format(team,draft_year))
			next_year = receiving_data_00_19[receiving_data_00_19["Tm"] == nteam]
			next_year = next_year[next_year["Year"] == draft_year]
		prev_passing_offense_df = nfl_passing_offense[draft_year-1]
		prev_passing_stats = prev_passing_offense_df[prev_passing_offense_df["Tm"]==team]
		if len(prev_passing_stats) == 0:
			new_team = input("Give the correct abbrev for team {} in year {}\n".format(team,draft_year-1))
			prev_passing_stats = prev_passing_offense_df[prev_passing_offense_df["Tm"]==newteam]
		prev_year_total_tgt = prev_passing_stats["Att"].item()
		prev_year_total_rec = prev_passing_stats["Cmp"].item()
		prev_year_total_yds = prev_passing_stats["Yds"].item()
		vacated_tgt = 0
		vacated_rec = 0
		vacated_yds = 0
		for idx,row in prev_year.iterrows():
			player_name = row["Player"]
			if player_name in next_year["Player"].values:
				continue
			else:
				vacated_tgt+=int(row["Tgt"])
				vacated_rec+=int(row["Rec"])
				vacated_yds+=int(row["Yds"])
		tgtshare=0
		recshare=0
		ydsshare=0
		for idx, row in next_year.iterrows():
			player_name = row["Player"]
			if player_name in prev_year["Player"].values or player_name == name:
				continue
			else:
				player_last_year = receiving_data_00_19[receiving_data_00_19["Player"] == player_name]
				player_last_year = player_last_year[player_last_year["Year"] == draft_year-1]
				if len(player_last_year) > 1:
					print(player_last_year)
					age = None
					for _,row in player_last_year.iterrows():
						if age:
							if row["Age"] == age:
								continue
							else:
								v = input("Need help please pick one row for {} in year {} who next year played for {}\n".format(player_name,draft_year-1,team))
								player_last_year = player_last_year.iloc[[v]]
								break
						else:
							age = row["Age"]
				if len(player_last_year) == 0:
					continue
				tshare,rshare,yshare=0,0,0
				niter=0
				for _,row in player_last_year.iterrows():
					prev_team_name = row["Tm"]
					prev_team_stats = nfl_passing_offense[draft_year-1]
					prev_team_stats = prev_team_stats[prev_team_stats["Tm"] == prev_team_name]
					if len(prev_team_stats) == 0:
						prev_team_name = input("Need a team name for %s in year %s\n" % (prev_team_name, draft_year-1))
					prev_team_stats = prev_team_stats[prev_team_stats["Tm"] == prev_team_name]
					tshare += float(row["Tgt"]) / float(prev_team_stats["Att"])
					rshare += float(row["Rec"]) / float(prev_team_stats["Cmp"])
					yshare += float(row["Yds"]) / float(prev_team_stats["Yds"])
					niter+=1
				tgtshare+=tshare/niter
				recshare+=rshare/niter
				ydsshare+=yshare/niter
		v_tgt_pct.append(float(vacated_tgt)/float(prev_year_total_tgt) - tgtshare)
		v_rec_pct.append(float(vacated_rec)/float(prev_year_total_rec) - recshare)
		v_yds_pct.append(float(vacated_yds)/float(prev_year_total_yds) - ydsshare)
	return np.array(v_tgt_pct),np.array(v_rec_pct),np.array(v_yds_pct)

def generate_return_data(wr_data):
	create_return_data()
	ret_dict = {"punt_returns": [], "kick_returns": [], "kick_return_yards": [], "punt_return_yards": [], "kick_return_avg": [], "punt_return_avg": [], "kick_return_td": [], "punt_return_td": []}
	for idx, row in wr_data.iterrows():
		name = str(row[('Unnamed: 0_level_0', 'Name')])
		name = regex.sub('',name).lower()
		draft_year = int(row[('Unnamed: 6_level_0', "Draft Year")])
		school = str(row[('Unnamed: 1_level_0', 'School')])
		school = regex.sub('',str(school).lower())
		player_punt_match = punt_return_data_00_19[(punt_return_data_00_19.Player==name) & (punt_return_data_00_19.Year < draft_year) & (punt_return_data_00_19.School == school)]
		punt_returns = 0
		punt_return_yards = 0
		punt_return_td = 0
		punt_return_avg = 0
		if len(player_punt_match) > 0:
			for year in player_punt_match.Year.unique():
				specific_year_match = player_punt_match[player_punt_match["Year"] == year]
				specific_year_match = specific_year_match[specific_year_match["G"] == specific_year_match["G"].max()]
				punt_returns+=specific_year_match["Punt Ret"].item()
				punt_return_yards+=specific_year_match["Punt Ret Yds"].item()
				punt_return_td+=specific_year_match["Punt Ret TD"].item()
			if punt_returns:
				punt_return_avg = float(punt_return_yards)/float(punt_returns) 
		ret_dict["punt_returns"].append(punt_returns)
		ret_dict["punt_return_yards"].append(punt_return_yards)
		ret_dict["punt_return_avg"].append(punt_return_avg)
		ret_dict["punt_return_td"].append(punt_return_td)
		player_kick_match = kick_return_data_00_19[(kick_return_data_00_19.Player==name) & (kick_return_data_00_19.Year < draft_year) & (kick_return_data_00_19.School == school)]
		kick_returns = 0
		kick_return_yards = 0
		kick_return_td = 0
		kick_return_avg = 0
		if len(player_kick_match) > 0:
			for year in player_kick_match.Year.unique():
				specific_year_match = player_kick_match[player_kick_match["Year"] == year]
				specific_year_match = specific_year_match[specific_year_match["G"] == specific_year_match["G"].max()]
				kick_returns+=specific_year_match["Kick Ret"].item()
				kick_return_yards+=specific_year_match["Kick Ret Yds"].item()
				kick_return_td+=specific_year_match["Kick Ret TD"].item()
			if kick_returns:
				kick_return_avg = float(kick_return_yards)/float(kick_returns)
		ret_dict["kick_returns"].append(kick_returns)
		ret_dict["kick_return_yards"].append(kick_return_yards)
		ret_dict["kick_return_avg"].append(kick_return_avg)
		ret_dict["kick_return_td"].append(kick_return_td)
	return ret_dict

def generate_coaching_data(wr_data):
	create_coaching_history()
	create_landing_spot_info()
	ret_dict = {"hc_tenure":[], "oc_tenure":[], "hc_retained":[], "oc_retained":[]}
	for idx, row in wr_data.iterrows():
		name = str(row[('Unnamed: 0_level_0', 'Name')])
		name = regex.sub('',name).lower()
		draft_year = int(row[('Unnamed: 6_level_0', "Draft Year")])
		landing_spot_match = landing_spot[landing_spot["Name"] == name]
		if len(landing_spot_match) > 1:
			print(landing_spot_match)
			v = input("Pick one of the rows from above, name: {}, drafted {}\n".format(name,draft_year))
			landing_spot_match = landing_spot_match.iloc[[v]]
		print(landing_spot_match.get("Team"))
		team = landing_spot_match.get("Team").item()
		coaching_history_match = coaching_history[(coaching_history["Team"] == team) & (coaching_history["Year"] == draft_year)]
		if len(coaching_history_match) == 0:
			v = input("Please give the correct abbreviation for {} in the year {}\n".format(team,draft_year))
			coaching_history_match = coaching_history[(coaching_history["Team"] == v) & (coaching_history["Year"] == draft_year)]
		ret_dict["hc_tenure"].append(coaching_history_match["Cumulative HC Years"].item())
		ret_dict["oc_tenure"].append(coaching_history_match["Cumulative OC Years"].item())
		ret_dict["hc_retained"].append(coaching_history_match["Retained HC"].item())
		ret_dict["oc_retained"].append(coaching_history_match["Retained OC"].item())
	return ret_dict


def cross_validation(features, data, y, model=None, quantile=None, max_iter=1000, p_tol=1e-6, logistic=False):

	is_statsmodel = (model is None)

	yrs = [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
	result_df = pd.DataFrame()

	# Initialize empty values for the Y^ model, Y^ baseline, and actual Y.

	model_predicted = np.array([])
	actual_values = np.array([])
	if logistic:
		model_proba = np.array([])

	# Iterate through each year
	for yr in yrs:
		train = data[data["Draft Year"] != yr]
		test = data[data["Draft Year"] == yr]

		train_columns = train[features]

		X_train = np.array(train_columns.values.tolist())

		test_columns = test[features]

		X_test = np.array(test_columns.values.tolist())

		Y_train = np.array(train[[y]].values.tolist())
		Y_test = np.array(test[[y]].values.tolist())

		res=None
		if is_statsmodel:
			model = sm.QuantReg(Y_train,train_columns)
			res = model.fit(q=quantile,max_iter=max_iter,p_tol=p_tol)
			Y_pred = model.predict(res.params,exog=test_columns)
		else:
			model.fit(X_train, Y_train.ravel())
			Y_pred = model.predict(X_test)
			if logistic:
				Y_proba = model.predict_proba(X_test)
				Y_proba = [sample[1] for sample in Y_proba]

		# Append predictions
		model_predicted = np.append(model_predicted, Y_pred)
		if logistic:
			model_proba = np.append(model_proba,Y_proba)
		actual_values = np.append(actual_values, Y_test)

		# Append to our result dataframe to export later
		test = test[['Name', "Draft Year"]+features]
		# test["Model Projected Average PPR Points Per First 48 Games"] = inv_boxcox(Y_pred, value)
		if logistic:
			test["Model"] = Y_proba
			test["Model_Class"] = Y_pred
		test["Actual"] = Y_test
		if not logistic:
			test["Model"] = Y_pred
			test["Residual"] = np.abs(np.subtract(Y_pred.flatten(), Y_test.flatten()))
		result_df = result_df.append(test)

	# Calculate total r^2 and total RMSE on overall predictions, or if logistic balanced_accuracy and sensitivity
	if logistic:
		model_r_2 = balanced_accuracy_score(actual_values, model_predicted)
		model_rmse = recall_score(actual_values, model_predicted)
	else:
		model_r_2 = 1-(1-r2_score(actual_values, model_predicted))*((len(model_predicted)-1)/(len(model_predicted)-X_train.shape[1]-1))
		model_rmse = mean_squared_error(actual_values, model_predicted, squared=False)

	return model_rmse, model_r_2, result_df, res

def forward_stepwise_selection(possible_features, data, y, model=None, quantile=None, max_iter=1000, p_tol=1e-6, logistic=False):
	"""
	Feature selection procedure, returns best features
	"""
	feature_list = []
	if logistic:
		curr_min_rmse = -2342343243243
	else:
		curr_min_rmse = 2342343243243
	r2_for_best = None
	while True:
		if logistic:
			best_iter_rmse = -2342343243242
		else:
			best_iter_rmse = 2342343243242
		best_iter_r2 = None
		best_f = None
		for feature in possible_features:
			if feature in feature_list:
				print("skipping feature %s, it is already used" % str(feature))
				continue
			print("testing feature %s used" % str(feature))
			if model:
				if logistic:
					rmse, r2, _, _ = cross_validation(feature_list+[feature], data, y, model=model, logistic=True)
				else:
					rmse, r2, _, _ = cross_validation(feature_list+[feature], data, y, model=model)
			else:
				rmse, r2, _, _ = cross_validation(feature_list+[feature], data, y, quantile=quantile,max_iter=max_iter, p_tol=p_tol)
			if (not logistic) and rmse < best_iter_rmse:
				best_iter_rmse = rmse
				best_iter_r2 = r2
				best_f = feature
			if logistic and rmse > best_iter_rmse:
				best_iter_rmse = rmse
				best_iter_r2 = r2
				best_f = feature
		if logistic:
			print("best feature in this iteration was %s with recall score %s and balanced_accuracy %s" % (best_f, best_iter_rmse, best_iter_r2))
		else:
			print("best feature in this iteration was %s with rmse %s and r2 %s" % (best_f, best_iter_rmse, best_iter_r2))
		if (not logistic) and best_iter_rmse < curr_min_rmse:
			feature_list.append(best_f)
			curr_min_rmse = best_iter_rmse
			r2_for_best = best_iter_r2
		elif logistic and best_iter_rmse > curr_min_rmse:
			feature_list.append(best_f)
			curr_min_rmse = best_iter_rmse
			r2_for_best = best_iter_r2
		else:
			break
	return feature_list, curr_min_rmse, r2_for_best

def backwards_stepwise_selection(possible_features, data, y, model=None, quantile=None, max_iter=1000, p_tol=1e-6, logistic=False):
	"""
	Feature selection procedure, returns best features
	"""
	feature_list = possible_features
	if logistic:
		curr_min_rmse = -24932473294327489327423
	else:
		curr_min_rmse = 24932473294327489327423
	r2_for_best = None
	while True:
		if logistic:
			best_iter_rmse = -2342343243242
		else:	
			best_iter_rmse = 2342343243242
		best_iter_r2 = None
		best_f = None
		for feature in feature_list:
			print("testing feature %s removed" % str(feature))
			if model:
				if logistic:
					rmse, r2, _, _ = cross_validation(feature_list+[feature], data, y, model=model, logistic=True)
				else:
					rmse, r2, _, _ = cross_validation(feature_list+[feature], data, y, model=model)
			else:
				rmse, r2, _, _ = cross_validation([j for j in feature_list if j != feature], data, y, quantile=quantile,max_iter=max_iter, p_tol=p_tol)
			if (not logistic) and rmse < best_iter_rmse:
				best_iter_rmse = rmse
				best_iter_r2 = r2
				best_f = feature
			elif logistic and rmse > best_iter_rmse:
				best_iter_rmse = rmse
				best_iter_r2 = r2
				best_f = feature
		print("best feature in this iteration was %s with rmse %s and r2 %s" % (best_f, best_iter_rmse, best_iter_r2))
		if (not logistic) and best_iter_rmse < curr_min_rmse:
			curr_min_rmse = best_iter_rmse
			r2_for_best = best_iter_r2
			feature_list = [i for i in feature_list if i != best_f]
		elif logistic and best_iter_rmse > curr_min_rmse:
			curr_min_rmse = best_iter_rmse
			r2_for_best = best_iter_r2
			feature_list = [i for i in feature_list if i != best_f]
		else:
			break
	return feature_list, curr_min_rmse, r2_for_best

def lowess(x,y,fraction=.2):
	x,indices = np.unique(x,return_index=True)
	y = [y[i] for i in indices]
	return sm.nonparametric.lowess(y,x,frac=fraction)

































