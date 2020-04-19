import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


#constants
wr_max_years = 15

# Helper Functions

def get_total_years(row, df):
	name = row["Unnamed: 0_level_0"]["Name"]
	year_drafted = row["Unnamed: 6_level_0"]["Draft Year"]
	matching_rows = df.loc[df['Player'] == name]
	if matching_rows.empty:
		# case where there are no matching rows, return 2019 - year in row, maximum from const.
		print("DEBUG (18): actual years: %s" % (2019-int(year_drafted) + 1))
		return min(wr_max_years, (2019-int(year_drafted) + 1))
	elif len(matching_rows) > 1:
		# case where there are multiple matching rows, have user pick one :)
		print("DEBUG: There were multiple matching rows for %s, drafted in year %s" % (name, year_drafted))
		print("DEBUG: These are the matching rows:")
		print(matching_rows)
		value = input("Tell me which index to keep: (None, 0, 1, 2 etc.)\n")
		if value == "None":
			print("DEBUG: None of the matching rows were actually of the player. He is still playing")
			print("DEBUG: actual years: %s" % (2019-int(year_drafted) + 1))
			return min(wr_max_years, 2019-int(year_drafted) + 1)
		value = int(value)
		matching_rows = matching_rows.iloc[[value]]
	# happy path, exactly one result
	print("DEBUG (33): actual years: %s" % (int(matching_rows["Year"]) - int(year_drafted) + 1))
	return (int(matching_rows["Year"]) - int(year_drafted) + 1)



