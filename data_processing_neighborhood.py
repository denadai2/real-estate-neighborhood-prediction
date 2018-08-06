import pandas as pd
import numpy as np
import math
import glob
import argparse
import psycopg2


def weighted_avg_and_std(values, weights):
	"""
	Return the weighted average and standard deviation.

	values, weights -- Numpy ndarrays with the same shape.
	"""
	average = np.average(values, weights=weights)
	variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
	return average, math.sqrt(variance)


# calculate total size
def total_size(vector):
	return np.sum(vector)


def get_weights(vector):
	""" Calculate the vector weights
	:param vector: Positive vector
	:return: Vector weights
	:raise: TypeError if negative total size
	"""
	ts = total_size(vector)
	if ts <= 0:
		raise TypeError('Vector must be positive')
	else:
		return np.true_divide(vector, ts)


def is_heavy_industry(ateco_code):
	return (5 <= ateco_code <= 9) | (19 <= ateco_code <= 30) | (36 <= ateco_code <= 39) | (42 <= ateco_code <= 43)


def is_shop_activity(ateco_code):
	return (45 <= ateco_code <= 47) | (94 <= ateco_code <= 96) | (ateco_code == 56)


def is_creative_industry(ateco_code):
	return (58 <= ateco_code <= 63) | (73 <= ateco_code <= 74) | (90 <= ateco_code <= 91) | (ateco_code == 71)


def make_argument_parser():
	"""
	Creates an ArgumentParser to read the options for this script from
	sys.argv
	:return:
	"""
	parser = argparse.ArgumentParser(
		description="Housing parameters"
	)
	return parser


def main():
	parser = make_argument_parser()
	args = parser.parse_args()
	print("PARAMETERS", args)

	con = psycopg2.connect(database="dsaa", user="nadai", host="localhost")

	df = pd.read_sql_query('select geoid, shape_area as census_area FROM census_areas_onfocus', con=con, index_col='geoid')

	comp_df = pd.read_csv('data/companies.csv', dtype={'geoid', str})
	comp_df = comp_df.set_index('geoid')
	assert comp_df.index.is_unique

	# Land value
	land_df = pd.read_csv('data/land_value.csv', dtype={'geoid', str})
	land_df = land_df.set_index('geoid')
	assert land_df.index.is_unique

	path = r'data/census/industry'
	allFiles = glob.glob(path + "/*.txt")
	list_ = []
	for file_ in allFiles:
		temp_df = pd.read_csv(file_, index_col=None, header=0, encoding="ISO-8859-1", delimiter=';')
		list_.append(temp_df)
	frame = pd.concat(list_)
	frame['sez2011'] = frame['PROCOM'].astype(str) + frame['NSEZ'].astype(str).str.zfill(7)
	frame['sez2011'] = frame.sez2011.astype(str).str.zfill(13)
	frame = frame.set_index('sez2011')

	frame = frame.rename(columns={'NUM_UNITA': 'num_companies', 'ADDETTI': 'employees'})
	frame['ATECO3'] = frame['ATECO3'].astype(str).str.zfill(3).str[:2].astype(np.int32)

	frame['heavy_ind'] = frame['ATECO3'].apply(is_heavy_industry).astype(np.int32) * frame['num_companies']
	frame['shop_ind'] = frame['ATECO3'].apply(is_shop_activity).astype(np.int32) * frame['num_companies']
	frame['creative_ind'] = frame['ATECO3'].apply(is_creative_industry).astype(np.int32) * frame['num_companies']

	industry_df = frame.groupby('sez2011').agg(
		{'num_companies': 'sum', 'heavy_ind': 'sum', 'shop_ind': 'sum', 'creative_ind': 'sum', 'employees': 'sum'})
	assert industry_df.index.is_unique

	path = r'data/census'  # use your path
	allFiles = glob.glob(path + "/*.csv")
	list_ = []
	for file_ in allFiles:
		temp_df = pd.read_csv(file_, index_col=None, header=0, encoding="ISO-8859-1", delimiter=';')
		list_.append(temp_df)
	frame = pd.concat(list_)
	frame = frame.rename(columns={'SEZ2011': 'sez2011'})
	frame['sez2011'] = frame.sez2011.astype(str).str.zfill(13)
	frame = frame.set_index('sez2011')

	frame = frame[['P1', 'A3', 'E1', 'E4', 'E3', 'P137', 'P139', 'A44',
				   'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16']]
	frame = frame.rename(columns={
		'P1': 'population',
		'A3': 'vacant_buildings',
		'E1': '#_buildings',
		'E3': '#_res_buildings',
		'E4': '#_comm_buildings',
		'A44': 'm2_residential'
	})
	assert frame.index.is_unique

	land_use_df = pd.read_csv('data/generated_files/urban_atlas_census.csv', dtype={'sez2011': str})
	land_use_df = land_use_df.set_index('sez2011')
	assert land_use_df.index.is_unique

	placepulse_df = pd.read_csv('data/generated_files/placepulse_census.csv', dtype={'sez2011': str})
	placepulse_df = placepulse_df.set_index('sez2011')
	assert placepulse_df.index.is_unique

	walkability_df = pd.read_parquet('data/generated_files/osm_walkability_census.parquet')
	assert walkability_df.index.is_unique

	# MERGE
	df = pd.merge(df, frame, left_index=True, right_index=True, how='left')
	df = pd.merge(df, industry_df, left_index=True, right_index=True, how='left')
	df = pd.merge(df, comp_df, left_index=True, right_index=True, how='left')
	df = pd.merge(df, land_df, left_index=True, right_index=True, how='left')
	df = pd.merge(df, land_use_df, left_index=True, right_index=True, how='left')
	df = pd.merge(df, placepulse_df, left_index=True, right_index=True, how='left')
	df = pd.merge(df, walkability_df, left_index=True, right_index=True, how='left')

	assert df.index.is_unique

	print(df.columns)

	print("FILTERING DATASET")
	df = df.astype(np.float32)
	df = df.sort_index()

	print("SPATIAL MATRIX")
	spatial_matrix = pd.read_sql_query('select geoid1, geoid2 FROM spatial_matrix', con=con, index_col='geoid1')
	spatial_matrix = pd.merge(pd.DataFrame(index=df.index), spatial_matrix, left_index=True, right_on='geoid2')
	spatial_matrix = pd.merge(pd.DataFrame(index=df.index), spatial_matrix, left_index=True, right_index=True)
	spatial_matrix['dist'] = 1
	spatial_matrix['dist'] = spatial_matrix['dist'].astype(np.float32)

	print("WRITING MATRIX")
	spatial_matrix.to_parquet('data/generated_files/spatial_matrix_dsaa.parquet')
	df.to_parquet('data/generated_files/dataset_dsaa.parquet')


if __name__ == '__main__':
	main()
