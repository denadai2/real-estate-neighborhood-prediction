import pandas as pd
import numpy as np

import psycopg2
import requests
import sys
from joblib import Parallel, delayed


def make_trip(lon1, lat1, dest):
	# PERSONALIZE HERE
	r = requests.get(
		'http://localhost:5000/route/v1/foot/{lon1},{lat1};{dest}'.format(lon1=lon1, lat1=lat1, dest=dest))
	routes = r.json()['routes']
	distance = sys.maxsize
	if routes:
		distance = routes[0]['distance']
	return distance


def walkscore_list(clon, clat, list_meters, i=0):
	dists = [make_trip(clon, clat, x) for x in list_meters]
	return i, np.average(walkscore(np.array(dists)))


def walkscore(meters, max_walk=1500):
	max_walk = max_walk
	score = np.exp(-5 * (meters / max_walk) ** 5)
	score = np.clip(score, 0, 1)
	return score


def main():
	# PERSONALIZE HERE
	con = psycopg2.connect(database="dsaa", user="nadai", host="localhost")

	census_df = pd.read_sql_query('select geoid, clon, clat FROM census_areas_onfocus', con=con, index_col='geoid')

	osm_indexes = [
		'prox_grocery',
		'prox_restaurants',
		'prox_shops',
		'prox_school',
		'prox_enter',
		'prox_coffee',
		'prox_metro',
		'prox_parks',
		'prox_railway',
		'prox_library',
	]

	df_transports = pd.read_sql_query('select geoid, dist as dist_airport FROM osm_walkability_prox_airport', con=con,
									  index_col='geoid')
	df_transports2 = pd.read_sql_query(
		'select geoid, dist as dist_industrial_area FROM osm_walkability_prox_industrial_area', con=con,
		index_col='geoid')
	df_transports3 = pd.read_sql_query('select geoid, c as n_bus_stops FROM osm_walkability_nbus_stops', con=con,
									   index_col='geoid')
	df_transports4 = pd.read_sql_query('select geoid, dist as dist_sea FROM osm_walkability_prox_sea', con=con, index_col='geoid')
	df_transports4.loc[:, 'dist_sea'] = df_transports4['dist_sea'].apply(lambda x: walkscore(x, 5000))

	df_transports = pd.merge(df_transports, df_transports2, right_index=True, left_index=True)
	df_transports = pd.merge(df_transports, df_transports3, right_index=True, left_index=True)
	df_transports = pd.merge(df_transports, df_transports4, right_index=True, left_index=True)

	for idx in osm_indexes:
		df = pd.read_sql_query('select * FROM osm_walkability_{}'.format(idx), con=con, index_col='geoid')
		df = pd.merge(df, census_df, right_index=True, left_index=True)
		fetched_array = df[['clon', 'clat', 'dests']].values

		print("PROCESSING", idx)
		results = Parallel(n_jobs=15)(delayed(walkscore_list)(x[0], x[1], x[2], i) for i, x in enumerate(fetched_array))
		results = sorted(results, key=lambda x: x[0])
		df[idx] = np.array(results)[:, 1]

		df_transports = pd.merge(df[[idx]], df_transports, right_index=True, left_index=True)

	df_transports = df_transports.astype(np.float32)
	df_transports.to_parquet('data/generated_files/osm_walkability_census.parquet')


if __name__ == '__main__':
	main()