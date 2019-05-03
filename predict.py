import pandas as pd
import numpy as np
import argparse

import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error
from scipy import stats
import pickle
import sys


def compute_land_use_mix(entropy_stats):
	entropy_zero = lambda x: stats.entropy(x) if np.sum(x) > 0 else np.nan

	land_use_mixes = np.apply_along_axis(entropy_zero, 1, entropy_stats)
	return land_use_mixes / np.log(entropy_stats.shape[1])


def weighted_avg_and_std(values, weights):
	"""
	Return the weighted average and standard deviation.

	values, weights -- Numpy ndarrays with the same shape.
	"""
	average = np.average(values, weights=weights)
	variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
	return average.astype(np.float32), np.sqrt(variance).astype(np.float32)


def compute_age_mean_std(vec):
	age_intervals = [
		[1919, 1945],
		[1946, 1960],
		[1961, 1970],
		[1971, 1980],
		[1981, 1990],
		[1991, 2000],
		[2001, 2010],
	]
	age_mean_intervals = []
	for low, high in age_intervals:
		now = age_intervals[-1][1]
		avg = float((now - low) + (now - high)) / 2.
		age_mean_intervals.append(avg)
	age_mean_intervals = np.array(age_mean_intervals)

	age_means = np.zeros(len(vec))
	age_stds = np.zeros(len(vec))
	for i, f in enumerate(vec):
		f[-2] += f[-1]
		age_mean = 0
		age_std_w = 0
		if np.sum(f[1:-1]) > 0:
			age_mean, age_std_w = weighted_avg_and_std(age_mean_intervals, f[1:-1])
		age_means[i] = age_mean
		age_stds[i] = age_std_w

	np.place(age_stds, age_means == 0, np.nan) # only where means = 0
	np.place(age_means, age_means == 0, np.nan)
	return age_means, age_stds


def median_absolute_p_error(y_true, y_pred):
	return np.median(np.abs(100 * (y_pred - y_true) / y_true))


def create_spatial_matrix(X, spatial_matrix):

	mapping_sez_index = {k: i for i, k in enumerate(X.index.tolist())}
	nrows = X.values.shape[0]
	# transform spatial matrix in sez1, sez2, dist
	sp = spatial_matrix.reset_index().values

	rows = [mapping_sez_index[s[0]] for s in sp]
	cols = [mapping_sez_index[s[1]] for s in sp]
	dists = [d for _, _, d in sp]

	S = csr_matrix((dists+dists, (rows+cols, cols+rows)), shape=(nrows, nrows), dtype=np.float32)
	# Row normalization
	S = normalize(S, norm='l1', axis=1)

	return S


def train_validate_test_split_with_care(df, care_df, validate_percent=.15, test_percent=.15, seed=None, return_indexes=False):
	# care is the dataframe with houses
	np.random.seed(seed)
	perm = np.random.permutation(care_df.index)
	m = len(care_df.index)
	validate_end = int(validate_percent * m)
	test_end = int(test_percent * m) + validate_end

	train_idxs = perm[test_end:]
	val_idxs = perm[:validate_end]
	test_idxs = perm[validate_end:test_end]

	if return_indexes:
		return train_idxs, val_idxs, test_idxs

	validate = df.loc[val_idxs].copy()
	test = df.loc[test_idxs].copy()
	train = df.loc[train_idxs].copy()
	return train, validate, test


def kfold(care_df, n_folds=5, shuffle=True, seed=None):
	np.random.seed(seed)
	perm = care_df.index.tolist()
	if shuffle:
		perm = np.random.permutation(perm)

	num_samples = len(care_df.index)
	fold_size = num_samples // n_folds

	for i in range(1, n_folds):
		current = (i-1)*fold_size
		end = fold_size*i

		if i == n_folds -1:
			yield perm[:current], perm[current:]
		else:
			yield np.concatenate((perm[:current], perm[end:]), axis=0), perm[current:end]


def make_argument_parser():
	"""
	Creates an ArgumentParser to read the options for this script from
	sys.argv
	:return:
	"""
	parser = argparse.ArgumentParser(
		description="Launch MCMC computation for crime"
	)
	parser.add_argument('--target', '-T',
						help='Target variable', default='price', choices=['price'])
	parser.add_argument('--njobs', '-J',
						default=8, type=int)
	parser.add_argument('--testcity', '-C',
						default=None, type=str)
	parser.add_argument('--gpu', default=None, type=int)
	parser.add_argument('--gridsearch', dest='gridsearch', action='store_true')
	parser.add_argument('--no-gridsearch', dest='gridsearch', action='store_false')
	parser.add_argument('--cv', dest='cv', action='store_true')
	parser.add_argument('--no-cv', dest='cv', action='store_false')
	parser.add_argument('--save', dest='save', action='store_true')
	parser.add_argument('--no-save', dest='save', action='store_false')
	parser.add_argument('--neigh', dest='neigh', action='store_true')
	parser.add_argument('--no-neigh', dest='neigh', action='store_false')
	parser.add_argument('--open', dest='open', action='store_true', help="Include all indexes (non open ones)")
	parser.add_argument('--no-open', dest='open', action='store_false', help="Use only open indexes")

	parser.set_defaults(save=True, neigh=True, open=False, cv=False, gridsearch=False)
	return parser


def main():
	SEED = 42 # Just to replicate our final experiment
	parser = make_argument_parser()
	args = parser.parse_args()
	print("PARAMETERS", args)

	df_case = pd.read_parquet('data/generated_files/selling_houses.parquet')

	df = pd.read_parquet('data/generated_files/dataset_dsaa_release.parquet')

	if args.neigh:
		print("COMPUTING NEIGHBORHOOD FEATURES")
		# SPATIAL MATRIX
		spatial_matrix = pd.read_parquet('data/generated_files/spatial_matrix_dsaa.parquet')
		spatial_matrix = pd.merge(pd.DataFrame(index=df.index), spatial_matrix, left_index=True, right_on='sez1')
		spatial_matrix = pd.merge(pd.DataFrame(index=df.index), spatial_matrix, left_index=True, right_index=True)

	place_features = []
	neighborhood_columns = []
	additional_features = []

	if args.neigh:
		place_features = [
			'employees',
			'population',
			'vacant_buildings',
			'm2_residential',

			'prox_metro',
			'prox_railway',
			'dist_airport',
			'n_bus_stops',
			'prox_parks',
			'dist_industrial_area',
			'dist_sea',
			'prox_coffee', 'prox_enter', 'prox_shops', 'prox_restaurants', 'prox_school', 'prox_grocery',
			'prox_library',
		]

		neighborhood_columns = [
			'#_buildings',
			'#_res_buildings',
			'#_comm_buildings',
			'urban_sum', 'comm_sum', 'other_sum', 'green_sum',

			'census_area',

			'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16',
			'num_companies',

			'employees',
			'population',
			'vacant_buildings',
			'm2_residential',

			'heavy_ind',
			'shop_ind',
			'creative_ind',
		]
		additional_features = [
			'neigh_land_use_mix3',

			'neigh_buildings_age_mean', 'neigh_buildings_age_std',
			'neigh_n_blocks'
		]

		if not args.open:
			neighborhood_columns.extend(['score', 'fatturato', 'assessed_land_value'])

	house_features = ['sqmt',
					  'constructionYear', 'energyClass', 'expensesCondominium',

					  'newFloorNumber', 'newHeating', 'portiere', 'infissi', 'giardino', 'arredato', 'esposizione',
					  'kitchen',
					  'condition', 'land_value',

					  # 'Piscina',
					  'Idromassaggio',
					  'Mansarda',
					  'Cantina',
					  'Camino',
					  'has_terrace', 'garage',

					  'placeType_Appartamento', 'placeType_Attico / Mansarda',
					  'placeType_Casa indipendente', 'placeType_Loft / Open Space',
					  'placeType_Villetta a schiera',
					  'property_class', 'property_type',
					  'locali', 'camere', 'altro', 'bagni',
					  ]
	new_neighborhood_columns = ['neigh_{}'.format(c) for c in neighborhood_columns]

	all_features = house_features + place_features + new_neighborhood_columns + additional_features
	print("FEATURES")
	print(all_features)

	for x in new_neighborhood_columns:
		df[x] = 0.
	for x in additional_features:
		df[x] = 0.
	df = df.astype(np.float32)

	best_gridsearch_params = None
	best_gridsearch_MAE = sys.maxsize
	default_params = {'subsample': 0.9, 'reg_lambda': 5, 'reg_alpha': 1, 'n_estimators': 4000, 'min_child_weight': 3,
					  'max_depth': 20, 'learning_rate': 0.001, 'silent': 1, 'random_state': SEED, }
	additional_model_params = {'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': 1}
	search_params = {'learning_rate': [0.001], 'subsample': [0.9],
					 'max_depth': np.arange(15, 25, 5), 'min_child_weight': np.arange(3, 15, 3),
					 'n_estimators': np.arange(4000, 5000, 1000),
					 'reg_lambda': [1, 3, 5], 'reg_alpha': [0, 1, 2],
					 'random_state': [SEED]}
	if args.gpu:
		additional_model_params['gpu_id'] = args.gpu
		additional_model_params['tree_method'] = 'gpu_exact'
		additional_model_params['objective'] = 'gpu:reg:linear'

	# Dataframe with one house per census block
	care_df = pd.merge(df, pd.DataFrame(index=df_case.groupby(df_case.index).first().index), right_index=True,
					   left_index=True)

	n_splits = 5
	if args.cv:
		param_list = [default_params for _ in range(n_splits)]
		fold_iterator = kfold(care_df)
	elif args.gridsearch:
		n_iterations = 15
		train_idxs, _, test_idxs = train_validate_test_split_with_care(df, care_df, seed=SEED, return_indexes=True,
																	   validate_percent=0)
		param_list = list(ParameterSampler(search_params, n_iter=n_iterations))

		folds = list(kfold(care_df))
		# Cross-validation for each param-group
		fold_iterator = [x for _ in range(len(param_list)) for x in folds]
		param_list = [p for p in param_list for _ in folds]
	elif args.testcity:
		df_train = df[~(df.index.str.startswith(args.testcity))]
		df_test = df[(df.index.str.startswith(args.testcity))]

		train_idxs = df_train.groupby(df_train.index).first().index
		test_idxs = df_test.groupby(df_test.index).first().index

		param_list = [default_params]
		fold_iterator = [[train_idxs, test_idxs]]
	else:
		care_df = pd.merge(df, pd.DataFrame(index=df_case.groupby(df_case.index).first().index), right_index=True,
						   left_index=True)
		train_idxs, _, test_idxs = train_validate_test_split_with_care(df, care_df, seed=SEED, return_indexes=True,
																	   validate_percent=0)

		param_list = [default_params]
		fold_iterator = [[train_idxs, test_idxs]]

	# Cross-validation - or single city - loop
	cv_iter = 0
	ys_true = []
	ys_pred = []
	for (train_index, test_index), params in zip(fold_iterator, param_list):
		care_df2 = pd.merge(df.loc[train_index, :], pd.DataFrame(index=df_case.groupby(df_case.index).first().index), right_index=True,
						   left_index=True)
		df_train, df_val, _ = train_validate_test_split_with_care(df, care_df2, validate_percent=0.3, seed=SEED)
		df_test = df.loc[test_index, :].copy()
		test_indexes = [df.index.get_loc(x) for x in df_test.index.tolist()]
		params = {**params, **additional_model_params}

		df_cv = pd.concat((df_train, df_val))
		if args.neigh:
			spatial_matrix_temp = pd.merge(pd.DataFrame(index=df_cv.index), spatial_matrix, left_index=True,
										   right_on='sez1')
			spatial_matrix_temp = pd.merge(pd.DataFrame(index=df_cv.index), spatial_matrix_temp, left_index=True,
										   right_index=True)
			S = create_spatial_matrix(df_cv, spatial_matrix_temp)
			n_blocks = np.array(S.sum(axis=1)).flatten()

			# neighborhood features training
			df_cv.loc[:, new_neighborhood_columns] = S.dot(df_cv.loc[:, neighborhood_columns].copy().fillna(0).values)
			n_non_nan_scores = S.dot((~np.isnan(df_cv['score'].values)).astype(int))

			# additional features
			df_cv.loc[:, 'neigh_census_area'] = df_cv['neigh_census_area'] / n_blocks
			df_cv.loc[:, 'neigh_n_blocks'] = n_blocks
			df_cv.loc[:, 'neigh_m2_residential'] = df_cv['neigh_m2_residential'] / df_cv['census_area']
			if not args.open:
				df_cv.loc[:, 'neigh_score'] = df_cv['neigh_score'] / n_non_nan_scores
			df_cv.loc[:, 'neigh_buildings_age_mean'], df_cv.loc[:, 'neigh_buildings_age_std'] = compute_age_mean_std(
				df_cv.loc[:, ['E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16']].copy().values)
			df_cv.loc[:, 'neigh_land_use_mix3'] = compute_land_use_mix(
				df_cv.loc[:, ['neigh_urban_sum', 'neigh_comm_sum', 'neigh_green_sum']].copy().values)

		# Merge with housing listings
		df_cv_m = pd.merge(df_cv, df_case, right_index=True, left_index=True)

		print("USING PARAMS")
		print(params)
		print("NUM ROWS (CV):", len(df_cv_m))

		if args.neigh:
			spatial_matrix_temp = pd.merge(pd.DataFrame(index=df_train.index), spatial_matrix, left_index=True, right_on='sez1')
			spatial_matrix_temp = pd.merge(pd.DataFrame(index=df_train.index), spatial_matrix_temp, left_index=True, right_index=True)
			S_train = create_spatial_matrix(df_train, spatial_matrix_temp)
			n_blocks = np.array(S_train.sum(axis=1)).flatten()

			# neighborhood features training
			df_train.loc[:, new_neighborhood_columns] = S_train.dot(df_train.loc[:, neighborhood_columns].copy().fillna(0).values)
			n_non_nan_scores = S_train.dot((~np.isnan(df_train['score'].values)).astype(int))

			# additional features
			df_train.loc[:, 'neigh_census_area'] = df_train['neigh_census_area'] / n_blocks
			df_train.loc[:, 'neigh_n_blocks'] = n_blocks
			df_train.loc[:, 'neigh_m2_residential'] = df_train['neigh_m2_residential'] / df_train['census_area']
			if not args.open:
				df_train.loc[:, 'neigh_score'] = df_train['neigh_score'] / n_non_nan_scores
			df_train['neigh_buildings_age_mean'], df_train['neigh_buildings_age_std'] = compute_age_mean_std(
					df_train.loc[:, ['E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16']].copy().values)
			df_train.loc[:, 'neigh_land_use_mix3'] = compute_land_use_mix(df_train.loc[:, ['neigh_urban_sum', 'neigh_comm_sum', 'neigh_green_sum']].copy().values)

		# neighborhood features training
		val_indexes = [df_cv.index.get_loc(x) for x in df_val.index.tolist()]
		if args.neigh:
			df_val.loc[:, new_neighborhood_columns] = df_cv.loc[:, new_neighborhood_columns].values[val_indexes]
			df_val.loc[:, additional_features] = df_cv.loc[:, additional_features].values[val_indexes]

		# Merge with housing listings
		df_train_m = pd.merge(df_train, df_case, right_index=True, left_index=True)
		df_val_m = pd.merge(df_val, df_case, right_index=True, left_index=True)

		X_train = df_train_m[all_features].reset_index(drop=True)
		y_train = df_train_m[args.target].reset_index(drop=True)
		X_val = df_val_m[all_features].reset_index(drop=True)
		y_val = df_val_m[args.target].reset_index(drop=True)
		xgdmat = xgb.DMatrix(X_train, y_train, feature_names=all_features)
		xgdmat_val = xgb.DMatrix(X_val, y_val, feature_names=all_features)

		eval_set = [(xgdmat, 'train'), (xgdmat_val, 'validation')]
		final_gb = xgb.train(params, xgdmat, num_boost_round=params['n_estimators'], evals=eval_set, early_stopping_rounds=50, verbose_eval=150)
		if args.testcity is None and args.save:
			print("SAVING MODEL")
			pickle.dump(final_gb, open('data/generated_files/trained_{target}.model'.format(target=args.target), "wb"))

		# TEST DATA
		if args.neigh:
			S = create_spatial_matrix(df, spatial_matrix)
			n_blocks = np.array(S.sum(axis=1)).flatten()

			# neighborhood features training
			df.loc[:, new_neighborhood_columns] = S.dot(df.loc[:, neighborhood_columns].copy().fillna(0).values)
			n_non_nan_scores = S.dot((~np.isnan(df['score'].values)).astype(int))

			# additional features
			df.loc[:, 'neigh_census_area'] = df['neigh_census_area'] / n_blocks
			df.loc[:, 'neigh_n_blocks'] = n_blocks
			df.loc[:, 'neigh_m2_residential'] = df['neigh_m2_residential'] / df['census_area']
			if not args.open:
				df.loc[:, 'neigh_score'] = df['neigh_score'] / n_non_nan_scores
			df['neigh_buildings_age_mean'], df['neigh_buildings_age_std'] = compute_age_mean_std(
					df.loc[:, ['E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16']].copy().values)
			df.loc[:, 'neigh_land_use_mix3'] = compute_land_use_mix(df.loc[:, ['neigh_urban_sum', 'neigh_comm_sum', 'neigh_green_sum']].copy().values)

			# neighborhood features training
			df_test.loc[:, new_neighborhood_columns] = df.iloc[test_indexes][new_neighborhood_columns]
			df_test.loc[:, additional_features] = df.iloc[test_indexes][additional_features]

		# Merge with housing listings
		df_test_m = pd.merge(df_test, df_case, right_index=True, left_index=True)

		X_test = df_test_m[all_features].reset_index(drop=True)
		y_test = df_test_m[args.target].reset_index(drop=True)
		testdmat = xgb.DMatrix(X_test, feature_names=all_features)
		y_pred = final_gb.predict(testdmat, ntree_limit=final_gb.best_ntree_limit)

		# Save results
		if args.testcity is None and args.save:
			tosave_df = pd.DataFrame(list(zip(df_test_m.index.tolist(), y_pred, y_test-y_pred, df_test_m['price'].values)))
			tosave_df.columns = ['geoid', 'prediction', 'difference', 'price']
			tosave_df.to_csv('data/generated_files/raw_predictions_ubicomp_{target}.csv'.format(target=args.target), index=False)

			#save_test_df = pd.concat((df_test_m[all_features].reset_index(drop=True), tosave_df.reset_index(drop=True)), axis=1)
			#save_test_df.to_csv('data/generated_files/test_df_ubicomp_{target}.csv'.format(target=args.target))

		ys_true.extend(y_test)
		ys_pred.extend(y_pred)
		print("")
		print("FOLD RESULTS DATA")
		print("MAE", mean_absolute_error(y_test, y_pred))
		print("MdAPE", median_absolute_p_error(y_test, y_pred))

		if cv_iter % n_splits == n_splits-1:
			avg_mae = mean_absolute_error(ys_true, ys_pred)
			if avg_mae < best_gridsearch_MAE:
				best_gridsearch_params = params
				best_gridsearch_MAE = avg_mae
			ys_true = []
			ys_pred = []
		cv_iter += 1

	if not args.gridsearch:
		ys_true = np.array(ys_true)
		ys_pred = np.array(ys_pred)
		print("")
		print("CROSS-VALIDATION RESULTS")
		print("MAE", mean_absolute_error(ys_true, ys_pred))
		print("MdAPE", median_absolute_p_error(ys_true, ys_pred))

	if args.gridsearch:
		print("BEST PARAMS:")
		print(best_gridsearch_params)


if __name__ == '__main__':
	main()
