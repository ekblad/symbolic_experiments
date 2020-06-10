# this script combines sensitivity data from both experiments into a long-form data set with the following fields:

# Experiment || Model ID || Complexity || Training Error || Test Error || Input || Category || Neighbor || Lag || Cluster || Sensitivities

# Experiment: Regression or Classification
# Model ID: seed/trial number and model number, fix the convention
# Input: Real fields from before, only recorded when a non-zero sensitivity is recorded
# Category: Economic-Tree/Non, Water-Pump/Deliv, Land-Tree/Non
# Neighbor: Home, Neighbor 1, etc.
# Lag: Lag 1-6, None
# Cluster: Cluster 1-5
# Sensitivity: Last 6 columns - local and total sensitivity indices and confidence intervals

# primarily for plotting/visual analysis

import pickle
import os
import operator
import copy
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans

random_seed = 0
num_clusters = 5 # number of clusters
dir_path = os.path.dirname(os.path.realpath(__file__))
files = ['sr_cluster_data.csv','sc_cluster_data.csv']
pickles = ['sr_sensitivity.pickle','sc_sensitivity.pickle']
experiments = ['regression','classification']
sensitivities = ['S1','S1_conf','ST','ST_conf'] #,'S2','S2_conf'] # second-orders mostly NaN

for o in [0,1]:
	data = pd.read_csv(files[o],header=0,index_col=0).sample(frac=1.0,random_state=random_seed)
	# print(data)
	# exit()
	data_metrics = copy.copy(data[data.columns[-3:]])

	est = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1, random_state=random_seed)
	kmeans = est.fit(data_metrics)
	labels = est.labels_
	print(labels)
	# s = pd.Series(labels)

	# clust = pd.DataFrame(index=data.index.values)
	data_metrics['cluster'] = labels
	data_metrics['nodes'] = data['nodes'].values
	data_metrics['edges'] = data['edges'].values
	data_metrics['labels'] = data['labels'].values
	data_metrics['experiment'] = [experiments[o] for i in labels]


	with open(os.path.join(dir_path,pickles[o]),'rb') as f:
		sens_data = pickle.load(f)

	sens = {}
	input_list = defaultdict(list)

	for i in sens_data:
		for j in sens_data[i]:
			sens[j] = {}
			for ind_,k in enumerate(sens_data[i][j]['inputs']):
				input_list[j].append(k)
				sens[j][k] = [sens_data[i][j]['sensitivity_indices'][l][ind_] for l in sensitivities]#.append(sens_data[i][j]['sensitivity_indices']['ST'][ind_])

	sens = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in sens.items() ])).T.stack()
	# print(sens)
	# exit()

	# create new
	sens_org = pd.DataFrame(sens.values.tolist(), columns=sensitivities[:4])
	sens_org['model_id'] = [i for i in sens.index.get_level_values(0)]
	sens_org['inputs'] = [i for i in sens.index.get_level_values(1)]
	sens_org['experiment'] = [experiments[o] for i in sens.index]
	sens_org = sens_org.set_index(['experiment','model_id','inputs'])
	data_metrics.index.name = 'model_id'
	data_metrics = data_metrics.set_index(['experiment',data_metrics.index])
	sens_org = sens_org.join(data_metrics,how='outer')

	if o == 0:
		stor = sens_org
	else:
		stor = stor.append(sens_org)

stor.reset_index(inplace=True)

# Category: econ-tree/non, water-pump/deliv, land-tree/non

def categorize(term):
	trees = ['ALMOND','ALMONDHULLS','APRICOT','NECTARINES','PISTACHIO','PLUMS','WALNUT']
	if ('ppu' in term) or ('value' in term):
		if any(tree in term for tree in trees):
			category = 'econ_tree'
		else:
			category = 'econ_non'
	elif 'Pump' in term or 'Deliv' in term:
		if 'Pump' in term:
			category = 'water_pump'
		else:
			category = 'water_deliv'
	elif 'tree' in term:
		category = 'land_tree'
	elif 'non_' in term:
		category = 'land_non'
	else:
		category = 'none'
	return category

def lag(term):
	lags = ['lag1','lag2','lag3','lag4','lag5','lag6']
	if any(lag_ in term for lag_ in lags):
		lag = lags[np.argmax([lag_ in term for lag_ in lags])]
	else:
		lag = 'present'
	return lag

def neighbor(term):
	if 'neighbor' in term:
		neighbor = term[:10]
	else:
		neighbor = 'home'
	return neighbor

stor['category'] = [categorize(i) for i in stor.inputs.values]
stor['lag'] = [lag(i) for i in stor.inputs.values]
stor['neighborhood'] = [neighbor(i) for i in stor.inputs.values]

# reorganize columns
stor = stor[['experiment','model_id','nodes','edges','labels','inputs', 
       'training_error', 'complexity', 'test_error', 'cluster', 'category',
       'lag', 'neighborhood',
       'S1', 'S1_conf', 'ST', 'ST_conf'
       ]]

stor.to_csv('model_sensitivities.csv')