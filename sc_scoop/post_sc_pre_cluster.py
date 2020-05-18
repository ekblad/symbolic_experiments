# this script collects all models from all trials and prepares for 
# clustering/plotting by organizing into:
# model ID || inputs ... || functions ... || metrics ...
# where we record input and function values as the number of 
# times they occur in a given model

import pickle
import os
import operator
import pandas as pd
import numpy as np
from collections import Counter

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path,'sc_all_results.pickle')
with open(file_path,"rb") as f:
	data = pickle.load(f)

funcs = ['lt','ite','vadd','vsub','vmul','vdiv','vneg','vsin','vcos']
inputs = data['seed_0_results']['inputs']

func_list = []
input_list = []
term_list = []

func_dicts = {}
input_dicts = {}
training_error = []
test_error = []
complexity = []

for i in data:
	for j,tree in enumerate(data[i]['labels']):
		training_error.append(data[i]['train_mse'][j])
		test_error.append(data[i]['test_mse'][j])
		complexity.append(data[i]['complexity'][j])

		model_id = 'seed_{0}_model_{1}'.format(i[5:7],j)
		print(model_id)
		model_funcs = []
		model_inputs = []

		for k in tree:
			for l in k:
				if l in funcs:
					func_list.append(l)
					model_funcs.append(l)
				elif l in inputs:
					input_list.append(l)
					model_inputs.append(l)
				else:
					term_list.append(float(l))

		func_counts = []
		input_counts = []

		for l in funcs:
			if l in model_funcs:
				func_counts.append(model_funcs.count(l))
			else:
				func_counts.append(0)
		for l in inputs:
			if l in model_inputs:
				input_counts.append(model_inputs.count(l))
			else:
				input_counts.append(0)

		if not ((input_counts == 0) or (np.sum(input_counts) == 0)):
			input_dicts[model_id] = np.zeros((len(inputs),))
		else:
			# input_dicts[model_id] = np.divide(input_counts,np.sum(input_counts))
			input_dicts[model_id] = input_counts

		if not ((func_counts == 0) or (np.sum(func_counts) == 0)):
			func_dicts[model_id] = np.zeros((len(funcs),))
		else:
			# func_dicts[model_id] = np.divide(func_counts,np.sum(func_counts))
			func_dicts[model_id] = func_counts

input_df = pd.DataFrame(input_dicts,index=inputs)
func_df = pd.DataFrame(func_dicts,index=funcs)
frames = [input_df,func_df]
infn_df = pd.concat(frames)

infn_df = infn_df.T
infn_df['training_error'] = training_error
infn_df['complexity'] = complexity
infn_df['test_error'] = test_error
infn_df.to_csv('sc_cluster_data.csv')
