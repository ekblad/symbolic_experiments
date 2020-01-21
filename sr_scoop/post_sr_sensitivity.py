#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import os 
import sys
import random
import pickle

import operator

import numpy as np
import pandas as pd
from scipy.special import softmax

# deap imports
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import hypervolume
# from deap.tools.indicator import hv
from deap import creator
from deap import tools
from deap import gp

from SALib.sample import saltelli
from SALib.analyze import sobol

# Set random seed
rand_seed = 0
random.seed(rand_seed)
np.random.seed(rand_seed)

def main():
	def labels(string_):
		from pandas import read_csv
		reader = read_csv(string_,header=0,index_col=None,low_memory=False,chunksize = 10000)
		for i,chunk in enumerate(reader):
			input_list = chunk.columns.tolist()[:-8]
			return input_list

	input_list = labels('../sym_data_4.csv')

	def protectedDiv(left, right):
		with np.errstate(divide='ignore',invalid='ignore'):
			x = np.divide(left, right)
			if isinstance(x, np.ndarray):
				x[np.isinf(x)] = 1
				x[np.isnan(x)] = 1
			elif np.isinf(x) or np.isnan(x):
				x = 1
		return x

	def if_then_else(input_, output1, output2):
		return np.where(input_,output1,output2)

	pset = gp.PrimitiveSetTyped("MAIN", [float for i in np.arange(0,len(input_list))], float,"IN")

	pset.addTerminal(1, bool)
	pset.addTerminal(0, bool)
	pset.addPrimitive(operator.lt, [float, float], bool)
	pset.addPrimitive(if_then_else, [bool, float, float], float, name="ite")
	pset.addPrimitive(np.add, [float, float], float, name="vadd")
	pset.addPrimitive(np.subtract, [float, float], float, name="vsub")
	pset.addPrimitive(np.multiply, [float, float], float, name="vmul")
	pset.addPrimitive(protectedDiv, [float, float], float, name="vdiv")
	pset.addPrimitive(np.negative, [float], float, name="vneg")
	pset.addPrimitive(np.sin, [float], float, name="vsin")
	pset.addPrimitive(np.cos, [float], float, name="vcos")

	pset.addEphemeralConstant("rand10", lambda: np.random.randint(0,100)/10.,float)
	pset.addEphemeralConstant("rand100", lambda: np.random.randint(0,100)/1.,float)

	def boolTree(tree,samples):
		func = toolbox.compile(expr=tree)
		vals = func(*samples.T)
		lens = len(tree)/90.
		return vals,lens

	def evalInd(individual,samples,values):
		max_len = np.max([len(ind) for ind in individual])
		if max_len > 90: # can't compile trees greater in length/depth than this
			return 999, 999

		tree_vals = np.zeros((len(samples),len(individual)))
		tree_lens = np.zeros((len(individual),))
		for i,ind in enumerate(individual):
			tree_vals[:,i],tree_lens[i] = boolTree(ind,samples=samples)
		avg_len = np.mean(tree_lens)
		tree_vals[np.isnan(tree_vals)] = 0

		predictions = np.mean(tree_vals,axis=1)
		diff_train = np.mean(np.subtract(predictions,values)**2)

		if np.isnan(diff_train):
			diff_train = 999
			avg_len = 999

		return diff_train, avg_len

	# # build keyword arguments - inputs to functions 
	kwargs = {}
	for num,label in enumerate(input_list):
		arg = 'IN' + str(num)
		kwargs[arg] = str(label)

	pset.renameArguments(**kwargs)

	# these are specifying min problem in weights
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
	creator.create("FitnessTree", base.Fitness, weights=(-1.0,))
	creator.create("Tree", gp.PrimitiveTree, fitness=creator.FitnessTree)
	creator.create("Individual", list, fitness=creator.FitnessMin)

	toolbox = base.Toolbox()
	toolbox.register("blank_class", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) # all trees equivalent at generation
	toolbox.register("ind", tools.initIterate, creator.Tree, toolbox.blank_class)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.ind, n=3)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("compile", gp.compile, pset=pset)
	toolbox.register("evalInd", evalInd)
	toolbox.register("boolTree", boolTree)
	toolbox.register("select", tools.selNSGA2)
	toolbox.register("sort", tools.sortNondominated)
	toolbox.register("mate", gp.cxOnePoint)
	toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=2)
	toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

	return toolbox
		
if __name__ == "__main__":

	dir_path = os.path.dirname(os.path.realpath(__file__))
	toolbox = main()
	funcs = ['lt','ite','vadd','vsub','vmul','vdiv','vneg','vsin','vcos']
	trial = np.arange(0,21)
	sym_data = pd.read_csv('../sym_data_4.csv',header=0,index_col=None,low_memory=False)
	results = {}

	for trial in trial:
		print(trial)
		seed_results = {}
		local_path = os.path.join(dir_path,'seed_{}_results'.format(trial))

		file_path = os.path.join(local_path,"sr_{}_results.pickle".format(trial))
		
		with open(file_path,"rb") as f:
			data = pickle.load(f)
			hof = data['halloffame']
			input_list = data['inputs']
			output_list = data['outputs']

		samples = sym_data[input_list].values
		values = sym_data[output_list].values

		def func(item,samples=samples,values=values):
			return toolbox.evalInd(item,samples,values)

		def func2(individual,samples=samples):
			tree_vals = np.zeros((len(samples),len(individual)))
			tree_lens = np.zeros((len(individual),))
			for i,ind in enumerate(individual):
				tree_vals[:,i],tree_lens[i] = toolbox.boolTree(ind,samples=samples)
			tree_vals[np.isnan(tree_vals)] = 0
			predictions = np.mean(tree_vals,axis=1)
			return predictions

		labels = []
		for i,item in enumerate(hof):
			model_id = 'seed_{0}_model_{1}'.format(trial,i)
			label = []
			for item2 in item:
				label2 = gp.graph(item2)[2]
				label.append(label2)
			labels.append(label)
			model_funcs = []
			model_inputs = []
			model_terms = []
			for k in range(0,3):
				for l in labels[i][k]:
					if labels[i][k][l] in funcs:
						model_funcs.append(labels[i][k][l])
					elif labels[i][k][l] in input_list:
						if labels[i][k][l] not in model_inputs:
							model_inputs.append(labels[i][k][l])
					else:
						model_terms.append(float(labels[i][k][l]))

			# Define SALib problem for this model:
			problem = {
						'num_vars': len(model_inputs),
						'names': model_inputs,
						'bounds': [[np.min(sym_data[col].values),np.max(sym_data[col].values)] for col in model_inputs]
						}

			param_values = saltelli.sample(problem,1000*len(model_inputs))
			dat_local = sym_data[input_list].iloc[0:param_values.shape[0]]

			# augment samples for this test
			dat_dict = {}
			for input_ in input_list:
				if input_ in model_inputs:
					dat_dict[input_] = param_values[:,model_inputs.index(input_)]
				else:
					dat_dict[input_] = np.zeros((param_values.shape[0],))
			dat_local = pd.DataFrame.from_dict(dat_dict)
			
			# evaluate on augmented samples
			samples_aug = dat_local.values
			evaluations = func2(item,samples=samples_aug)

			# perform analysis
			Si = sobol.analyze(problem,evaluations)
		
			seed_results[model_id] = {}
			seed_results[model_id]['inputs'] = model_inputs
			# Si is a Python dict with the keys "S1", "S2", "ST", "S1_conf", "S2_conf", and "ST_conf"
			seed_results[model_id]['sensitivity_indices'] = Si

		results['seed_{0}_results'.format(trial)] = seed_results

	with open('sr_sensitivity.pickle', 'wb') as f:
	    # Pickle the 'data' dictionary using the highest protocol available.
	    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


