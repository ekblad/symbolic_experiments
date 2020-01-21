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

# this script takes data generated across trials during optimization and 
# collects it into one pickle file for post-processing

import os 
import sys
import random
import pickle

import operator

import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
from pygraphviz import AGraph

# deap imports
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import hypervolume
# from deap.tools.indicator import hv
from deap import creator
from deap import tools
from deap import gp

def main():
	def labels(string_):
		from pandas import read_csv
		reader = read_csv(string_,header=0,index_col=None,low_memory=False,chunksize = 10000)
		for i,chunk in enumerate(reader):
			input_list = chunk.columns.tolist()[:-8]
			output_list =  chunk.columns.tolist()[-2:] # for regression (last two cols for classification)
			return input_list,output_list

	# Define new functions - can define your own this way:
	def protectedDiv(left, right):
		with np.errstate(divide='ignore',invalid='ignore'):
			x = np.divide(left, right)
			if isinstance(x, np.ndarray):
				x[np.isinf(x)] = 1
				x[np.isnan(x)] = 1
			elif np.isinf(x) or np.isnan(x):
				x = 1
		return x

	def if_then_else(input, output1, output2):
		return np.where(input,output1,output2)

	input_list,output_list = labels('../sym_data_4.csv')

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

		tree_vals[np.isnan(tree_vals)] = 0
		soft = softmax(tree_vals,axis=1)
		bools = np.argmax(soft,axis=1)-1

		# Calculate accuracy
		num_right = np.sum([bools == values])
		acc_train = 1-np.sum(num_right)/len(values) # minimizing

		if np.isnan(acc_train):
			acc_train = 999
			avg_len = 999
		else:
			avg_len = np.mean(tree_lens)		

		return acc_train, avg_len

	# setting up optimization here...

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
	trial = np.arange(0,21)

	results = {}

	for trial in trial:
		print(trial)
		seed_results = {}
		local_path = os.path.join(dir_path,'seed_{}_results'.format(trial))

		file_path = os.path.join(local_path,"sc_{}_results.pickle".format(trial))
		with open(file_path,"rb") as f:
			data = pickle.load(f)

		hof = data['halloffame']
		pop = data['population']
		stats = data['logbook']
		input_list = data['inputs']
		output_list = data['outputs']
		samples = data['samples']
		values = data['values']
		test_samples = data['test_samples']
		test_values = data['test_values']

		def func(item,samples=samples,values=values):
			return toolbox.evalInd(item,samples,values)

		def func2(individual,samples=samples):
			tree_vals = np.zeros((len(samples),len(individual)))
			tree_lens = np.zeros((len(individual),))
			for i,ind in enumerate(individual):
				tree_vals[:,i],tree_lens[i] = toolbox.boolTree(ind,samples=samples)
			tree_vals[np.isnan(tree_vals)] = 0
			soft = softmax(tree_vals,axis=1)
			bools = np.argmax(soft,axis=1)-1
			return bools

		convergence = []
		for line in stats:
			convergence.append(line['min'][0])

		gen, smin = stats.select("gen", "min")

		fits = np.zeros(len(hof))
		lens = np.zeros(len(hof))
		tests = np.zeros(len(hof))

		hof_store = []
		nodes = []
		edges = []
		labels = []

		for i,item in enumerate(hof):
			node = []
			edge = []
			label = []
			for item2 in item:
				node2, edge2, label2 = gp.graph(item2)
				node.append(node2)
				edge.append(edge2)
				label.append(label2)
			nodes.append(node)
			edges.append(edge)
			labels.append(label)

			fits[i] = item.fitness.values[0]
			tests[i] = func(item,samples=test_samples,values=test_values)[0]
			lens[i] = item.fitness.values[1]

		seed_results['train_mse'] = fits
		seed_results['complexity'] = lens
		seed_results['test_mse'] = tests
		# seed_results['halloffame'] = hof_store # can't pickle
		seed_results['nodes'] = nodes
		seed_results['edges'] = edges
		seed_results['labels'] = labels
		seed_results['inputs'] = input_list

		results['seed_{0}_results'.format(trial)] = seed_results

	with open('sc_all_results.pickle', 'wb') as f:
		# Pickle the 'data' dictionary using the highest protocol available.
		pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

