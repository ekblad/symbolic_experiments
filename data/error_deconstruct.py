
import os,sys,random,pickle,copy,re,operator
import pandas as pd
import numpy as np
import scipy as sp
from scipy.special import softmax
from plot_utils import *

import ast
from sklearn.metrics import confusion_matrix

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import hypervolume
from deap import creator
from deap import tools
from deap import gp

def Toolbox():
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

	input_list,output_list = labels('../../sym_data_4.csv')

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

def main():

	data = pd.read_csv('model_sensitivities.csv',header=0,index_col=0,low_memory=False)
	data = data.dropna()
	data_base = data[data.cluster == 3][data.experiment == 'classification']

	dir_path = os.path.dirname(os.path.realpath(__file__))
	toolbox = Toolbox()
	model_error = {}

	for model in data_base.model_id.unique(): # this opens many files multiple times, messier to avoid

		model_error[model] = {}
		trial_model = re.findall(r'\d+',model)

		local_path = os.path.join(dir_path,'sc/seed_{}_results'.format(trial_model[0]))
		file_path = os.path.join(local_path,"sc_{}_results.pickle".format(trial_model[0]))

		with open(file_path,"rb") as f: 
			data = pickle.load(f)

		hof = data['halloffame']
		model_construct = hof[int(trial_model[1])]
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

		A = confusion_matrix(y_true=values, 
				  y_pred=func2(model_construct,samples), 
				  )

		B = confusion_matrix(y_true=test_values, 
							  y_pred=func2(model_construct,test_samples), 
							  )

		num_right_train = A.trace()
		total_train = np.sum(A)
		percent_train = 100*(num_right_train/total_train)

		num_right_test = B.trace()
		total_test = np.sum(B)
		percent_test = 100*(num_right_test/total_test)

		A,B =  A.astype(float),B.astype(float)
		for i in np.arange(0,A.shape[0]):
			A_sum = np.sum(A[i])
			B_sum = np.sum(B[i])
			A[i] = np.divide(A[i],A_sum)
			B[i] = np.divide(B[i],B_sum)

		np.fill_diagonal(np.fliplr(A), 0)
		A_wrong = np.sum(A,axis=1)
		np.fill_diagonal(np.fliplr(B), 0)
		B_wrong = np.sum(B,axis=1) # percent wrong each class

		model_error[model]['training_error'] = tuple(A_wrong)
		model_error[model]['test_error'] = tuple(B_wrong)

	model_error = pd.concat({k: pd.DataFrame(v).T for k, v in model_error.items()}, axis=0)
	model_error = model_error.unstack(level=-1)
	model_error.columns = model_error.columns.map(lambda x: '|'.join([str(i) for i in x]))
	print(model_error)
	model_error.to_csv('model_class_acc.csv')

if __name__ == '__main__':
	main()