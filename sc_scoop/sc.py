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
from shutil import rmtree
import pickle

import operator

import numpy as np
from scipy.special import softmax
from sklearn import metrics
from plot_utils import *
import matplotlib.pyplot as plt
# from pygraphviz import AGraph

# deap imports
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import hypervolume
from deap import creator
from deap import tools
from deap import gp

import scoop

trial = int(sys.argv[1])
rand_seed = trial
rank = trial
print('this is trial #{}'.format(rank))
random.seed(rand_seed)
np.random.seed(rand_seed)
complexity_bound = 90.

def stand(dat):
	for k in np.arange(0,np.shape(dat)[1]):
		dat[:,k] = np.around((dat[:,k]-np.mean(dat[:,k]))/np.std(dat[:,k]),decimals=4)
	dat[np.isnan(dat)] = 0
	return dat

def data_(string_,rand_seed):
	random.seed(rand_seed)
	np.random.seed(rand_seed)
	from pandas import read_csv
	reader = read_csv(string_,header=0,index_col=None,low_memory=False,chunksize = 10000)
	for i,chunk in enumerate(reader):
		# if i > 0:
		# 	break
		# print('~~ loading mini-batch #{} ~~'.format(i))
		if i == 0:
			input_list = chunk.columns.tolist()
			output_list =  chunk.columns.tolist()[-2:] # for regression (last two cols for classification)

		
		cant_use = ['home_tree_state_lag0',
					'home_non_state_lag0',
					'home_total_lag0',
					'home_tree_percent_lag0',
					'home_tree_change_lag0',
					'home_non_change_lag0',
					# 'home_tree_change_sign',
					# 'home_non_change_sign'
					]

		if i == 0:
			for element in cant_use:
				input_list.remove(element)
			for element in output_list:
				input_list.remove(element)

		# shuffle rows (there might be a vague order to the data)
		in_df_ = chunk.sample(frac=1,random_state=rand_seed)

		# data splitting
		train_test = 0.5
		msk = np.random.rand(len(in_df_)) < train_test

		# train data
		in_df = in_df_[msk] # train

		samp = np.array(in_df[input_list].values)
		valu = np.array(in_df[output_list[0]].values)
		if i == 0:
			samples = samp
			values = valu
		else:
			samples = np.concatenate((samples,samp),axis=0)
			values = np.concatenate((values,valu),axis=0)

		# test data
		test = in_df_[~msk] # test

		test_samp = np.array(test[input_list].values)
		test_valu = np.array(test[output_list[0]].values)
		if i == 0:
			test_samples = test_samp
			test_values = test_valu
		else:
			test_samples = np.concatenate((test_samples,test_samp),axis=0)
			test_values = np.concatenate((test_values,test_valu),axis=0)

	#standardize outputs
	# values = np.around((values-np.mean(values))/np.std(values),decimals=4)
	# values[np.isnan(values)] = 0
	# test_values = np.around((test_values-np.mean(test_values))/np.std(test_values),decimals=4)
	# test_values[np.isnan(test_values)] = 0

	return input_list, output_list, stand(samples), values, stand(test_samples), test_values 

input_list, output_list, samples, values, test_samples, test_values  = data_('../../sym_data_4.csv',rand_seed=rand_seed)

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

if not scoop.IS_ORIGIN:
	pset.addEphemeralConstant("rand10", lambda: np.random.randint(0,100)/10.,float)
	pset.addEphemeralConstant("rand100", lambda: np.random.randint(0,100)/1.,float)

def boolTree(tree,samples=samples):
	func = toolbox.compile(expr=tree)
	vals = func(*samples.T)
	lens = len(tree)/90.
	return vals,lens

def evalInd(individual,samples=samples,values=values):
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

if __name__ == "__main__":

	checkpoint = None

	# set random seed
	random.seed(rand_seed)
	np.random.seed(rand_seed)

	toolbox.register("map",scoop.futures.map)

	NGEN = 20000
	STAG = 2500
	FREQ = 2000
	MU = 96 # must be multiple of 4 for selection
	CXPB = 0.2

	dir_path = os.path.dirname(os.path.realpath(__file__))
	local_path = os.path.join(dir_path,'seed_{}_results'.format(rand_seed))
	if 'seed_{}_results'.format(rand_seed) in os.listdir(dir_path):
			rmtree('seed_{}_results'.format(rand_seed)) # only turn on if need to do again
	os.mkdir(local_path)
	os.chdir(local_path)

	if checkpoint:
		# A file name has been given, then load the data from the file
		with open(checkpoint, "r") as cp_file:
			cp = pickle.load(cp_file)
		pop = cp["population"]
		start_gen = cp["generation"]
		hof = cp["halloffame"]
		logbook = cp["logbook"]
		random.setstate(cp["rndstate"])
		np.random.set_state(cp["rndstate_np"])
	else:
		# Start a new evolution
		pop = toolbox.population(n=MU)
		start_gen = 0
		hof = tools.ParetoFront()
		logbook = tools.Logbook()
		logbook.header = "gen", "evals", "min", "avg", "max", "std"

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	
	stats.register("min", np.min, axis=0)
	stats.register("avg", np.mean, axis=0)
	stats.register("max", np.max, axis=0)
	stats.register("std", np.std, axis=0)

	# Evaluate the individuals with an invalid fitness
	invalid_ind = [ind for ind in pop if not ind.fitness.valid]
	fitnesses = toolbox.map(toolbox.evalInd, invalid_ind)
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit

	# This is just to assign the crowding distance to the individuals
	# no actual selection is done
	pop = toolbox.select(pop, len(pop))
	
	record = stats.compile(pop)
	logbook.record(gen=0, evals=len(pop), **record)
	print(logbook.stream)

	# Begin the generational process
	gen = start_gen
	count = 0
	convergence = []
	
	while gen < NGEN and count < STAG:
		# Vary the population
		offspring = tools.selTournamentDCD(pop, len(pop))
		offspring = [toolbox.clone(ind) for ind in pop]

		for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
			for i,(ind1_,ind2_) in enumerate(zip(ind1,ind2)):
				if random.random() <= CXPB:
					toolbox.mate(ind1_, ind2_)	
				ind1_ = toolbox.mutate(ind1_) # bug was here!
				ind2_ = toolbox.mutate(ind2_)
				ind1[i] = ind1_[0]
				ind2[i] = ind2_[0]

			del ind1.fitness.values, ind2.fitness.values

		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evalInd, invalid_ind)

		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# Select the next generation population
		pop = toolbox.select(pop + offspring, MU) # selection every n gens
		
		record = stats.compile(pop)
		logbook.record(gen=gen, evals=len(pop), **record)
		hof.update(pop)

		min_ = record['min'][0]

		if gen < 10:
			convergence.append(min_) # first few rounds
			print(gen, count, min_, np.nanmin(convergence))
		else:
			print(gen, count, min_, np.nanmin(convergence))
			if np.nanmin(convergence) > min_:
				count = 0 # reset count if new min is found
				print('New minimum found: {}'.format(min_))
			convergence.append(min_)
		
		# increment generation
		gen += 1

		# increment count
		count += 1

		if gen % FREQ == 0:
			# Fill the dictionary using the dict(key=value[, ...]) constructor
			cp = dict(trial=trial, population=pop, generation=gen, halloffame=hof,
				logbook=logbook, inputs = input_list, outputs = output_list,
				rndstate=random.getstate(), rndstate_np=np.random.get_state())

			with open("trial_{0}_checkpoint_{1}.pkl".format(trial,gen), "wb") as cp_file:
				pickle.dump(cp, cp_file)

	# pickle results first
	data = dict(trial = rank,
				population = pop,
				# history=history,
				halloffame = hof,
				logbook = logbook,
				# toolbox = toolbox,
				inputs = input_list, 
				outputs = output_list, 
				samples = samples, 
				values = values, 
				test_samples = test_samples, 
				test_values = test_values,
				rndstate = random.getstate(),   
				rndstate_np = np.random.get_state())

	with open('sc_{}_results.pickle'.format(trial), 'wb') as f:
		# Pickle the 'data' dictionary using the highest protocol available.
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

	# plotting from here down

	gen, smin = logbook.select("gen", "min")

	front = np.array([ind.fitness.values for ind in pop if ind.fitness.values[0] < 1.2 ])
	test_front = np.asarray([toolbox.evalInd(ind,samples=test_samples,values=test_values) for ind in pop if ind.fitness.values[0] < 1.2])

	fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize=(12,5), dpi=300)
	
	ax1.plot(gen,np.vstack(smin)[:,0],c="c")
	ax1.set_title('Convergence')
	ax1.set_xlabel('Generation')
	ax1.set_ylabel('Performance')

	ax2.scatter(front[:,1], front[:,0], c="b",label='Train')
	ax2.scatter(front[:,1],test_front[:,0], c="r",label='Test')
	ax2.set_title('Tradeoff')
	ax2.set_xlabel('Complexity')
	ax2.legend()

	fig.savefig('sc_{}_plot.png'.format(trial))


	fits = np.zeros(len(hof))
	lens = np.zeros(len(hof))
	tests = np.zeros(len(hof))
	
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

	for i,item in enumerate(hof):
		fits[i] = item.fitness.values[0]
		tests[i] = func(item,samples=test_samples,values=test_values)[0]
		lens[i] = item.fitness.values[1]

	min_ = np.argmin(fits+tests)

	fig, ax = plt.subplots(1,1,figsize=(6,5))
	ax.scatter(lens,fits,label='training')
	ax.scatter(lens,tests,label='testing')
	ax.set_xlabel('Complexity (num. of nodes)')
	ax.set_ylabel('Error of Individual')

	ax.legend()
	plt.savefig('sc_{}_front.png'.format(trial),dpi=300)

	best = hof[min_]

	A = metrics.confusion_matrix(y_true=values, 
					  y_pred=func2(best,samples), 
					  )

	B = metrics.confusion_matrix(y_true=test_values, 
						  y_pred=func2(best,test_samples), 
						  )

	print(A)
	num_right_train = A[0,0]+A[1,1]+A[2,2]
	total_train = np.sum(A)
	percent_train = 100*(num_right_train/total_train)
	print(percent_train)
	print(B)
	num_right_test = B[0,0]+B[1,1]+B[2,2]
	total_test = np.sum(B)
	percent_test = 100*(num_right_test/total_test)
	print(percent_test)

	A,B =  A.astype(float),B.astype(float)
	for i in np.arange(0,A.shape[0]):
		A_sum = np.sum(A[i])
		B_sum = np.sum(B[i])
		A[i] = np.divide(A[i],A_sum)
		B[i] = np.divide(B[i],B_sum)

	fig,ax = plt.subplots()

	im1 = triamatrix(A, ax, rot=0, cmap="Blues")
	im2 = triamatrix(B, ax, rot=180, cmap="Reds")
	ax.set_xlim(-0.5,A.shape[1]-0.5)
	ax.set_ylim(-0.5,A.shape[0]-0.5)
	labels = [item.get_text() for item in ax.get_xticklabels()]
	labels[1],labels[3],labels[5] = '-1','0','+1'
	ax.set_xticklabels(labels)
	labels = [item.get_text() for item in ax.get_yticklabels()]
	labels[1],labels[3],labels[5] = '-1','0','+1'
	ax.set_yticklabels(labels)
	for label in ax.xaxis.get_ticklabels()[::2]:
		label.set_visible(False)
	for label in ax.yaxis.get_ticklabels()[::2]:
		label.set_visible(False)
	cbar1 = fig.colorbar(im1, ax=ax, )
	cbar1.ax.set_ylabel('Training')
	cbar2 = fig.colorbar(im2, ax=ax, )
	cbar2.ax.set_ylabel('Testing')
	plt.title('Training accuracy: %2.2f %%, Testing accuracy: %2.2f %%' % (percent_train,percent_test))
	plt.ylabel('True Change')
	plt.xlabel('Predicted Change')

	plt.savefig('sc_{}_confusion.png'.format(trial),dpi=300)

	# for j,ind in enumerate(hof[min_]):
	# 	nodes, edges, labels = gp.graph(ind)
	# 	g = AGraph()
	# 	g.graph_attr['overlap'] = False
	# 	g.graph_attr['size'] = '6,6'
	# 	g.graph_attr['dpi'] = 1200
	# 	g.node_attr['fontsize'] = 8
	# 	g.add_nodes_from(nodes)
	# 	g.add_edges_from(edges)
	# 	g.layout(prog="neato")

	# 	for i in nodes:
	# 		n = g.get_node(i)
	# 		n.attr["label"] = labels[i]

	# 	g.draw('sc_{0}_tree_{1}.png'.format(trial,j))

	os.chdir(dir_path)

