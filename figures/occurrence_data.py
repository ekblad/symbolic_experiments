# plotting results

import pickle
import os
import operator
import pandas as pd
import numpy as np
import seaborn as sns

from collections import Counter
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path,'sc_all_results.pickle')
data = pickle.load(open(file_path,'rb'))

funcs = ['lt','if_then_else','vadd','vsub','vmul','vdiv','vexp','vlog','vneg','vsin','vcos','vtan']
# for i in data:
# 	print(i)
inputs = data['seed_0_results']['inputs']
# inputs_df = ['model_ID'] + inputs
# input_df = pd.DataFrame(columns= inputs_df)

func_list = []
input_list = []
term_list = []

func_dicts = {}
input_dicts = {}
training_error = []
test_error = []
complexity = []

for i in data:
	if i[5:7] ==  '63': # or i[5:7] ==  '62':
		continue

	for count,j in enumerate(data[i]['labels']):
		# # print(j) # a tree
		training_error.append(data[i]['train_mse'][count])
		test_error.append(data[i]['test_mse'][count])
		complexity.append(data[i]['complexity'][count])



		model_id = 'seed_{0}_model_{1}'.format(i[5:7],count)
		print(model_id)

		# model_terms = []
		model_funcs = []
		model_inputs = []

		for k in range(0,3):
			for l in j[k]:
				if j[k][l] in funcs:
					func_list.append(j[k][l])
					model_funcs.append(j[k][l])
				elif j[k][l] in inputs:
					input_list.append(j[k][l])
					model_inputs.append(j[k][l])
				else:
					term_list.append(float(j[k][l]))

		func_counts = []
		input_counts = []

		for l in funcs:
			if l in model_funcs:
				func_counts.append(model_funcs.count(l))
			else:
				func_counts.append(0)
		# print(func_counts)
		# exit()
		for l in inputs:
			if l in model_inputs:
				input_counts.append(model_inputs.count(l))
			else:
				input_counts.append(0)

		# func_counts = np.array(func_counts,ndmin=1).T
		# input_counts = np.array(input_counts,ndmin=1).T
			# print(input_counts)
			# exit()

			# input_dict = {}
			# input_dict[model_id] = np.divide(input_counts,np.sum(input_counts))
			# func_dict = {}
			# func_dict[model_id] = np.divide(func_counts,np.sum(func_counts))
			# print(input_dict)
			# print(func_dict)
			# input_dicts.append(input_dict)
			# func_dicts.append(func_dict)
		# print(np.sum(input_counts))
		if not input_counts or np.sum(input_counts) == 0:
			# print('hey')
			input_dicts[model_id] = np.zeros((len(inputs),))
		else:
			input_dicts[model_id] = np.divide(input_counts,np.sum(input_counts))

		if not func_counts or np.sum(func_counts) == 0:
			# print('hey')
			func_dicts[model_id] = np.zeros((len(funcs),))
		else:
			func_dicts[model_id] = np.divide(func_counts,np.sum(func_counts))
		# func_dicts[model_id] = np.divide(func_counts,np.sum(input_counts))
		# print(input_dicts)
		# # print(func_dicts)
		# print(pd.DataFrame.from_dict(input_dicts))
		# # print(pd.DataFrame(func_dicts,index=funcs))
		# exit()
input_df = pd.DataFrame(input_dicts,index=inputs)
func_df = pd.DataFrame(func_dicts,index=funcs)
print(input_df)
print(func_df)
frames = [input_df,func_df]
infn_df = pd.concat(frames)
# print(infn_df.T)
infn_df = infn_df.T
infn_df['training_error'] = training_error
infn_df['complexity'] = complexity
infn_df['test_error'] = test_error
print(infn_df)
infn_df.to_csv('sc_cluster_data.csv')
exit()

# input_df.to_csv('tree_input_freq.csv')
# func_df.to_csv('tree_func_freq')

# input_counts = Counter(input_list)
# # print(input_counts)
# sorted_input = sorted(input_counts.items(), key=operator.itemgetter(1))
# # print(sorted_input)
# labelz = []
# countz = []
# for i in sorted_input:
# 	if i[1] > 100:

# 		labelz.append(i[0])
# 		countz.append(i[1])

# # Initialize the figure with a logarithmic x axis
# f, ax = plt.subplots(figsize=(10, 7))
# # ax.set_xscale("log")

# # Load the example planets dataset
# input_df=input_df.loc[labelz].unstack(level=-1) #.melt(value_vars=input_df.columns)
# input_df = input_df[input_df.values != 0]
# print(input_df)

# input_plot = pd.DataFrame()
# input_plot['Inputs'] = input_df.index.get_level_values(1)
# input_plot['Input Occurrence Percentage per Model'] = input_df.values
# input_plot.Inputs = input_plot.Inputs.astype("category")
# input_plot.Inputs.cat.set_categories(labelz, inplace=True)
# input_plot.sort_values(["Inputs"])
# # input_plot.set_index('Inputs')
# # print(input_plot)
# # labelz_2 = input_df.index.get_level_values(1)
# # input_plot = input_plot.loc[list(set(labelz))]


# # planets = sns.load_dataset("planets")
# # print(planets)
# # exit()

# # Plot the orbital period with horizontal boxes
# sns.boxplot(x='Input Occurrence Percentage per Model', y='Inputs', data=input_plot,
#             whis="range", palette="vlag")

# # # Add in points to show each observation
# # sns.swarmplot(x='occurrence_percentages', y='inputs', data=input_plot,
# #               size=2, color=".3", linewidth=0)

# # Tweak the visual presentation
# # ax.xaxis.grid(True)
# ax.set(ylabel="")
# ax.set(xlim=[0,1])
# plt.gcf().subplots_adjust(left=0.3)
# # sns.despine(trim=True, left=True)
# plt.savefig('input_dists.png',dpi=1000)
# # exit()
# # fig, ax = plt.subplots(1,1,figsize=(10,8))
# # ax.scatter(countz,labelz,color='blue',s=0.8)
# # # ax.scatter(data[i]['complexity'],data[i]['test_mse'],color='orange',s=0.5)
# # # for j in data[i]:
# # # 	print(j)
# # # exit()
# # ax.set_xlabel('Count')
# # # ax.set_ylabel('Count')
# # ax.set_title('Input Feature Frequency')

# # plt.xscale('log')
# # plt.xlim(100,10000)
# # # plt.show()
# # plt.xticks(rotation=90)
# # plt.rc('font', size=1)
# # plt.gcf().subplots_adjust(left=0.3)
# # plt.savefig('input_dists.png',dpi=1000)

# fig, ax = plt.subplots(1,1,figsize=(10,8))
# ax.scatter(countz,labelz,color='blue',s=0.8)
# # ax.scatter(data[i]['complexity'],data[i]['test_mse'],color='orange',s=0.5)
# # for j in data[i]:
# # 	print(j)
# # exit()
# ax.set_xlabel('Count')
# # ax.set_ylabel('Count')
# ax.set_title('Input Feature Frequency')

# plt.xscale('log')
# plt.xlim(100,10000)
# # plt.show()
# plt.xticks(rotation=90)
# plt.rc('font', size=1)
# plt.gcf().subplots_adjust(left=0.3)
# plt.savefig('inputs.png',dpi=1000)

# func_counts = Counter(func_list)
# # print(input_counts)
# sorted_funcs = sorted(func_counts.items(), key=operator.itemgetter(1))
# print(sorted_funcs)
# labelz = []
# countz = []

# for i in sorted_funcs:
# 	if i[1] > 100:
# 		labelz.append(i[0])
# 		countz.append(i[1])

# f, ax = plt.subplots(figsize=(7, 7))
# # ax.set_xscale("log")

# # Load the example planets dataset
# func_df=func_df.loc[labelz].unstack(level=-1) #.melt(value_vars=input_df.columns)
# func_df = func_df[func_df.values != 0]
# print(func_df)

# func_plot = pd.DataFrame()
# func_plot['Functions'] = func_df.index.get_level_values(1)
# func_plot['Function Occurrence Percentage per Model'] = func_df.values
# func_plot.Functions = func_plot.Functions.astype("category")
# func_plot.Functions.cat.set_categories(labelz, inplace=True)
# func_plot.sort_values(["Functions"])
# # input_plot.set_index('Inputs')
# # print(input_plot)
# # labelz_2 = input_df.index.get_level_values(1)
# # input_plot = input_plot.loc[list(set(labelz))]


# # planets = sns.load_dataset("planets")
# # print(planets)
# # exit()

# # Plot the orbital period with horizontal boxes
# sns.boxplot(x='Function Occurrence Percentage per Model', y='Functions', data=func_plot,
#             whis="range", palette="vlag")

# # # Add in points to show each observation
# # sns.swarmplot(x='occurrence_percentages', y='inputs', data=input_plot,
# #               size=2, color=".3", linewidth=0)

# # Tweak the visual presentation
# # ax.xaxis.grid(True)
# ax.set(ylabel="")
# ax.set(xlim=[0,1])
# plt.gcf().subplots_adjust(left=0.17)
# # sns.despine(trim=True, left=True)
# plt.savefig('func_dists.png',dpi=1000)

# fig, ax = plt.subplots(1,1,figsize=(6.5,6.5))
# ax.scatter(countz,labelz,color='blue',s=3.0)
# # ax.scatter(data[i]['complexity'],data[i]['test_mse'],color='orange',s=0.5)
# # for j in data[i]:
# # 	print(j)
# # exit()
# ax.set_xlabel('Count')
# # ax.set_ylabel('Count')
# ax.set_title('Function Frequency')

# plt.xscale('log')
# plt.xlim(1000,10000)
# # plt.show()
# plt.xticks(rotation=90)
# plt.rc('font', size=3.0)
# plt.gcf().subplots_adjust(left=0.17)
# plt.savefig('funcs.png',dpi=1000)







# exit()
# # df = pd.DataFrame.from_dict(input_counts, orient='index')
# # df.plot(kind='bar')
# # pd.Series(input_list).value_counts().plot('area')
# # plt.show()

# # exit()

# fig, ax = plt.subplots(1,1,figsize=(6.5,6.5))

# # # plt.axhline(1,linestyle='--',color='r',label='100% error')

# # # plt.ylim(0,1000)
# # # plt.yscale('log')
# # ax.legend()
# # plt.savefig('gp_{0}_front.png'.format(rand_seed),dpi=300)
# for i in data:

# 	ax.scatter(data[i]['complexity'],data[i]['train_mse'],color='blue',s=0.5)
# 	ax.scatter(data[i]['complexity'],data[i]['test_mse'],color='orange',s=0.5)
# 	# for j in data[i]:
# 	# 	print(j)
# 	# exit()
# ax.set_xlabel('# NODES')
# ax.set_ylabel('MSE')
# ax.set_title('Perennial Crop Changes - Regression')

# # plt.yscale('log')
# plt.ylim(0.75,1.3)
# # plt.show()
# plt.savefig('pareto_front_tree.png',dpi=300)



