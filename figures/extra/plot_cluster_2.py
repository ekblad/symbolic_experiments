
import os
import copy
import pandas as pd
import numpy as np
import scipy as sp
import pareto
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler, colors
from plot_utils import *
import pygraphviz as pgv
import networkx as nx
from shutil import rmtree
import ast
from sklearn.cluster import KMeans

params = {
			# 'text.latex.preamble': ['\\usepackage{gensymb}'],
			# 'text.usetex': True,
			'font.family': 'Helvetica',
			}
rcParams.update(params)
from matplotlib.lines import Line2D

sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 1})
sns.set_style('white')
sns.set_palette("cividis")

def main():
	random_seed = 0
	data = pd.read_csv('model_sensitivities.csv',header=0,index_col=0,low_memory=False)
	data = data.dropna()

	data_base = data[data.cluster == 3][data.experiment == 'classification']
	base_metrics = copy.copy(data_base[['model_id','training_error', 'complexity', 'test_error']]).set_index('model_id').sample(frac=1.0,random_state=random_seed)
	num_clusters = 5 # number of clusters
	est = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1, random_state=random_seed)
	kmeans = est.fit(base_metrics)
	base_metrics['cluster_2'] = est.labels_
	base_metrics.reset_index(inplace=True)
	


	of_cols = [1,2,3]
	titles = ['Negative','No Change','Positive']
	# base_nondom = pareto.eps_sort([list(base_metrics.itertuples(False))], of_cols)
	# base_sorted = pd.DataFrame.from_records(base_nondom, columns=list(base_metrics.columns.values))

	dir_path = os.path.dirname(os.path.realpath(__file__))
	local_path = os.path.join(dir_path,'baseline_clusters')
	if 'baseline_clusters' in os.listdir(dir_path):
		rmtree('baseline_clusters') # only turn on if need to do again
	os.mkdir(local_path)
	os.chdir(local_path)

	fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(8,8))
	ax = axes.flatten()
	for cluster in base_metrics.cluster_2.unique():
		for model in base_metrics[base_metrics.cluster_2 == cluster].model_id.unique():
			
			nodes = ast.literal_eval(data_base[data_base['model_id']==model]['nodes'].unique()[0])
			edges = ast.literal_eval(data_base[data_base['model_id']==model]['edges'].unique()[0])
			labels = ast.literal_eval(data_base[data_base['model_id']==model]['labels'].unique()[0])

			for i,(node,edge,label) in enumerate(zip(nodes,edges,labels)):

				G=nx.DiGraph()
				G.add_nodes_from(node)
				G.add_edges_from(edge)
				plt.sca(ax[i])
				ax[i].set_title(titles[i])
				pos=nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
				nx.draw(G, pos,ax=ax[i], labels=label, with_labels = True, arrows = False)
				ax[i].set_axis_off()
			ax[-1].scatter(x=base_metrics.complexity,y=base_metrics.training_error,c='slategray',marker='.',alpha=0.3) #label='Cluster '+ str(cluster),s=marker_size,)
			ax[-1].scatter(x=base_metrics.complexity,y=base_metrics.test_error,c='slategray',marker='+',alpha=0.3)
			ax[-1].scatter(x=base_metrics[base_metrics.cluster_2 == cluster].complexity,y=base_metrics[base_metrics.cluster_2 == cluster].training_error,c='midnightblue',marker='.',alpha=0.3) #label='Cluster '+ str(cluster),s=marker_size,)
			ax[-1].scatter(x=base_metrics[base_metrics.cluster_2 == cluster].complexity,y=base_metrics[base_metrics.cluster_2 == cluster].test_error,c='midnightblue',marker='+',alpha=0.3)
			ax[-1].scatter(x=base_metrics[base_metrics['model_id']==model].complexity,y=base_metrics[base_metrics['model_id']==model].training_error,c='Blue',marker='.')
			ax[-1].scatter(x=base_metrics[base_metrics['model_id']==model].complexity,y=base_metrics[base_metrics['model_id']==model].test_error,c='Red',marker='+')
			# ax[-1].scatter(x=base_sorted.complexity,y=base_sorted.training_error,c='slategray',marker='x',alpha=0.3) #label='Cluster '+ str(cluster),s=marker_size,)
			# ax[-1].scatter(x=base_sorted.complexity,y=base_sorted.test_error,c='slategray',marker='+',alpha=0.3)
			# ax[-1].scatter(x=base_sorted[base_sorted['model_id']==model].complexity,y=base_sorted[base_sorted['model_id']==model].training_error,c='Blue',marker='x')
			# ax[-1].scatter(x=base_sorted[base_sorted['model_id']==model].complexity,y=base_sorted[base_sorted['model_id']==model].test_error,c='Red',marker='+')
			# ax[-1].set_ylim(bottom=0.3,top=0.5)
			# ax[-1].set_xlim(left=0,right=0.2)
			plt.savefig('{}.eps'.format(model),format='eps',bbox_inches='tight',dpi=600,transparent=True)
			for i,a in enumerate(ax):
				plt.sca(ax[i])
				plt.cla()
	os.chdir(dir_path)

if __name__ == '__main__':
	main()

exit()

plots = {'category':'Category',
		'neighborhood':'Neighborhood',
		'lag':'Lag'
		} # three in total
orders = {'category':['land_tree','land_non','water_pump','water_deliv','econ_tree','econ_non'],
		'neighborhood':['home','neighbor_1','neighbor_2','neighbor_3','neighbor_4','neighbor_5'],
		'lag':['present','lag1','lag2','lag3','lag4','lag5','lag6'],
		# 'cluster':[0,1,2,3,4],
		# 'color':['midnightblue','Red']
		}

def translate(item):
	if item == 'land_tree':
		return 'Tree Acreage'
	if item == 'land_non':
		return 'Non-Tree Acreage'
	if item == 'water_pump':
		return 'Water Pumping'
	if item == 'water_deliv':
		return 'Water Deliveries'
	if item == 'econ_tree':
		return 'Tree Prices/Values'
	if item == 'econ_non':
		return 'Non-Tree Prices/Values'
	if item == 'home':
		return 'Current Plot Data'
	if item == 'neighbor_1':
		return 'Neighbor 1 Data'
	if item == 'neighbor_2':
		return 'Neighbor 2 Data'
	if item == 'neighbor_3':
		return 'Neighbor 3 Data'
	if item == 'neighbor_4':
		return 'Neighbor 4 Data'
	if item == 'neighbor_5':
		return 'Neighbor 5 Data'
	if item == 'present':
		return 'Present Data'
	if item == 'lag1':
		return "Previous Year's Data"
	if item == 'lag2':
		return 'Two Years Previous'
	if item == 'lag3':
		return 'Three Years Previous'
	if item == 'lag4':
		return 'Four Years Previous'
	if item == 'lag5':
		return 'Five Years Previous'
	if item == 'lag6':
		return 'Six Years Previous'

sensitivity = 'ST' # plotting for total sensitivity indices
data[sensitivity] = [1. if i > 2 else i for i in data[sensitivity].values] # clean outliers
data = data[data.experiment == 'classification']
data_base = data[data.cluster == 2]
data_1 = data[data.cluster == 0]
data_3 = data[data.cluster == 1]
datas = [data_base,data_1,data_3]
col_list = ['Baseline Cluster','Robust Cluster','Overfit Cluster']
df_store = pd.DataFrame(columns = col_list)
switch = True
for order in orders:
	for ind in orders[order]:
		nums = []
		for i,data in enumerate(datas):
			nums.append(np.mean((data[data[order]==ind][sensitivity].values>0)))

			df_work = pd.DataFrame()
			df_work['Sensitivity'] = [j for j in data[data[order]==ind][sensitivity].values]# if j > 0 ]
			df_work['Cluster'] = [col_list[i] for j in df_work['Sensitivity'].values]
			df_work['Category'] = [translate(ind) for j in df_work['Sensitivity'].values]

			df_bins = pd.DataFrame()
			df_bins['Bins'] = np.arange(0,1.01,0.01)
			df_bins['Cluster'] = [col_list[i] for j in df_bins['Bins'].values]
			df_bins['Category'] = [translate(ind) for j in df_bins['Bins'].values]
			df_bins['CDF'] = [np.sum(df_work['Sensitivity'].values < j)/len(df_work['Sensitivity'].values) for j in df_bins['Bins'].values]
			if switch:
				df_sens = df_work
				df_cdf = df_bins
				switch = False
			else:
				df_sens = pd.concat([df_sens,df_work])
				df_cdf = pd.concat([df_cdf,df_bins])
		df_store.loc[ind] = nums

df_store = df_store.stack()
print(df_store)
# exit()
df_plot = pd.DataFrame()
df_plot['Category'] = [translate(i) for i in df_store.index.get_level_values(0)]
df_plot['Cluster'] = df_store.index.get_level_values(1)
df_plot['Average Sensitivity'] = df_store.values
print(df_plot)
# exit()

idx = pd.IndexSlice
df_cdf = df_cdf.set_index(['Category','Bins','Cluster']).unstack(level=-1)
df_cdf.loc[:,idx['CDF','Robust Difference']] = df_cdf.loc[:,idx['CDF','Robust Cluster']] - df_cdf.loc[:,idx['CDF','Baseline Cluster']]
df_cdf.loc[:,idx['CDF','Overfit Difference']] = df_cdf.loc[:,idx['CDF','Overfit Cluster']] - df_cdf.loc[:,idx['CDF','Baseline Cluster']]
df_cdf = df_cdf.stack()
df_cdf['Category'] = df_cdf.index.get_level_values(0)
df_cdf['Bins'] = df_cdf.index.get_level_values(1)
df_cdf['Cluster'] = df_cdf.index.get_level_values(2)

colors = ['midnightblue','Blue','Red','Blue','Red'] #,'c','m','y','b']
styles=['-','-','-','--','--']
alpha=0.18
x_pos=[0.0512,0.025,]
good_diffs = []
overfit_diffs = []
labels = []
good_avgs = []
over_avgs = []
base_avgs = []

# fig,ax = plt.subplots(1,2,sharex=True,sharey=True,squeeze=True,figsize=(12,8))

for i,cat in enumerate(df_sens['Category'].unique()[:-1]):
	data_plot = df_cdf[df_cdf['Category'] == cat]
	labels.append(cat)
	# a = ax.flat[i]
	# baseline = data_plot[data_plot['Cluster'] == 'Baseline Cluster']['Bins'].values
	for j,clust in enumerate(data_plot['Cluster'].unique()):
		if clust not in ['Robust Difference','Overfit Difference']:
			continue
		x = data_plot[data_plot['Cluster'] == clust]['Bins'].values
		y = data_plot[data_plot['Cluster'] == clust]['CDF'].values		
		int_ = sp.integrate.simps(y, x=x)
		if clust == 'Robust Difference':
			good_diffs.append(int_)
		elif clust == 'Overfit Difference':
			overfit_diffs.append(int_)
	for j,clust in enumerate(df_plot['Cluster'].unique()):
		val = df_plot[(df_plot['Cluster'] == clust) & (df_plot['Category'] == cat)]['Average Sensitivity'].values
		if clust == 'Baseline Cluster':
			base_avgs.append(val)
		elif clust == 'Robust Cluster':
			good_avgs.append(val)
		elif clust == 'Overfit Cluster':
			over_avgs.append(val)

print(base_avgs)
# fig,ax = plt.subplots(figsize=(3,2.5))
shape = (3,6)
base = np.flipud(np.array(base_avgs).reshape(shape))
good = np.flipud(np.array(good_avgs).reshape(shape))
over = np.flipud(np.array(over_avgs).reshape(shape))
# labels = np.flipud(np.array(labels).reshape(shape))
# all_ = np.flipud(np.array([base_avgs,good_avgs,over_avgs]))

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10,7))
# ax.figure.tight_layout()
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Generate a custom diverging colormap
cmap = sns.cubehelix_palette(start=2.75, rot=0.0, dark=0, light=0.99, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(df_plot.pivot(index='Cluster', columns='Category', values='Average Sensitivity')[labels],
			# mask=mask,
			cmap=cmap, 
			vmax=np.max(df_plot.pivot(index='Cluster', columns='Category', values='Average Sensitivity')[labels].values),
			vmin=0.,
			# center=0,
			square=True,
			linewidths=0.7,
			annot=False, 
			cbar=True,
			cbar_kws={"shrink":.3,'pad':0.01,'use_gridspec':False,'location':'top','anchor':(1,0)}
			)

cbar_ax = fig.axes[-1]
cbar_ax.tick_params(axis=u'both', which=u'both',length=0)
# ax.figure.subplots_adjust(left = 0.1) # change 0.3 to suit your needs.
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, horizontalalignment='right',)
ax.set_title('Average Sensitivity',ha='right')
fig.savefig('plot_avg_sens_heatmap.pdf',format='pdf',bbox_inches='tight',dpi=600,transparent=True)

