import pickle
import os
import operator
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
# import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set(style="white",palette="GnBu",context='paper',color_codes=True) #,rc={"axes.facecolor": (0, 0, 0, 0),'figure.figsize':(14,10)})
# sns.set(font_scale=1.3)
sns.set_style('white')
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path,'sc_sensitivity.pickle')

np.random.seed(0)
data = pd.read_csv('sc_cluster_data.csv',header=0,index_col=0).sample(frac=1.0)
data.index = [val if val[6] is not '_' else val[:6] + val[7:] for val in data.index.values] # oops, used two naming conventions
funcs = ['lt', 'ite', 'vadd', 'vsub', 'vmul', 'vdiv', 'vneg', 'vsin', 'vcos']

data2 = data[data.columns[-3:]]
scaler = StandardScaler().fit(data2)
clustering = scaler.transform(data2)

# for name, est in estimators:
est = KMeans(n_clusters=3)
est.fit(clustering)
labels = est.labels_
s = pd.Series(labels)

clust = pd.DataFrame(index=data.index.values)
clust['cluster'] = s.values
group = {}

with open(file_path,'rb') as f:
	sens_data = pickle.load(f)

sens = {}
input_list = defaultdict(list)

for i in sens_data:
	for j in sens_data[i]:
		sens[j] = {}
		for ind_,k in enumerate(sens_data[i][j]['inputs']):
			input_list[j].append(k)
			sens[j][k] = sens_data[i][j]['sensitivity_indices']['ST'][ind_]#.append(sens_data[i][j]['sensitivity_indices']['ST'][ind_])
sens = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in sens.items() ]))
meds = sens.median(axis=1,skipna=True)
meds.sort_values(ascending=False,inplace=True)

meds = meds[meds.values>0.1]
sens.to_csv('model_sensitivities.csv')
input_list = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in input_list.items() ]))
count_nans = [1  if str(inp) != 'nan' else 0 for inp in input_list.values.flatten()]
print(count_nans)
print(np.sum(count_nans),len(count_nans))
exit()
all_input_list = [inp for inp in input_list.values.flatten() if str(inp) != 'nan']
input_counts = Counter(all_input_list)
sorted_input = sorted(input_counts.items(), key=operator.itemgetter(1))

labelz = meds.index.values
countz = pd.DataFrame(columns=['Inputs','Counts','Cluster'])
sensitiv = pd.DataFrame(columns=['Inputs','Sensitivity','Cluster'])

for l in [0,1,2]:
	group[l] =  list(clust[clust.cluster == l].index.values)

	cluster_input_list = [inp for inp in input_list[group[l]].values.flatten()]
	cluster_input_list = [inp for inp in cluster_input_list if str(inp) != 'nan']

	input_counts = Counter(cluster_input_list)
	input_counts = dict((k, input_counts[k]) for k in labelz if k in input_counts)
	input_counts_df = pd.DataFrame.from_dict(input_counts,orient='index')
	input_counts_df.columns=['Counts']
	input_counts_df.Counts = input_counts_df.Counts.values/len(group[l])
	input_counts_df['Inputs']=input_counts_df.index
	input_counts_df = input_counts_df.reset_index().drop(['index'],axis=1)
	input_counts_df['Cluster'] = ['Cluster_{}'.format(l) for val in input_counts_df.index]

	countz = countz.append(input_counts_df)

	plot_ = sens[group[l]].T[labelz].unstack(level=-1)

	input_plot = pd.DataFrame()
	input_plot['Inputs'] = plot_.index.get_level_values(0)
	input_plot['Sensitivity'] = plot_.values
	input_plot['Cluster'] = ['Cluster_{}'.format(l) for val in plot_.index.values]
	input_plot = input_plot[input_plot.Sensitivity>=0]
	input_plot = input_plot[~np.isnan(input_plot.Sensitivity)]
	
	sensitiv = sensitiv.append(input_plot)

cluster_list = ['Cluster_1','Cluster_2']
countz=countz[['Inputs','Counts','Cluster']]
countz = countz[countz.Cluster != 'Cluster_0']
countz.Inputs = pd.Categorical(countz.Inputs,categories=labelz,ordered=True)
countz.Cluster = pd.Categorical(countz.Cluster,categories=cluster_list,ordered=True)
countz.sort_values(['Inputs','Cluster'],ascending=['True','True'],inplace=True)

sensitiv = sensitiv[sensitiv.Cluster != 'Cluster_0']
sensitiv.Inputs = pd.Categorical(sensitiv.Inputs,categories=labelz,ordered=True)
sensitiv.Cluster = pd.Categorical(sensitiv.Cluster,categories=cluster_list,ordered=True)
sensitiv.sort_values(['Inputs','Cluster'],ascending=['True','True'],inplace=True)

print(countz.Inputs.values)
print(sensitiv.Cluster.unique())

# plotting
print(countz)
print(sensitiv)

colors = ['Red','Blue'] #,'c','m','y','b']

f, axes = plt.subplots(1,2,figsize=(14, 7)) #,sharey=True)
# sns.boxplot(x='Occurrence', y='Inputs', sens_data=f_plot,whis="range", palette="vlag",ax=axes[0])
g1 = sns.barplot(x='Counts', y='Inputs', order=labelz, hue='Cluster',hue_order=['Cluster_1','Cluster_2'], data=countz, palette=colors, ax=axes[0])

axes[0].set_xscale('log')

# axes[0].autoscale(enable=True, axis='y', tight=True)
# sns.swarmplot(x='Occurrence', y='Inputs', sens_data=f_plot, size=2, color=".3", linewidth=0,ax=axes[0])
g2 = sns.stripplot(x='Sensitivity', 
			y='Inputs', 
			order=labelz,
			hue='Cluster', 
			hue_order=['Cluster_1','Cluster_2'],
			data=sensitiv, 
			dodge=True,
			jitter=True,
			# legend=False,
			# whis=range,

			palette=colors, 
			ax=axes[1],
			)

axes[0].legend_.remove()
axes[1].legend_.remove()
# for patch in axes[1].artists:
#     r, g, b, a = patch.get_facecolor()
#     patch.set_facecolor((r, g, b, .8))
# sns.swarmplot(x='Sensitivity', y='Inputs', hue='Cluster', data=sensitiv, size=2, color=".3", linewidth=0, ax=axes[1])

# # Add in points to show each observation
# sns.swarmplot(x='occurrence_percentages', y='inputs', sens_data=input_plot,
#               size=2, color=".3", linewidth=0)

# Tweak the visual presentation
# ax.xaxis.grid(True)

axes[0].axvline(np.nanmedian(countz.loc[countz.Cluster == 'Cluster_1'].Counts.values),color='Red')
axes[0].axvline(np.nanmedian(countz.loc[countz.Cluster == 'Cluster_2'].Counts.values),color='Blue')
axes[1].axvline(np.nanmedian(sensitiv.loc[sensitiv.Cluster == 'Cluster_1'].Sensitivity.values),color='Red')
axes[1].axvline(np.nanmedian(sensitiv.loc[sensitiv.Cluster == 'Cluster_2'].Sensitivity.values),color='Blue')

axes[1].set(ylabel="")
axes[0].set(xlabel="Percent of models")
axes[1].get_yaxis().set_visible(False)
axes[0].set(xlim=[0.001,1])
axes[0].set_yticklabels(labelz)
vals = axes[0].get_xticks()
axes[0].set_xticklabels(['{:,.0%}'.format(x) for x in vals])
# axes[0].tight_layout()
# axes[0].autoscale(enable=True, axis='y', tight=True)

# axes[1].autoscale(enable=True, axis='y', tight=True)
plt.gcf().subplots_adjust(left=0.28,wspace = 0.08,right=0.97)
# g1.legend_.remove()
# g2.legend_.remove()
# plt.legend()
# plt.tight_layout()
# plt.gcf().subplots_adjust(left=0.3)
# sns.despine(trim=True, left=True)
plt.savefig('sc_sensitivity.png',dpi=2000)

