
# this script generates a plot of the model space metrics
# - training error/test error versus complexity - and a plot of 
# the test error under clustering of the 3-d metric space

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set(style="white",palette="GnBu",context='paper',color_codes=True)
# sns.set(font_scale=1.3)
sns.set_style('white')
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

np.random.seed(0)
data = pd.read_csv('sr_cluster_data.csv',header=0,index_col=0).sample(frac=1.0)
data.index = [val if val[6] is not '_' else val[:6] + val[7:] for val in data.index.values] 
# to fix mixed naming conventions - if needed

data_metrics = data[data.columns[-3:]] # training, test, complexity columns

# standardize data
scaler = StandardScaler().fit(data_metrics) 
clustering = scaler.transform(data_metrics)

# cluster 3 columns
est = KMeans(n_clusters=3) # visually verify number of clusters are sufficient
est.fit(clustering)
labels = est.labels_
s = pd.Series(labels)
clust = pd.get_dummies(s,prefix='class')
clust.index = data.index.values

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='Blue', label='Training Error',markerfacecolor='Blue'),# markersize=15),
                   Line2D([0], [0], marker='x', color='Red', label='Test Error',markerfacecolor='Red')]#, markersize=15)]
fig, ax = plt.subplots(1,2,figsize=(13,6.5), sharey=True)
for i,label in enumerate(range(0,np.max(labels)+1)):
	ax[0].scatter(data.loc[labels==label].complexity.values,data.loc[labels==label].training_error.values,marker='o',color='Blue',s=20,alpha=0.5)
	ax[0].scatter(data.loc[labels==label].complexity.values,data.loc[labels==label].test_error.values,marker='x',color='Red',s=20, alpha=0.5)
ax[0].set_title('Regression Experiments - Metric Space')
ax[0].set_xlabel('Avg. # Nodes')
ax[0].set_ylabel('Mean Square Error')
ax[0].legend(handles=legend_elements)
# plt.savefig('metric_space_2.png',dpi=1200)

colors = ['c','Red','Blue','m','y','b']
custom_lines = [Line2D([0], [0], color=colors[0], alpha=0.9, lw=8),
                Line2D([0], [0], color=colors[1], alpha=0.9, lw=8),
                Line2D([0], [0], color=colors[2], alpha=0.9, lw=8)]
# fig, ax = plt.subplots(1,1,figsize=(6.5,6.5))
for i,label in enumerate(range(0,np.max(labels)+1)):
	# ax.scatter(data.loc[labels==label].complexity.values,data.loc[labels==label].training_error.values,s=5,marker='o',color=colors[i],alpha=0.1)
	ax[1].scatter(data.loc[labels==label].complexity.values,data.loc[labels==label].test_error.values,s=20,marker='x',color=colors[i],alpha=0.5)
ax[1].set_title('Test error after clustering metric space (k=3)')
ax[1].set_xlabel('Avg. # Nodes')
ax[1].set_ylabel('')
ax[1].legend(custom_lines, ['Cluster_0', 'Cluster_1', 'Cluster_2'])
plt.savefig('metric_space.png',dpi=1200)
	
