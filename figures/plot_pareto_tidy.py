
# this script generates kernel density estimation plots of sensitivity indices
# for results according to cluster (hue), and category, neighborhood, and lag of input

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler, colors
params = {
			# 'text.latex.preamble': ['\\usepackage{gensymb}'],
			# 'text.usetex': True,
			'font.family': 'Helvetica',
			}
rcParams.update(params)
from matplotlib.lines import Line2D

sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 0.5})
sns.set_style('white')
sns.set_palette("cividis")

marker_size=6
dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv('model_sensitivities.csv',header=0,index_col=0) # ,low_memory=False)

data_cols = ['experiment', 'model_id', 'inputs', 'training_error', 'complexity',
			'test_error', 'cluster', 'category', 'lag', 'neighborhood', 'S1',
			'S1_conf', 'ST', 'ST_conf']
subset = ['experiment', 'model_id', 'training_error', 'complexity',
			'test_error', 'cluster',]

orders = {'cluster':[0,1,2,3,4]
			} 
data = data[subset]

fig, axs = plt.subplots(2,1,figsize=(5,7)) 

data_reg = data[data['experiment'] == 'regression']
data_reg['Complexity'] = data_reg['complexity']
data_reg['Cluster'] = data_reg['cluster']
data_reg['Training Error'] = data_reg['training_error']
data_reg['Test Error'] = data_reg['test_error']

axs[0].hist2d(data_reg['Complexity'].values,
				data_reg['Training Error'].values,
				norm=colors.LogNorm(),
				bins=20,alpha=0.5,cmap='Blues')
alpha1 = 0.9
alpha2 = 0.9
alpha3 = 0.9
alphas=[alpha1,alpha1,alpha2,alpha1,alpha1]
colors_=['slategray','slategray','slategray','Blue','slategray',]
for cluster in orders['cluster']:
	x = data_reg[data_reg['Cluster'] == cluster]['Complexity'].values
	y = data_reg[data_reg['Cluster'] == cluster]['Test Error'].values
	axs[0].scatter(x=x,y=y,c=colors_[cluster],marker='x',alpha=alphas[cluster],label='Cluster '+ str(cluster),s=marker_size,)

data_class = data[data['experiment'] == 'classification']
data_class['Complexity'] = data_class['complexity']
data_class['Cluster'] = data_class['cluster']
data_class['Training Error'] = data_class['training_error']
data_class['Test Error'] = data_class['test_error']

axs[1].hist2d(data_class['Complexity'].values,
				data_class['Training Error'].values,
				norm=colors.LogNorm(),
				bins=20,alpha=0.5,cmap='Blues')

alphas=[alpha1,alpha2,alpha1,alpha2,alpha3]
colors_=['Blue','Red','midnightblue','slategray','slategray']
for cluster in orders['cluster']:
	x = data_class[data_class['Cluster'] == cluster]['Complexity'].values
	y = data_class[data_class['Cluster'] == cluster]['Test Error'].values
	axs[1].scatter(x=x,y=y,c=colors_[cluster],marker='x',alpha=alphas[cluster],label='Cluster '+ str(cluster),s=marker_size,)

axs[0].text(x= 0.95,y=0.95,s='(A)',ha='right',va='top',transform=axs[0].transAxes)
axs[1].text(x= 0.95,y=0.95,s='(B)',ha='right',va='top',transform=axs[1].transAxes)
# axs[1].text('(B)',ha='right',va='top')

axs[0].legend(frameon=False,loc=7)
axs[1].legend(frameon=False,loc=7)

axs[0].set_ylabel('Mean Square Error')
axs[1].set_ylabel('Misclassification Rate')
axs[1].set_xlabel('Complexity')

fig.tight_layout(pad=1.0)
for i,ax in enumerate(axs.flatten()):
	ax.margins(x=0.1,y=0.1)
	if i == 0:
		ax.set_ylim(bottom=0.8,top=1.05)
	else:
		ax.set_ylim(bottom=0.3,top=0.8)
	ax.set_xlim(left=0,right=0.85)
plt.savefig('metric_space_tidy.pdf',format='pdf',bbox_inches='tight',dpi=600,transparent=True)

exit()