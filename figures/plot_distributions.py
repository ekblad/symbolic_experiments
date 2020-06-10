import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler, colors
params = {
			# 'text.latex.preamble': ['\\usepackage{gensymb}'],
			# 'text.usetex': True,
			'font.family': 'Helvetica',
			}
rcParams.update(params) 
import seaborn as sns
sns.set_context("paper", font_scale=1.6, rc={"lines.linewidth": 0.5})
sns.set_style('white')
# sns.set_palette("cividis")
from matplotlib.lines import Line2D

def main():
	legend_elements = [
						Line2D([0], [0], marker='o', color='Blue', label='Training Error',
						markerfacecolor='none', markersize=15),
						Line2D([0], [0], marker='x', color='Red', label='Test Error',
						markerfacecolor='none', markersize=15),
						]


	dir_path = os.path.dirname(os.path.realpath(__file__))
	local_path = os.path.join(dir_path,'sym_data_plot.csv')

	data = pd.read_csv(local_path,header=0,index_col=0,low_memory=False)
	for i in data.columns:
		print(i)
	features = ['home_tree_change_lag1',
				'home_RefugeDeliv_state_lag1',
				'home_RefugePump_state_lag1',
				# 'home_RiceDeliv_state_lag1',
				'home_RicePump_state_lag1',
				'home_UrbanDeliv_state_lag1',
				# 'home_UrbanPump_state_lag1',
				]
	labels = ['Tree Acreage Changes',
				'Refuge Deliveries',
				'Refuge Pumping',
				# 'Rice Deliveries',
				'Rice Pumping',
				'Urban Deliveries',
				# 'Urban Pumping',
				]
	order = ['Tree Acreage Changes',
				'Refuge Deliveries',
				'Refuge Pumping',
				# 'Rice Deliveries',
				'Rice Pumping',
				'Urban Deliveries',
				# 'Urban Pumping',
				
				]			
	data = data[features]
	data.columns = labels
	data = data[order]
	data = (data-data.mean())/data.std()
	print(np.sum(data['Tree Acreage Changes'].values**2)/len(data))
	exit()

	fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(5.8,5.8))
	data.plot.hist(bins=50, 
		# alpha=0.6,
		ax=axes,stacked=True)
	# for feat in features:
	# 	sns.kdeplot(data[feat].values, label=feat,clip_on=True,lw=2, bw=.2,cut=0)
	# axes.set_facecolor('none')
	axes.set_xlabel('Standardized Value')
	axes.set_ylabel('Count')
	axes.set_yscale('log')
	axes.set_title('Model Feature Histograms')
	axes.legend(frameon=False)
	# plt.show()
	# exit()
	plt.savefig('distributions_callout.pdf',format='pdf',bbox_inches='tight',dpi=600,transparent=True)

if __name__ == '__main__':
	main()
