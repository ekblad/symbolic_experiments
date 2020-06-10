import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler, colors
params = {
			# 'text.latex.preamble': ['\\usepackage{gensymb}'],
			# 'text.usetex': True,
			'font.family': 'Helvetica',
			}
rcParams.update(params) 
import seaborn as sns
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 0.5})
sns.set_style('white')
sns.set_palette("cividis")
from matplotlib.lines import Line2D

def main():
	legend_elements = [
						Line2D([0], [0], marker='o', color='Blue', label='Training Error',
						markerfacecolor='none', markersize=15),
						Line2D([0], [0], marker='x', color='Red', label='Test Error',
						markerfacecolor='none', markersize=15),
						]


	dir_path = os.path.dirname(os.path.realpath(__file__))
	local_path = os.path.join(dir_path,'model_sensitivities.csv')

	data = pd.read_csv(local_path,header=0,index_col=0,low_memory=False)
	data = data[data.experiment=='classification'][data.cluster==3]
	for i in data.columns:
		print(i)

	# exit()
	fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(8,8))
	# ax = axes.flatten()
	model_list = ['seed_2_model_54','seed_3_model_51','seed_6_model_33']
	axes.scatter(x=data.complexity,y=data.training_error,marker='o',facecolors='none',edgecolors='midnightblue',alpha=0.3) #label='Cluster '+ str(cluster),s=marker_size,)
	axes.scatter(x=data.complexity,y=data.test_error,marker='x',facecolors='midnightblue',edgecolors='midnightblue',alpha=0.3)
	for model in model_list:
		# print(data[data.model_id==model].labels.unique())
		axes.scatter(x=data[data.model_id==model].complexity,y=data[data.model_id==model].training_error,edgecolors='Blue',marker='o',facecolors='none',alpha=0.9) 
		axes.scatter(x=data[data.model_id==model].complexity,y=data[data.model_id==model].test_error,edgecolors='Red',marker='x',facecolors='Red',alpha=0.9)
	# axes.set_facecolor('none')
	axes.set_xlabel('Complexity')
	axes.set_ylabel('Misclassification Rate')
	axes.set_title('Parsimonious Cluster - Classification')
	axes.legend(frameon=False,handles=legend_elements)
	# plt.show()
	# exit()
	plt.savefig('baseline_metrics.pdf',format='pdf',bbox_inches='tight',dpi=600,transparent=True)

if __name__ == '__main__':
	main()
