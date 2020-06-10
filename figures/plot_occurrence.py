
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams
params = {
			# 'text.latex.preamble': ['\\usepackage{gensymb}'],
			# 'text.usetex': True,
			'font.family': 'Helvetica',
			'lines.solid_capstyle':'butt',
			'lines.markeredgewidth': 1,
			}
rcParams.update(params)

sns.set_context("paper", font_scale=1.6, rc={"lines.linewidth": 2})
sns.set_style('white')
sns.set_palette("cividis")

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
	sens = pd.read_csv('model_sensitivities.csv',header=0,index_col=0) # ,low_memory=False)
	occu = pd.read_csv('sc_occurrence_data.csv',header=0,index_col=0)
	occu.index = [val if val[6] is not '_' else val[:6] + val[7:] for val in occu.index.values] # oops, used two naming conventions
	occu['cluster'] = [sens.loc[((sens['model_id']==i) & (sens['experiment']=='classification')),'cluster'].values[0] for i in occu.index.values]

	clust_list = ['Dominated Cluster','Overfit Cluster','Parsimonious Cluster',]
	occu = occu[[True if i in [1,2,3] else False for i in occu['cluster'].values]]
	occu['Cluster'] = [clust_list[i-1] for i in occu['cluster']]
	occu = occu.drop(['training_error', 'complexity', 'test_error','cluster'],axis=1)
	occu[occu.columns[:-1]] = occu[occu.columns[:-1]] > 0
	occu = occu.groupby(['Cluster']).sum()

	inpu = occu[occu.columns[:-9]].stack()
	inputs = pd.DataFrame()
	inputs['Cluster'] = inpu.index.get_level_values(0)
	inputs['Input'] = inpu.index.get_level_values(1)
	inputs['category'] = [categorize(i) for i in inputs['Input'].values]
	inputs['Category'] = [translate(i) for i in inputs['category'].values]
	inputs['Lag'] = [lag(i) for i in inputs['Input'].values]
	inputs['Neighbor'] = [neighbor(i) for i in inputs['Input'].values]
	inputs['Occurrence'] = inpu.values

	func = occu[occu.columns[-9:]].stack()
	functions = pd.DataFrame()
	functions['Cluster'] = func.index.get_level_values(0)
	functions['Function'] = [func_trans(i) for i in func.index.get_level_values(1)]
	functions['Occurrence'] = func.values

	plots = {'category':'Category',
		'neighborhood':'Neighborhood',
		'lag':'Lag'
		} # three in total
	orders = {'category':['land_tree','land_non','water_pump','water_deliv','econ_tree','econ_non'],
		'neighborhood':['home','neighbor_1','neighbor_2','neighbor_3','neighbor_4','neighbor_5'],
		'lag':['present','lag1','lag2','lag3','lag4','lag5','lag6'],
		'function':['Addition','Subtraction','Multiplication','Division','Negative','Sine','Cosine','Less Than','If-Then-Else'],
		'Category':['Tree Acreage','Non-Tree Acreage','Tree Prices/Values','Non-Tree Prices/Values','Water Deliveries','Water Pumping'],
		'Cluster':['Parsimonious Cluster','Dominated Cluster','Overfit Cluster'],
		# 'color':['midnightblue','Red']
		}

	colors = ['midnightblue','Red','Blue'] #,'c','m','y','b']

	fig, axes = plt.subplots(1,2,figsize=(8,6))


	g2 = sns.boxplot(x='Occurrence', 
					y='Category', 
					order=orders['Category'],
					hue='Cluster', 
					hue_order=orders['Cluster'],
					data=inputs,
					whis='range',
					dodge=True,
					# width=0.8,
					linewidth=2,
					palette=colors, 
					ax=axes[0],
					)

	g1 = sns.scatterplot(x='Occurrence', 
						y='Function', 
						marker='o',
						palette=colors,
						s=100,
						alpha=0.9,
						hue='Cluster',
						hue_order=orders['Cluster'],
						data=functions,
						ax=axes[1]
						)	

	adjust_box_widths(fig, 0.8)

	for i,artist in enumerate(axes[0].artists):
	# Set the linecolor on the artist to the facecolor, and set the facecolor to None
		col = artist.get_facecolor()
		artist.set_edgecolor(col)
		artist.set_facecolor('None')

		# Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
		# Loop over them here, and use the same colour as above
		for j in range(i*6,i*6+6):
			line = axes[0].lines[j]
			line.set_color(col)
			line.set_mfc(col)
			line.set_mec(col)
			line.set_solid_capstyle('butt')

		med_line = axes[0].lines[i*6+4].set_ydata(axes[0].lines[i*6+2].get_ydata())

	axes[0].set_xscale('log')
	axes[0].legend_.remove()

	axes[1].legend_.remove()
	axes[1].legend(frameon=False,markerscale=2,bbox_to_anchor=(1, 1),ncol=4,bbox_transform=plt.gcf().transFigure)
	axes[1].yaxis.set_label_position("right")
	axes[1].yaxis.tick_right()

	axes[0].text(x= 0.95,y=0.8,s='(A)',ha='right',va='top',transform=axes[0].transAxes)
	axes[1].text(x= 0.05,y=0.8,s='(B)',ha='left',va='top',transform=axes[1].transAxes)

	# for patch in axes[0].artists:
	#     r, g, b, a = patch.get_facecolor()
	#     patch.set_facecolor((r, g, b, .9))

	# plt.tight_layout()
	plt.subplots_adjust(wspace=0.05)
	fig.savefig('plot_occurrence.pdf',format='pdf',bbox_inches='tight',dpi=600,transparent=True)

from matplotlib.patches import PathPatch

def adjust_box_widths(g, fac):
	"""
	Adjust the withs of a seaborn-generated boxplot.
	"""
	# iterating through Axes instances
	for ax in g.axes:

		# iterating through axes artists:
		for i,c in enumerate(ax.get_children()):

			# searching for PathPatches
			if isinstance(c, PathPatch):
				# getting current width of box:
				p = c.get_path()
				verts = p.vertices
				# print(verts)
				verts_sub = verts[:-1]
				xmin = np.min(verts_sub[:, 1])
				xmax = np.max(verts_sub[:, 1])
				xmid = 0.5*(xmin+xmax)
				xhalf = 0.5*(xmax - xmin)

				# setting new width of box
				xmin_new = xmid-fac*xhalf
				xmax_new = xmid+fac*xhalf
				verts_sub[verts_sub[:, 1] == xmin, 1] = xmin_new
				verts_sub[verts_sub[:, 1] == xmax, 1] = xmax_new

				# setting new width of median line
				for l in ax.lines:
					if np.all(l.get_xdata() == [xmin, xmax]):
						l.set_xdata([xmin_new, xmax_new])

def categorize(term):
	trees = ['ALMOND','ALMONDHULLS','APRICOT','NECTARINES','PISTACHIO','PLUMS','WALNUT']
	if ('ppu' in term) or ('value' in term):
		if any(tree in term for tree in trees):
			category = 'econ_tree'
		else:
			category = 'econ_non'
	elif 'Pump' in term or 'Deliv' in term:
		if 'Pump' in term:
			category = 'water_pump'
		else:
			category = 'water_deliv'
	elif 'tree' in term:
		category = 'land_tree'
	elif 'non_' in term:
		category = 'land_non'
	else:
		category = 'none'
	return category

def lag(term):
	lags = ['lag1','lag2','lag3','lag4','lag5','lag6']
	if any(lag_ in term for lag_ in lags):
		lag = lags[np.argmax([lag_ in term for lag_ in lags])]
	else:
		lag = 'present'
	return lag

def neighbor(term):
	if 'neighbor' in term:
		neighbor = term[:10]
	else:
		neighbor = 'home'
	return neighbor

def func_trans(term):
	if term == 'lt':
		return 'Less Than'
	elif term == 'ite':
		return 'If-Then-Else'
	elif term == 'vadd':
		return 'Addition'
	elif term == 'vsub':
		return 'Subtraction'
	elif term == 'vmul':
		return 'Multiplication'
	elif term == 'vdiv':
		return 'Division'
	elif term == 'vneg':
		return 'Negative'
	elif term == 'vsin':
		return 'Sine'
	elif term == 'vcos':
		return 'Cosine'

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

if __name__ == "__main__":
	main()
