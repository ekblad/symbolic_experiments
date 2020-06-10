
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
from matplotlib.lines import Line2D

sns.set_context("paper") #, font_scale=1.3, rc={"lines.linewidth": 4})
sns.set_style('white')
sns.set_palette("cividis")
dir_path = os.path.dirname(os.path.realpath(__file__))

d = pd.read_csv('sym_data_plot.csv',header=0) # ,low_memory=False)

prices = [
			"ppu_ALFALFA",
			"ppu_ALMOND",
			# "ppu_ALMONDHULLS",
			# "ppu_APIARYPRODUCTSBEESWAX",
			# "ppu_APIARYPRODUCTSHONEY",
			"ppu_APRICOT",
			"ppu_COTTONSEED",
			"ppu_GRAPE",
			# "ppu_GRAPESRAISIN",
			# "ppu_GRAPESTABLE",
			# "ppu_MILKMANUFACTURING",
			# "ppu_MILKMARKETFLUID",
			"ppu_NECTARINES",
			"ppu_PISTACHIO",
			"ppu_PLUMS",
			"ppu_WALNUT",
			"ppu_WHEAT"
			]
values = [
		"value_ALFALFA",
		"value_ALMOND",
		# "value_ALMONDHULLS",
		# "value_APIARYPRODUCTSBEESWAX",
		# "value_APIARYPRODUCTSHONEY",
		"value_APRICOT",
		"value_COTTONSEED",
		"value_GRAPE",
		# "value_GRAPESRAISIN",
		# "value_GRAPESTABLE",
		# "value_MILKMANUFACTURING",
		# "value_MILKMARKETFLUID",
		"value_NECTARINES",
		# "value_PASTUREIRRIGATED",
		# "value_PASTURERANGE",
		"value_PISTACHIO",
		"value_PLUMS",
		"value_WALNUT",
		"value_WHEAT",
		]

water = [
		"home_NonPondDeliv_state_lag1",
		"home_NonPondPump_state_lag1",
		]

land = [
		"home_tree_state_lag0",
		"home_non_state_lag0",
		]


prices_trees = [
			'ppu_ALMOND',
			# 'ppu_ALMONDHULLS',
			'ppu_APRICOT',
			'ppu_NECTARINES',
			'ppu_PISTACHIO',
			'ppu_PLUMS',
			'ppu_WALNUT',
				]

values_trees = [
			'value_ALMOND',
			# 'value_ALMONDHULLS',
			'value_APRICOT',
			'value_NECTARINES',
			'value_PISTACHIO',
			'value_PLUMS',
			'value_WALNUT',
				]

pal = ['midnightblue','goldenrod','slategray']
print(d)

def translate(item):
	if item == land[0]:
		return 'Tree Crops'
	elif item == land[1]:
		return 'Non-Tree Crops'
	elif 'NonPondDeliv' in item:
		return 'Ag. Deliveries'
	elif 'NonPondPump' in item:
		return 'Ag. Pumping'
	else:
		return 'Total'

def translate_econ(item):
	if ((item in prices) and (item in prices_trees)):
		return 'Tree Crops'
	elif ((item in prices) and (item not in prices_trees)):
		return 'Non-Tree Crops'
	elif ((item in values_trees) and (item in values)):
		return 'Tree Crops'
	else:
		return 'Non-Tree Crops'

# format land use data
d_land = d[land]
d_land['Year'] = d.year
d_land['Total'] = d_land[land[0]] + d_land[land[1]]
d_land = d_land.groupby(['Year',])[land+['Total']].sum()
d_land = d_land.stack()

d_land_plot = pd.DataFrame()
d_land_plot['Year'] = d_land.index.get_level_values(0)
d_land_plot[''] = [translate(i) for i in d_land.index.get_level_values(1)]
d_land_plot['Square Miles'] = d_land.values*0.0015625 # to square miles

# format price data
d_prices = d[prices]
d_prices['Year'] = d.year
d_prices= d_prices.melt(id_vars = ['Year'],var_name='Field', value_name='Price Per Unit ($)')
d_prices[''] = [translate_econ(i) for i in d_prices.Field]
print(d_prices)

# format value data
d_value = d[values]
d_value['Year'] = d.year
d_value= d_value.melt(id_vars = ['Year'],var_name='Field', value_name='Value ($M)')
d_value['Value ($M)'] = d_value['Value ($M)'].values/1000000 # to million dollars
d_value[''] = [translate_econ(i) for i in d_value.Field]
print(d_value)

d_water = d[water]
d_water['Year'] = d.year
d_water = d_water.melt(id_vars = ['Year'],var_name='Field', value_name='TAF')
d_water['TAF'] = d_water['TAF'].values * 2.29569e-5/1000 # to million acre-feet
d_water[''] = [translate(i) for i in d_water.Field.values]
d_water = d_water.groupby(['Year',''])['TAF'].sum()

d_water_plot = pd.DataFrame()
d_water_plot['Year'] = d_water.index.get_level_values(0)
d_water_plot[''] = d_water.index.get_level_values(1)
d_water_plot['TAF'] = d_water.values

fig, axs = plt.subplots(2,2,figsize=(8,6),sharex ='col')

# plot crops
sns.lineplot(x="Year", y='Square Miles', hue='', 
			hue_order=["Non-Tree Crops", "Tree Crops", "Total"],
			palette=pal,data=d_land_plot,ax=axs[0,0])

palettes = ["GnBu","YlOrBr",]
# plot prices
for j,field in enumerate(["Non-Tree Crops", "Tree Crops"]):
	plot_ = d_prices[d_prices[''] == field]
	plot_[''] = [i[4:] if i != 'ppu_COTTONSEED' else i[4:10] for i in plot_['Field'].values]
	sns.lineplot(x='Year',y='Price Per Unit ($)',
		legend='brief',hue='',palette=palettes[j],data=plot_,ax=axs[1,0])
# plot values
for j,field in enumerate(["Non-Tree Crops", "Tree Crops"]):
	plot_ = d_value[d_value[''] == field]
	plot_[''] = [i[6:] for i in plot_['Field'].values]
	sns.lineplot(x='Year',y='Value ($M)',
		legend='brief',hue='',palette=palettes[j],data=plot_,ax=axs[1,1])

# plot water
sns.lineplot(x='Year',y='TAF',hue = '',
			hue_order=["Ag. Deliveries", "Ag. Pumping"],
			palette=pal[:2],data=d_water_plot,ax=axs[0,1])

list_text = ['(A)','(B)','(D)','(C)']
fig.tight_layout(pad=1.0)
for i,ax in enumerate(axs.flatten()):
	handles, labels = ax.get_legend_handles_labels()
	ax.get_legend().remove()
	if i == 0:
		ax.legend(handles=handles,labels=labels,frameon=False,loc='lower right')
	if i == 1:
		ax.legend(handles=handles,labels=labels,frameon=False,loc='lower left')
	if i == 2:
		ax.legend(handles=handles[0:4],labels=labels[0:4],ncol=4,fontsize='x-small',
			bbox_to_anchor=(0.5, 0.95),frameon=False,loc='center')
	if i == 3:
		ax.legend(handles=handles[4:],labels=labels[4:],ncol=3,fontsize='x-small',
			bbox_to_anchor=(0.5, 0.925),frameon=False,loc='center')
	if i in [2,3]:
		ax.set_yscale('log')
		if i == 2:
			ax.set_ylim(bottom=100)
		else:
			ax.set_ylim(bottom=1)
	else:
		ax.set_ylim(bottom=0)

	ax.margins(x=0,y=0.05) 
	ax.text(x= 0.95,y=0.05,s=list_text[i],ha='center',va='center',transform=ax.transAxes)
	# axes[1].text(x= 0.05,y=0.8,s='(B)',ha='left',va='top',transform=axes[1].transAxes)

plt.savefig('time_series.pdf',format='pdf',bbox_inches='tight',dpi=600,transparent=True)

exit()