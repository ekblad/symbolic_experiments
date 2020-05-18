
import os
from shutil import rmtree
import copy
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import AxesGrid,make_axes_locatable
params = {
			# 'text.latex.preamble': ['\\usepackage{gensymb}'],
			# 'text.usetex': True,
			'font.family': 'Helvetica',
			}
mpl.rcParams.update(params)
import seaborn as sns
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.io.shapereader as shpreader
sns.set(style="whitegrid", palette="pastel", color_codes=True)
from ast import literal_eval as make_tuple

def main():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	local_path = os.path.join(dir_path,'data_maps')
	if 'data_maps' in os.listdir(dir_path):
			rmtree('data_maps') # only turn on if need to do again
	os.mkdir(local_path)

	val = pd.read_csv('sym_data_map.csv',index_col=[0,1],low_memory=False)
	vals = ['neighbor_0_lat_long','ppu_ALFALFA', 'ppu_ALMOND', # 'ppu_ALMONDHULLS',
				# 'ppu_APIARYPRODUCTSBEESWAX', 'ppu_APIARYPRODUCTSHONEY', 'ppu_APRICOT','ppu_COTTONSEED',
				# 'ppu_GRAPE','ppu_GRAPESRAISIN','ppu_GRAPESTABLE','ppu_MILKMANUFACTURING','ppu_MILKMARKETFLUID',
				'ppu_NECTARINES','ppu_PISTACHIO','ppu_PLUMS','ppu_WALNUT','ppu_WHEAT','value_ALFALFA','value_ALMOND',
				# 'value_ALMONDHULLS','value_APIARYPRODUCTSBEESWAX','value_APIARYPRODUCTSHONEY',
				# 'value_APRICOT','value_COTTONSEED','value_GRAPE','value_GRAPESRAISIN','value_GRAPESTABLE',
				# 'value_MILKMANUFACTURING','value_MILKMARKETFLUID',
				'value_NECTARINES', #'value_PASTUREIRRIGATED','value_PASTURERANGE',
				'value_PISTACHIO','value_PLUMS','value_WALNUT','value_WHEAT',
				'home_tree_state_lag0','home_tree_change_lag0','home_tree_change_sign',]
	val = val[vals]
	water = pd.read_csv('water_lags_diff.csv', index_col=[0,1], header=0, low_memory=False)
	water.columns = [make_tuple(k)[0]+'_'+make_tuple(k)[1] for k in water.columns]
	waters = ['NonPondDeliv_state_lag0','NonPondDeliv_change_lag0','NonPondPump_state_lag0',
			'NonPondPump_change_lag0','Pumping_state_lag0','Pumping_change_lag0','RefugeDeliv_state_lag0',
			'RefugeDeliv_change_lag0','RefugePump_state_lag0','RefugePump_change_lag0','RiceDeliv_state_lag0',
			'RiceDeliv_change_lag0','RicePump_state_lag0','RicePump_change_lag0','UrbanDeliv_state_lag0',
			'UrbanDeliv_change_lag0','UrbanPump_state_lag0','UrbanPump_change_lag0',]
	water = water[waters]
	data = val.merge(water,how='inner',on=['comtrs','year'])

	projection = ccrs.PlateCarree()
	axes_class = (GeoAxes,dict(map_projection=projection))
	reader = shpreader.Reader('countyl010g.shp')
	counties = list(reader.geometries())
	gold = '#FFBF00'
	blue = '#022851'
	cmap = 'cividis'

	idx = pd.IndexSlice
	fig = plt.figure()#figsize=(14,10))
	os.chdir(local_path)
	for field in data.columns[1:]:
		for year in data.index.get_level_values(1).unique()[::10]:
			data_map = copy.copy(data[['neighbor_0_lat_long',field]])
			data_map = data_map.loc[idx[:,year],:]
			data_map['X'] = [make_tuple(i)[1] for i in data_map['neighbor_0_lat_long'].tolist()]
			data_map['Y'] = [make_tuple(i)[0] for i in data_map['neighbor_0_lat_long'].tolist()]
			data_map[field] = data_map[field].values/np.max(np.abs(data_map[field].values))

			axgr = AxesGrid(fig, 111, axes_class=axes_class,
							nrows_ncols=(1, 1),
							# axes_pad=0.1,
							# cbar_location='right',
							# cbar_mode='single',
							# cbar_pad=0.1,
							# cbar_size='5%',
							label_mode='')  # note the empty label_mode
			# # To add county lines
			COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

			for i, ax in enumerate(axgr):
				ax.set_extent([-120.8,-118.68, 34.9, 37.2], crs=ccrs.PlateCarree())
				ax.add_feature(cfeature.STATES.with_scale('10m'))
				ax.add_feature(cfeature.OCEAN.with_scale('10m'))
				ax.coastlines(resolution='10m')
				ax.add_feature(COUNTIES, facecolor='none', edgecolor='k')
				ax.scatter(data_map.X,data_map.Y, 
							# alpha=0.5, 
							s=0.95, 
							marker='s',
							label="",
							c=data_map[field].values, 
							cmap=cmap, 
							# vmin=0,vmax=1,
							transform=projection)

			fig.savefig('datamap_{0}_{1}.pdf'.format(field,year),bbox_inches='tight',format='pdf',dpi=50,transparent=True)
			plt.clf()
	os.chdir(dir_path)

if __name__ == '__main__':
	main()