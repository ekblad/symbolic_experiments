
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

gold = '#FFBF00'
blue = '#022851'
cmap = 'cividis'

locations = pd.read_csv('legal_to_latlong.csv')
locations = locations.set_index('CO_MTRS')

vals = pd.read_csv('complete_records.csv')
vals = vals[['year','comtrs','tree_crops','total']]
vals = vals.set_index('comtrs')
vals['percent_trees'] = vals.tree_crops/vals.total
vals = vals[['year','percent_trees']]
vals = vals.pivot(columns='year',values='percent_trees')
vals = vals.reindex(locations.index)

def main():
	proj = 'ccrs.PlateCarree()'
	projection = ccrs.PlateCarree()
	axes_class = (GeoAxes,
				  dict(map_projection=projection))
	subset = [1974.0,2016.0]
	fig = plt.figure(figsize=(14,10))
	axgr = AxesGrid(fig, 111, axes_class=axes_class,
					nrows_ncols=(1, 1),
					axes_pad=0.1,
					# cbar_location='right',
					# cbar_mode='single',
					# cbar_pad=0.1,
					# cbar_size='5%',
					label_mode='')  # note the empty label_mode
	# # To add county lines
	reader = shpreader.Reader('countyl010g.shp')
	counties = list(reader.geometries())
	COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

	for i, ax in enumerate(axgr):
		ax.set_extent([-124.6,-114, 32.4, 42.2], crs=projection)
		ax.add_feature(cfeature.STATES.with_scale('10m'))
		ax.add_feature(cfeature.OCEAN.with_scale('10m'))
		ax.coastlines(resolution='10m')
		ax.scatter(locations.X, locations.Y, 
					# alpha=0.5, 
					s=2.0, 
					marker='s',
					label="",
					c=vals[subset[i+1]], 
					cmap=plt.get_cmap(cmap), 
					vmin=0,vmax=1,
					transform=ccrs.PlateCarree())
		ax.scatter(-119.0187073,35.4,
					s=100, 
					marker='v',
					label="Bakersfield",
					c=blue, 
					transform=ccrs.PlateCarree())
		ax.text(-118.8, 35.37, 'Bakersfield', c=blue,transform=ccrs.PlateCarree(),fontsize=16,fontweight='bold')
		ax.scatter(-119.772591,36.8,
					s=100, 
					marker='v',
					label="Fresno",
					c=blue, 
					transform=ccrs.PlateCarree())
		ax.text(-119.5, 36.8, 'Fresno', c=blue,transform=ccrs.PlateCarree(),fontsize=16,fontweight='bold')
		ax.scatter(-121.740517,38.6,
					s=100, 
					marker='v',
					label="Davis",
					c=blue, 
					transform=ccrs.PlateCarree())
		ax.text(-121.5, 38.5, 'Davis', c=blue,transform=ccrs.PlateCarree(),fontsize=16,fontweight='bold')
		ax.scatter(-118.243683,34.052235,
					s=100, 
					marker='v',
					label="Los Angeles",
					c=blue, 
					transform=ccrs.PlateCarree())
		ax.text(-118.1, 33.9, 'Los Angeles', c=blue,transform=ccrs.PlateCarree(),fontsize=16,fontweight='bold')

	fig.savefig('mapping_inset.pdf',format='pdf',dpi=300,transparent=True)

if __name__ == '__main__':
	main()
