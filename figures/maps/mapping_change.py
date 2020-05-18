
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
					nrows_ncols=(1, 2),
					axes_pad=0.1,
					cbar_location='bottom',
					cbar_mode='single',
					cbar_pad=0.0,
					cbar_size='1%',
					label_mode='')  # note the empty label_mode
	# # To add county lines
	reader = shpreader.Reader('countyl010g.shp')
	counties = list(reader.geometries())
	COUNTIES = cfeature.ShapelyFeature(counties, projection)

	for i, ax in enumerate(axgr):
		ax.set_extent([-120.8,-118.7, 34.3, 37.3], crs=projection)
		ax.add_feature(COUNTIES, facecolor='none', edgecolor='gainsboro')
		ax.add_feature(cfeature.STATES.with_scale('10m'))
		ax.add_feature(cfeature.OCEAN.with_scale('10m'))
		ax.coastlines(resolution='10m')
		ax.scatter(locations.X, locations.Y, 
					# alpha=0.9, 
					s=3, 
					marker='s',
					label="",
					c=vals[subset[i]], 
					cmap=cmap,
					# cmap=plt.get_cmap(cmap), 
					vmin=0,vmax=1,
					transform=projection)
		ax.scatter(-119.0187073,35.373291,
					s=100, 
					marker='v',
					label="Bakersfield",
					c=blue, 
					transform=projection)
		ax.text(-119.2, 35.43, 'Bakersfield', c=blue,transform=projection,fontsize=16,fontweight='bold')
		ax.scatter(-119.772591,36.746841,
					s=100, 
					marker='v',
					label="Fresno",
					c=blue, 
					transform=projection)
		ax.text(-120, 36.83, 'Fresno', c=blue,transform=projection,fontsize=16,fontweight='bold')
		ax.scatter(-120.6596156,35.2827524,
					s=100, 
					marker='v',
					label="SLO",
					c=blue, 
					transform=projection)
		ax.text(-120.64, 35.2, 'San Luis Obispo', c=blue,transform=projection,fontsize=16,fontweight='bold')
		ax.scatter(-119.698189,34.45,
					s=100, 
					marker='v',
					label="SB",
					c=blue, 
					transform=projection)
		ax.text(-119.72, 34.52, 'Santa Barbara', c=blue,transform=projection,fontsize=16,fontweight='bold')
		ax.text(.1,.95,str(subset[i])[:4],
			horizontalalignment='center',
			fontsize=30,
			fontweight='bold',
			c=blue,
			transform=ax.transAxes)

	for cax in axgr.cbar_axes:
		cax.remove()

	img = plt.imshow(np.array([[0,1]]),cmap=cmap)
	img.set_visible(False)
	cbar = fig.colorbar(img,
				orientation = 'horizontal',
				pad = 0.04,
				aspect = 60,
				shrink = 0.9725,
				ticks = None,
				drawedges = False,
				anchor = (0.5,-.02))
	cbar.ax.set_xticklabels(['0',None, None, None, None,'100'])
	cbar.ax.tick_params(size=0)
	cbar.ax.tick_params(labelsize=20)
	cbar.ax.set_xlabel('Percentage of Tree Crops',fontsize=20,labelpad=-18)

	# cbar.outline.set_visible(False)
	cbar.outline.set_edgecolor('black')
	cbar.outline.set_linewidth(1)

	fig.savefig('map_final.pdf',format='pdf',dpi=100,transparent=True)

if __name__ == '__main__':
	main()
