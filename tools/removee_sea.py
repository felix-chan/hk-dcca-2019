"""
Remove the sea area and find the largest area
 in each district
"""

#%%
import geopandas as gpd
import math

import shapely
from shapely.ops import cascaded_union

import folium

#%%
# Download the HK shapefile first
#  http://opendata.esrichina.hk/datasets/eea8ff2f12b145f7b33c4eef4f045513_0
# Save all unzipped files in ../raw_json/

raw_files = '../raw_json/'

#%%
# Read the shapefile of whole Hong Kong
hkmap = gpd.read_file("{dir}Hong_Kong_18_Districts.shp".format(
    dir=raw_files
))

#%%
# Read the geojson files created before
dc2019 = gpd.read_file("../map_files/HKDCCA2019.json")

#%%
# Combine all district in HK map into single polygon
combine_hk = gpd.GeoSeries(cascaded_union(hkmap['geometry']))
combine_hk.crs = {'init' :'epsg:4326'}
combine_hk_df = gpd.GeoDataFrame({
    'geometry': combine_hk
})

#%%
# Remove the sea are in each DC district
dc2019_no_sea = gpd.overlay(combine_hk_df, dc2019, how='intersection')

#%%
# Save the geojson file
with open('../map_files/HKDCCA2019_no_sea.json', 'w') as file:
    file.write(dc2019_no_sea.to_json())

#%% 
# Select the largest area for each district
# It is created because multiple islands may appeared in 
#  the same DC district

dc2019_no_sea_max = dc2019_no_sea.copy()
dc2019_no_sea_max['geometry'] = dc2019_no_sea_max['geometry'].\
    apply(lambda x: max(x, key=lambda a: a.area) 
        if type(x) == shapely.geometry.multipolygon.MultiPolygon else x)

#%%
# Save the geojson file
with open('../map_files/HKDCCA2019_no_sea_max.json', 'w') as file:
    file.write(dc2019_no_sea_max.to_json())

# Visualize the district
# Create map object
map_1 = folium.Map(location=[22.375985, 114.114600], zoom_start=13)

folium.GeoJson(
    "../map_files/HKDCCA2019_no_sea_max.json",
    tooltip = folium.features.GeoJsonTooltip(['Region', 'District'])
).add_to(map_1)

# Save the map into files
map_1.save('../docs/hkdc2019_no_sea_max.html')