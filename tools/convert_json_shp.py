"""
Convert geoJSON file to ESRI Shapefile
"""

#%%
import geopandas

#%%
# Read by geoPandas
hkdc = geopandas.read_file('../map_files/HKDCCA2019.json')

#%%
# Save into Shapefile
hkdc.to_file(driver = 'ESRI Shapefile', filename= "../map_files/shp/HKDCCA2019.shp")