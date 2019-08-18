"""
Convert the ESRI format JSON file from
 https://www.map.gov.hk/ to geoJSON file format.
"""

#%%
import json
import os
from pyproj import Proj, transform
from geojson import Feature, FeatureCollection, Polygon

#%%
# Please download the ESRI json file 
#  first from HK GeoInfo Map
#  https://www.map.gov.hk/gm/map/s/dce2019/district/A
# Save all files in ../raw_json/

raw_files = '../raw_json/'


#%%
# List the files downloaded
all_files = os.listdir(raw_files)
print("Total file exists: {total}".format(
    total = len(all_files)
))

#%%
# Initial GeoLocation convertor
inProj = Proj(init=('EPSG:2326 +proj=tmerc +lat_0=22.31213333333334 '
                    '+lon_0=114.1785555555556 +k=1 +x_0=836694.05 '
                    '+y_0=819069.8 +ellps=intl +towgs84=-162.619,'
                    '-276.959,-161.764,0.067753,-2.24365,-1.15883,'
                    '-1.09425 +units=m +no_defs'))
outProj = Proj(init='epsg:4326')

#%%
# Convert files one by one
file_output_shape = {}

for file_name in all_files:
    with open('{dir}{name}'.format(
        name=file_name,
        dir=raw_files
        ), 'r') as file:
        loc_json = file.read()
        loc_shp = json.loads(loc_json)
        
        for idx in range(0, len(loc_shp['features'])):
            district_code = loc_shp['features'][idx]['attributes']['CACODE']
            print("Converting {d}".format(d=district_code))
            Lxy = [transform(inProj,outProj,x1,y1) for x1,y1 in 
                loc_shp['features'][idx]['geometry']['rings'][0]]
            
            # Convert to lat long for checking
            Lxy = [(y1, x1) for x1, y1 in Lxy]
            
            file_output_shape[district_code] = Lxy

#%%
# Set the district code mapping
district_code = {
    'A': 'Central and Western District',
    'B': 'Wan Chai District',
    'C': 'Eastern District',
    'D': 'Southern District',
    'E': 'Yau Tsim Mong District',
    'F': 'Sham Shui Po District',
    'G': 'Kowloon City District',
    'H': 'Wong Tai Sin District',
    'J': 'Kwun Tong District',
    'K': 'Tsuen Wan District',
    'L': 'Tuen Mun District',
    'M': 'Yuen Long District',
    'N': 'North District',
    'P': 'Tai Po District',
    'Q': 'Sai Kung District',
    'R': 'Sha Tin District',
    'S': 'Kwai Tsing District',
    'T': 'Islands District'
}

#%%
# Create geoJSON object
features = []
for name in file_output_shape:
    # geoJSON use long lat
    temp_polygon = Polygon([[(y,x) for x,y in file_output_shape[name]]])
    district_name = district_code[name[0]]
    features.append(
        Feature(
            geometry = temp_polygon,
            properties = {
                'Region': name,
                'District': district_name
            }
        )
    )

#%% Save the geoJSON file
collection = FeatureCollection(features)

with open("../map_files/HKDCCA2019.json", "w") as f:
    f.write('%s' % collection)

print("Finish generate the geoJSON file")