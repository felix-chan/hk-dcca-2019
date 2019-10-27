"""
Script to convert Catrogram geojson file to hexagon geojson file

Arguments:
 `-o` Output geojson file location
 `-g` Output baseline grid geojson file location
 `-w` Hexagon width parameter (Real width = 2x width parameter)
 `-s` Source file grid standard (wgs / hk80 [default])
 `-t` Output file grid standard (wgs [default]/ hk80)
 `-p` Plot location of grid file

 last_argument: input catorgram geojson file path

e.g. 
```
python catro_to_hex.py -o output.json -g grid.json -w 200 -p output.png input.json
```
"""

import sys
import getopt

import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import math
from pyproj import Proj, transform

# Functions

def hex_shape(x, y, size):
    """
    Generate the hexagon shape with specific size
    """
    x_set = [math.cos(math.pi/6 * (i*2)) * size + x  for i in range(0, 6)]
    y_set = [math.sin(math.pi/6 * (i*2)) * size + y  for i in range(0, 6)]
    return list(map(lambda x,y: (x,y), x_set, y_set))

def gen_hex_points(r, min_x, min_y, max_x, max_y):
    """
    Generate the hexagon grid layout for hexagons
    """
    max_x_n = np.ceil((max_x - min_x) / r / 3) 
    max_y_n = np.ceil((max_y - min_y) / r / np.sqrt(3))
    
    # Generate grid pts
    x_cod = np.arange(0, max_x_n) * r * 3 + min_x
    y_cod = np.arange(0, max_y_n) * r * np.sqrt(3) + min_y
    
    x_cod_2 = x_cod + 1.5 * r
    y_cod_2 = y_cod + np.sqrt(3) / 2 * r
    
    grid1 = np.meshgrid(x_cod, y_cod)
    grid2 = np.meshgrid(x_cod_2, y_cod_2)
    
    grid1 = np.column_stack((np.concatenate(grid1[0]),np.concatenate(grid1[1])))
    grid2 = np.column_stack((np.concatenate(grid2[0]),np.concatenate(grid2[1])))
    
    all_points = []
    
    # Create grid geo
    for x in zip(grid1):
        all_points.append(Point(x[0]))
    for x in zip(grid2):
        all_points.append(Point(x[0]))
        
    return all_points

# Read the input file
default_args = {
    '-o': 'hex_from_catro.geojson', # Output geojson file
    '-g': None, # Output baseline grid geojson file
    '-w': 800, # Hexagon size
    '-s': 'hk80', # Source grid standard
    '-t': 'wgs', # Output grid standard
    '-p': None # Plot location of grid file
}

input_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'g:o:p:s:t:w:')

    for x, y in opts:
        default_args[x] = y

    if len(args) > 0:
        input_file = args[0]
    else:
        print('Please specific the input catrogram geojson file')
        exit(1)
except Exception as e:
    print(e)
    exit(1)

# Confirm the input parameters
print("""
Input catrogram geojson file: {ins}
Output hexagon geojson file: {hex}
Output grid geojson file: {grid}
Hexagon width paragram: {wid}
Input Coordinate systems: {ingis}
Output Coordinate systems: {outgis}
""".format(
    ins = input_file,
    hex = default_args['-o'],
    grid = '--null--' if default_args['-g'] is None else default_args['-g'],
    wid = default_args['-w'],
    ingis = default_args['-s'],
    outgis = default_args['-t']
).strip())

# Read catogram file
catomap = gpd.read_file(input_file)
assert 'CACODE' in catomap.columns
assert 'geometry' in catomap.columns

catomap['district'] = catomap['CACODE'].str.slice(stop=1).astype('category')
hex_size = float(default_args['-w'])

# Find the centroid of each shape and change to hk80
hk80_grid = inProj = Proj(init=('EPSG:2326 +proj=tmerc '
    '+lat_0=22.31213333333334 +lon_0=114.1785555555556 +k=1 +x_0=836694.05 '
    '+y_0=819069.8 +ellps=intl +towgs84=-162.619,-276.959,-161.764,'
    '0.067753,-2.24365,-1.15883,-1.09425 +units=m +no_defs'))
wgs_grid = Proj(init='epsg:4326')

if default_args['-s'] == 'hk80':
    catomap['cent'] = catomap['geometry'].centroid
else:
    catomap['cent'] = catomap['geometry'].centroid.apply(
        lambda geo: transform(wgs_grid, hk80_grid, geo.x, geo,y)
    )

# Generate hexagon base grid
map_grid = gen_hex_points(
    hex_size, 
    catomap.cent.apply(lambda x: x.x).min(), 
    catomap.cent.apply(lambda x: x.y).min(),
    catomap.cent.apply(lambda x: x.x).max() + 10, 
    catomap.cent.apply(lambda x: x.y).max()
)

map_grid_x = np.unique([x.x for x in map_grid])
map_grid_y = np.unique([x.y for x in map_grid])

# Save the grid base if necessary
if default_args['-g']:
    print('Saving the hexagon grid file')
    save_grid_file = default_args['-g']

    map_grid_json = gpd.GeoDataFrame({
        'hk80': gpd.GeoSeries(map_grid)
    })

    if default_args['-t'] == 'hk80':
        map_grid_json['geometry'] = map_grid_json['hk80']
    else:
        map_grid_json['geometry'] = map_grid_json['hk80']\
            .apply(lambda geo: Point(transform(hk80_grid,wgs_grid,geo.x,geo.y)))

    map_grid_json['xid'] = map_grid_json['hk80']\
        .apply(lambda x: np.where(map_grid_x == x.x)[0][0])
    map_grid_json['yid'] = map_grid_json['hk80']\
        .apply(lambda x: np.where(map_grid_y == x.y)[0][0])
    map_grid_json['id'] = map_grid_json\
        .apply(lambda x: '({x}, {y})'.format(x=x.xid, y=x.yid), axis=1)

    # Save to file
    with open(save_grid_file, 'w') as file:
        file.write(map_grid_json.drop(['hk80'], axis=1).to_json())

# Calculate the distance
print('Start matching centroid and hexagon grid')
hex_map = catomap.copy().drop(['geometry'], axis=1)
hex_map['geometry'] = hex_map['cent']

dis_mat = np.full((len(hex_map['district']), len(map_grid)), 99999999)

for i in range(len(hex_map['district'])):
    for j in range(len(map_grid)):
        dist = np.sqrt((hex_map.cent[i].x - map_grid[j].x)**2 + \
            (hex_map.cent[i].y - map_grid[j].y)**2)
        dis_mat[i,j] = dist

# Find the closest point for each hex
dis_mat_cache = dis_mat.copy()

match_set = []
while True:
    min_val = np.min(dis_mat)
    if min_val == 9999999:
        break
    
    loc = np.where(dis_mat == min_val)
    match_set.append((loc[0][0], loc[1][0]))
    
    dis_mat[loc[0][0],:] = 9999999
    dis_mat[:,loc[1][0]] = 9999999

hex_map['nex_hex'] = ''
for i, j in match_set:
    hex_map.at[i, 'nex_hex'] = map_grid[j]

hex_map['hex'] = hex_map['nex_hex']\
    .apply(lambda x: shapely.geometry.Polygon(hex_shape(x.x, x.y, hex_size)))

# Generate final map dataframe
drop_list = ['geometry']
if 'Shape_Area' in hex_map.columns:
    drop_list.append('Shape_Area')
if 'Shape_Leng' in hex_map.columns:
    drop_list.append('Shape_Leng')
final_map = hex_map.copy().drop(['geometry', 'Shape_Area', 'Shape_Leng'], axis=1)\
    .rename({
        'hex': 'geometry'
    }, axis=1)

final_map['xid'] = final_map['nex_hex']\
    .apply(lambda x: np.where(map_grid_x == x.x)[0][0])
final_map['yid'] = final_map['nex_hex']\
    .apply(lambda x: np.where(map_grid_y == x.y)[0][0])

# Save the hexagon plot
if default_args['-p']:
    image_loc = default_args['-p']

    final_map.plot(column='district')
    plt.savefig(image_loc, dpi=300)

# Convert into suitable geo format
if default_args['-t'] == 'wgs':
    output_map = final_map.copy()
    output_map['geometry'] = output_map['geometry']\
        .apply(lambda geo: Polygon([transform(hk80_grid,wgs_grid,x,y) \
            for x,y in geo.exterior.coords]))
    output_map['cent'] = output_map['nex_hex']\
        .apply(lambda geo: transform(hk80_grid,wgs_grid,geo.x,geo.y))
else:
    output_map['cent'] = output_map['nex_hex']

output_map['x'] = output_map['cent'].apply(lambda x: x[0])
output_map['y'] = output_map['cent'].apply(lambda x: x[1])

output_map.drop(['cent', 'nex_hex'], axis=1, inplace=True)

# Write into geojson file
print('Write to geojson file')

with open(default_args['-o'], 'w') as file:
    file.write(output_map.to_json())