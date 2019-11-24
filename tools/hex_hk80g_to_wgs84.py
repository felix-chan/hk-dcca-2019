from functools import partial
import geopandas as gpd
import pyproj
from shapely.geometry import Polygon
from shapely.ops import transform


# Read data
hexgeojson = 'working/hex_grid/map_grid.geojson'
hex_data = gpd.read_file(hexgeojson)


# Multiplication
multiply_1000 = partial(
    lambda x, y: (x * 1000 + 0, y * 1000 + 0)
)

# Projection
in_proj = (
    'EPSG:2326 +proj=tmerc +lat_0=22.31213333333334 '
    '+lon_0=114.1785555555556 +k=1 +x_0=836694.05 '
    '+y_0=819069.8 +ellps=intl +towgs84=-162.619,'
    '-276.959,-161.764,0.067753,-2.24365,-1.15883,'
    '-1.09425 +units=m +no_defs'
)
out_proj = 'epsg:4326'

in_proj = pyproj.Proj(init=in_proj)
out_proj = pyproj.Proj(init=out_proj)

proj_to_wgs84 = partial(
    pyproj.transform,
    in_proj, 
    out_proj
)


# Transformation
hex_data['geometry'] = hex_data['geometry'].apply(
    lambda x: transform(multiply_1000, x)
)
hex_data['geometry'] = hex_data['geometry'].apply(
    lambda x: transform(proj_to_wgs84, x)
)

# Save
with open('working/hex_grid/wgs84_' + hexgeojson, 'w') as f:
    f.write(hex_data.to_json())

