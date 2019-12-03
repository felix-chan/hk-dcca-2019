"""
Tools for converting the geojson format file to hexjson format
More about Hexagon coordinate
https://www.redblobgames.com/grids/hexagons/#coordinates
"""

import numpy as np

def convert_to_doubled(xid, yid):
    """
    Convert to Doubled Coordinates 
    https://www.redblobgames.com/grids/hexagons/#coordinates-doubled
    """
    u = list(xid)
    max_y = np.max(yid)
    v = list(max_y - np.array(yid))
    return (u, v)

def convert_to_axial(xid, yid):
    """
    Convert to Axial Coordinates
    https://www.redblobgames.com/grids/hexagons/#coordinates-axial
    """
    q = list(xid)
    max_y = np.max(yid)
    v = list(max_y - np.array(yid))
    r = [int((v1 - q1) / 2) for v1, q1 in zip(v, q)]
    return (q, r)

def double_to_cube(row, col):
    """
    Function to convert double to cube coordinate
    https://www.redblobgames.com/grids/hexagons/#conversions-doubled
    """
    v_col = col if (row + col % 2) == 0 else col + 1
    x = v_col
    z = int((row - v_col) / 2)
    y = -x - z
    return (x, y, z)

def cube_to_evenq(x, y, z):
    """
    Function to convert cube to even-q offset coordinate
    https://www.redblobgames.com/grids/hexagons/#conversions-offset
    """
    col = x
    row = int(z + (x + x&1) / 2)
    return (col, row)
