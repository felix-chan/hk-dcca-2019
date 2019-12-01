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