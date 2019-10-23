"""
Hexagon map object with can create json string for storage or 
 display
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import math
import shapely
from pyproj import Proj, transform

class hexMap:
    """
    Hexagon map object which can read or generate json file
     for hexagon map

    Parameters
    ----------
    file_loc: string
        File location of hexagon map geojson
    hex_grid_loc: string
        File location of hexagon based grid
    hex_width: float
        The 1/2 width of hexagon
    """

    hex_df = None
    grid_df = None
    hex_width = 100

    def __init__(self, file_loc, hex_grid_loc, hex_width = 100):
        hex_dataframe = gpd.read_file(file_loc)
        self.hex_df = hex_dataframe
        self.grid_df = gpd.read_file(hex_grid_loc)
        self.hex_width = hex_width


    def gen_json(self):
        """
        Generate json file which follow geojson format

        Parameters
        ----------
        None

        Returns
        -------
        geojson string: str object
        """

        json_text = self.hex_df.to_json()
        return json_text

    def _draw_hex(self, x, y, width, coord='wgs'):
        """
        Draw a hexagon with specific width

        Parameters
        ----------
        x: float
            X coordinate of centroid of hexagon
        y: float
            Y coordinate of centroid of hexagon
        width: float
            The 1/2 of width of hexagon
        coord: {'wgs', 'hk80'} string
            Type of input coordinate

        Returns
        -------
        Hexagon Polygon: shapely.geometry.Polygon
        """

        if coord == 'wgs':
            hk80_grid = Proj(init=('EPSG:2326 +proj=tmerc +lat_0=22.31213333333334 '
                                   '+lon_0=114.1785555555556 +k=1 +x_0=836694.05 '
                                   '+y_0=819069.8 +ellps=intl +towgs84=-162.619,'
                                   '-276.959,-161.764,0.067753,-2.24365,-1.15883,'
                                   '-1.09425 +units=m +no_defs'))
            wgs_grid = Proj(init='epsg:4326')

            hk80_x, hk80_y = transform(wgs_grid,hk80_grid,x,y)
        else:
            hk80_x, hk80_y = x, y

        x_set = [math.cos(math.pi/6 * (i*2)) * width + hk80_x  
                 for i in range(0, 6)]
        y_set = [math.sin(math.pi/6 * (i*2)) * width + hk80_y 
                 for i in range(0, 6)]

        if coord == 'wgs':
            hex_list = list(map(lambda x1,y1: transform(hk80_grid, wgs_grid, x1, y1), 
                                x_set, y_set))
        else:
            hex_list = list(map(lambda x1,y1: (x1,y1), x_set, y_set))
        
        return shapely.geometry.Polygon(hex_list)

    def swap_position(self, old_x, old_y, new_x, new_y):
        """
        Swap the location of two hexagon
        
        Parameters
        ----------
        old_x: string
            X coordinate of old hexagon
        old_y: string
            Y coordinate of old hexagon
        new_x: string
            X coordinate of target hexagon
        new_y: string
            Y coordinate of target hexagon

        Returns
        -------
        Success status of operation: boolean
        """

        # Force to case input into int
        old_x = int(old_x)
        old_y = int(old_y)
        new_x = int(new_x)
        new_y = int(new_y)

        # Check if old hexagon exists
        old_hex = np.sum((self.hex_df.xid == old_x) & 
                         (self.hex_df.yid == old_y))
        old_grid = np.sum((self.grid_df.xid == old_x) & 
                          (self.grid_df.yid == old_y))

        if (old_hex > 0) & (old_grid > 0):
            # original hex exists
            new_hex = np.sum((self.hex_df.xid == new_x) & 
                             (self.hex_df.yid == new_y))
            new_grid = np.sum((self.grid_df.xid == new_x) & 
                              (self.grid_df.yid == new_y))

            if new_grid > 0:
                # The target grid exists
                df_old_id = self.hex_df.index[(self.hex_df.xid == old_x) & 
                                              (self.hex_df.yid == old_y)][0]
                if new_hex > 0:
                    # Swap the location

                    swap_cols = ['id', 'CACODE', 'ENAME', 'CNAME', 
                                 'seat', 'district']

                    for col in swap_cols:
                        df_new_id = self.hex_df.index[(self.hex_df.xid == new_x) & 
                                                      (self.hex_df.yid == new_y)][0]
                        old_val = self.hex_df[col].iloc[df_old_id]
                        new_temp_col = self.hex_df[col].copy()
                        new_temp_col.iloc[df_old_id] = self.hex_df[col]\
                            .iloc[df_new_id]
                        new_temp_col.iloc[df_new_id] = old_val

                        self.hex_df[col] = new_temp_col
                    
                else:
                    # Change to new hex
                    df_new_id = self.grid_df.index[(self.grid_df.xid == new_x) & 
                                                   (self.grid_df.yid == new_y)]
                    new_wgs_x = self.grid_df.geometry.iloc[df_new_id].x.values[0]
                    new_wgs_y = self.grid_df.geometry.iloc[df_new_id].y.values[0]

                    new_wgs_polygon = self._draw_hex(new_wgs_x, new_wgs_y,
                                                     self.hex_width, 'wgs')

                    # Replace into table
                    self.hex_df.at[df_old_id, 'xid'] = new_x
                    self.hex_df.at[df_old_id, 'yid'] = new_y
                    self.hex_df.at[df_old_id, 'x'] = new_wgs_x
                    self.hex_df.at[df_old_id, 'y'] = new_wgs_y
                    self.hex_df.at[df_old_id, 'geometry'] = new_wgs_polygon

                return True

            else:
                # Target hex does not found
                return False
        else:
            # Original hex does not found
            return False

        
    def get_geoPandas(self):
        """
        Get the geoPandas object of whole hexagon map

        Parameters
        ----------
        None

        Returns
        -------
        Hexagon map dataframe: geoPandas.GeoDataFrame
        """

        return self.hex_df