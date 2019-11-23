"""
Hex Map Library for editing the geojson file
  where the geojson file are Coordinate system independent

Required hexMap.py

Example code
```
test1 = hexMapEditor('hex_map.geojson', 'hex_grid.geojson', 1)

# Plot the map
test1.plot_hex(base_grid=True, text=True, save_name='plot1.png')

# Swap position of 2 hexagon
test1.swap_position(8,16,8,26, coord='custom')

# Shift position of list of hexagon
CACODE_move_right = ['T01', 'T02', 'T04', 'T06']
test1.shift_position(CACODE_move_right, left=-1)

# Export the final GeoDataFrame
geo_df = test1.get_geoPandas()
```
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

from hexMap import hexMap

class hexMapEditor(hexMap):
    """
    Hexagon map editor object which can read or generate json file
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

    def __init__(self, file_loc, hex_grid_loc, hex_width = 100):
        hex_dataframe = gpd.read_file(file_loc)
        hex_dataframe['CACODE'] = hex_dataframe['id']
        hex_dataframe['id'] = range(len(hex_dataframe['CACODE']))
        hex_dataframe['x'] = hex_dataframe['hex_x']
        hex_dataframe['y'] = hex_dataframe['hex_y']
        hex_dataframe['ENAME'] = ''
        hex_dataframe['CNAME'] = ''
        hex_dataframe['seat'] = 1
        hex_dataframe['district'] = hex_dataframe['CACODE']\
            .str.replace('([A-Z]).*', '\\1')
        hex_dataframe.drop(['hex_x', 'hex_y'], axis=1, inplace=True)
        self.hex_df = hex_dataframe
        self.grid_df = gpd.read_file(hex_grid_loc)
        self.hex_width = hex_width
        
    def plot_hex(self, cacode_list = None, text = False, base_grid = False, 
        save_name = None):
        """
        Plot the Hexagon map on screen or save as png file

        Parameters
        ----------
        cacode_list: list or None
            Please provide a list if specific CACODE should be plotted on graph
        text: boolean
            Set True to display the CACODE on graph
        base_grid: boolean
            Set True to display the base grid on graph
        save_name: string or none
            Set a file name if the graph needed to be saved
        """
        _, ax = plt.subplots()
        ax.set_aspect('equal')

        if cacode_list:
            selected_hex = self.hex_df[self.hex_df['CACODE'].isin(cacode_list)]
        else:
            selected_hex = self.hex_df
        
        if base_grid:
            base_hex = [self._draw_hex(x.x, x.y, self.hex_width, coord='custom') 
                       for x in self.grid_df['geometry']]
            base_hex_gdf = gpd.GeoSeries(base_hex)
            base_hex_gdf.plot(ax=ax, color='white', edgecolor='#ffbcb8')
        
        selected_hex.plot(ax=ax, column='district')
        
        if text:
            for _, items in selected_hex.iterrows():
                ax.text(items['x'], items['y'], items['CACODE'], 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    fontsize=4)
                
        if save_name is not None:
            plt.savefig(save_name, dpi=200)
            
        plt.show()
        
    def shift_position(self, cacode_list, left = 0, up = 0):
        """
        Shift the position on a list of hexagon
        
        Parameters
        ----------
        cacode_list: list
            List if target specific CACODE
        left: int
            Horizontal movements, negative number as move to right
        up: int
            Verticle movements, negative number as move to bottom

        Returns
        -------
        Success status of operation: boolean
        """
        # Selected the affected positions
        affected_pos = self.hex_df[self.hex_df['CACODE'].isin(cacode_list)]\
            .copy()
        
        if len(self.hex_df['CACODE']) == 0:
            raise ValueError('Incorrect CACODE')
            
        if up % 2 == 0:
            update_x = 2 * left
            update_y = up
        else:
            update_x = 2 * left + 1
            update_y = up
            
        # Select the replaced position
        affected_pos['new_xid'] = affected_pos['xid'] + update_x
        affected_pos['new_yid'] = affected_pos['yid'] + update_y
        
        overlap_pos = affected_pos[['new_xid', 'new_yid']]\
            .rename({'new_xid': 'xid', 'new_yid': 'yid'}, axis=1)\
            .merge(self.hex_df, on=['xid', 'yid'], how='inner')
        
        if overlap_pos.shape[0] > 0:
            print(overlap_pos)
            raise ValueError('Some space will be replaced with shift')
        
        # Check the range
        check_exists = affected_pos[['new_xid', 'new_yid']]\
            .rename({'new_xid': 'xid', 'new_yid': 'yid'}, axis=1)\
            .merge(self.grid_df, on=['xid', 'yid'], how='left')
        
        if np.sum(check_exists['id'].isna()) > 0:
            raise ValueError('New position out of boundary')
            
        # Move to new position
        for _, rows in affected_pos.iterrows():
            self.swap_position(
                rows.xid, 
                rows.yid, 
                rows.new_xid, 
                rows.new_yid, 
                coord='custom'
            )

        return True