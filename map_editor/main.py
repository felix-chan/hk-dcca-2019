"""
Main Flask server script to handle the change of DC location
"""

from flask import Flask, render_template, request, redirect, Response, \
    send_from_directory
import pandas as pd
import numpy as np
import geopandas as gpd

from hexMap import hexMap

import os

# Read the json file
if os.path.isfile('./edit/save_cache.geojson'):
    current_map_loc = './edit/save_cache.geojson'
else:
    current_map_loc = './geojson/hex_input.geojson'

current_grid_loc = './geojson/hex_base_grid.geojson'

map_object = hexMap(current_map_loc, current_grid_loc, 800)

# Save json function
def save_json(json_string):
    with open('./edit/save_cache.geojson', 'w') as file:
        file.write(json_string)

def check_int(val):
    try:
        val = int(val)
        return True
    except Exception as e:
        return False

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    # Redirect to HA homepage

    return render_template('map_editor.html')

@app.route("/hex_grid/", methods=['GET'])
def show_grid():
    return send_from_directory('./geojson', 'hex_base_grid.geojson')

@app.route("/get_map/", methods=['GET', 'POST'])
def show_map():
    message = ''
    save_json_file = False
    if request.method == 'POST':
        from_x = request.form['from_x']
        from_y = request.form['from_y']
        to_x = request.form['to_x']
        to_y = request.form['to_y']

        if check_int(from_x) & check_int(from_y) & \
            check_int(to_x) & check_int(to_y):

            switch_status = map_object.swap_position(from_x, from_y, to_x, to_y)

            if switch_status:
                message = 'Successfully change the location'
                save_json_file = True
            else:
                message = 'Invalid input'
        else:
            message = 'Please submit integer for grid location'

    json_file = map_object.gen_json()
    if save_json_file:
        save_json(json_file)

    return Response(json_file, mimetype='application/json')

