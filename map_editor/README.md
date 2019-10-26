# GEOJSON map editor

It is a map editor to fine-tune the position of region on map. The files inside `./geojson` are just for proof of concept. They may not suitable to use in production.

## Trial run

Run the following code to see the effect of editor:

```sh
set FLASK_APP=main.py
python -m flask run
```

When the local web server is ready, access http://localhost:8000/

## Server design

The server side will do the following jobs:

 - Read intermediate geojson file if exists `./edit/save_cache.geojson`
 - Read original geojson if intermediate geojson does not exists `./geojson/hex_input.geojson`
 - Produce up-to-date json for clinent side
 - Get the submission and update the intermediate geojson file

There will be 2 API endpoint:

 - `/hex_grid/`: (GET) Return the base grid of map
 - `/get_map/`: (GET|POST) Post the swap information of 2 hexagon. Return the geojson of map

There will be 1 entrance:

 - `/`: The map editor