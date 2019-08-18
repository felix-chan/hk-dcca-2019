"""
Visualize the HK DC 2019 districts map
 using Leaflet.js
"""

#%%
import folium

#%%
# Create map object
map_1 = folium.Map(location=[22.375985, 114.114600], zoom_start=13)

folium.GeoJson(
    "../map_files/HKDCCA2019.json",
    tooltip = folium.features.GeoJsonTooltip(['Region', 'District'])
).add_to(map_1)

#%%
# Save the map into files
map_1.save('../docs/hkdc2019.html')