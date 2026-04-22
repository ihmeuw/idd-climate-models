"""
Code to create bounding boxes and clip Admin 0 shapefile
Paste this into a cell in visualize_tc_risk_output.ipynb
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import box

# Helper function to parse basin bound coordinates
def parse_basin_coordinate(coord_str):
    """
    Convert coordinate string like '260E' or '45S' to numeric value.
    
    Parameters
    ----------
    coord_str : str
        Coordinate string with direction (E/W/N/S) at the end
        
    Returns
    -------
    float
        Numeric coordinate value (positive East/North, negative West/South)
    """
    value = float(coord_str[:-1])
    direction = coord_str[-1]
    
    if direction in ['W', 'S']:
        value = -value
    
    # Convert longitude from 0-360 to -180-180 if needed
    if direction in ['E', 'W'] and value > 180:
        value = value - 360
    
    return value


# 1. Get bounding box from track data in ds
if 'lon' in ds.variables and 'lat' in ds.variables:
    # For track data, lon/lat are typically 1D arrays along the track
    lon_data = ds['lon'].values
    lat_data = ds['lat'].values
    
    # Remove NaN values
    lon_data = lon_data[~np.isnan(lon_data)]
    lat_data = lat_data[~np.isnan(lat_data)]
    
    # Get track bounding box
    track_lon_min = np.min(lon_data)
    track_lon_max = np.max(lon_data)
    track_lat_min = np.min(lat_data)
    track_lat_max = np.max(lat_data)
    
    print(f"Track bounding box:")
    print(f"  Longitude: [{track_lon_min:.2f}, {track_lon_max:.2f}]")
    print(f"  Latitude: [{track_lat_min:.2f}, {track_lat_max:.2f}]")
else:
    print("Warning: Could not find 'lon' and 'lat' in dataset")
    print(f"Available variables: {list(ds.variables.keys())}")


# 2. Get basin bounds from basin_data
basin_bounds_str = basin_data.basin_bounds
print(f"\nBasin bounds (string format): {basin_bounds_str}")

# Parse basin bounds: [lon_min, lat_min, lon_max, lat_max]
basin_lon_min = parse_basin_coordinate(basin_bounds_str[0])
basin_lat_min = parse_basin_coordinate(basin_bounds_str[1])
basin_lon_max = parse_basin_coordinate(basin_bounds_str[2])
basin_lat_max = parse_basin_coordinate(basin_bounds_str[3])

print(f"Basin bounds (numeric):")
print(f"  Longitude: [{basin_lon_min:.2f}, {basin_lon_max:.2f}]")
print(f"  Latitude: [{basin_lat_min:.2f}, {basin_lat_max:.2f}]")


# 3. Create combined bounding box (track + basin)
combined_lon_min = min(track_lon_min, basin_lon_min)
combined_lon_max = max(track_lon_max, basin_lon_max)
combined_lat_min = min(track_lat_min, basin_lat_min)
combined_lat_max = max(track_lat_max, basin_lat_max)

# Add a buffer (e.g., 5 degrees) to make it larger
buffer_deg = 5
final_lon_min = combined_lon_min - buffer_deg
final_lon_max = combined_lon_max + buffer_deg
final_lat_min = max(combined_lat_min - buffer_deg, -90)  # Don't go below -90
final_lat_max = min(combined_lat_max + buffer_deg, 90)   # Don't go above 90

print(f"\nFinal bounding box (with {buffer_deg}° buffer):")
print(f"  Longitude: [{final_lon_min:.2f}, {final_lon_max:.2f}]")
print(f"  Latitude: [{final_lat_min:.2f}, {final_lat_max:.2f}]")


# 4. Read Admin 0 shapefile
print(f"\n Loading Admin 0 shapefile from: {ADMIN0_SHP_FILENAME}")
admin0_gdf = gpd.read_file(ADMIN0_SHP_FILENAME)
print(f"✓ Loaded {len(admin0_gdf)} admin 0 regions")
print(f"  CRS: {admin0_gdf.crs}")
print(f"  Columns: {list(admin0_gdf.columns)}")


# 5. Clip to the final bounding box
# Create a bounding box polygon
bbox_polygon = box(final_lon_min, final_lat_min, final_lon_max, final_lat_max)

# Clip the shapefile
admin0_clipped = admin0_gdf.clip(bbox_polygon)
print(f"\n✓ Clipped to {len(admin0_clipped)} admin 0 regions within bounding box")

# Show a preview
print(f"\nClipped Admin 0 regions:")
print(admin0_clipped[['loc_name', 'geometry']].head())
