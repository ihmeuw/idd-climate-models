from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import snap, unary_union
from shapely.affinity import translate
import geopandas as gpd
from shapely import snap

# Clipping boxes
LEFT_BOX = box(-180, -90, 0, 90)   # west of prime meridian, will become 180-360
RIGHT_BOX = box(0, -90, 180, 90)   # east of prime meridian, stays 0-180
ERROR = 1e-3
BUFFER = 1e-4

def crosses_prime_meridian(geom):
    """Check if geometry crosses 0 longitude."""
    minx, _, maxx, _ = geom.bounds
    return minx < 0 and maxx > 0

def shift_left_piece(geom):
    """Shift a piece clipped to [-180, 0] to [180, 360].
    Uses <= 0 to ensure coords at exactly 0 also get shifted to 360."""
    def shift_coords(coords):
        return [(x + 360 if x <= 0 else x, y) for x, y in coords]
    def shift_poly(poly):
        return Polygon(shift_coords(poly.exterior.coords),
                       [shift_coords(r.coords) for r in poly.interiors])
    if geom.geom_type == 'Polygon':
        return shift_poly(geom)
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([shift_poly(p) for p in geom.geoms])
    return geom

def shift_normal(geom):
    """Shift a geometry entirely in negative space (maxx <= 0) to positive space.
    Uses < 0 since there's no split seam to worry about."""
    def shift_coords(coords):
        return [(x + 360 if x < 0 else x, y) for x, y in coords]
    def shift_poly(poly):
        return Polygon(shift_coords(poly.exterior.coords),
                       [shift_coords(r.coords) for r in poly.interiors])
    if geom.geom_type == 'Polygon':
        return shift_poly(geom)
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([shift_poly(p) for p in geom.geoms])
    return geom

def reproject_geometry(geom):
    minx, _, maxx, _ = geom.bounds

    # Case 1: entirely positive
    if minx >= 0:
        return geom

    # Case 2: entirely negative
    if maxx <= 0:
        return shift_normal(geom)

    # Case 3: mixed — handle part by part
    if geom.geom_type == 'Polygon':
        parts = [geom]
    else:
        parts = list(geom.geoms)

    result_parts = []
    for part in parts:
        pminx, _, pmaxx, _ = part.bounds
        if pminx < -180 + BUFFER and pmaxx > 180 - BUFFER:
            left = part.intersection(LEFT_BOX)
            right = part.intersection(RIGHT_BOX)
            left_shifted = shift_left_piece(left)
            combined = unary_union([left_shifted.buffer(BUFFER), right.buffer(BUFFER)]).buffer(-BUFFER)
            result_parts.append(combined)
        elif pminx < -180 + BUFFER:
            # Antimeridian fragment on negative side: shift to positive
            result_parts.append(shift_normal(part))
        elif pmaxx > 180 - BUFFER:
            # Antimeridian fragment on positive side: keep as-is
            result_parts.append(part)
        elif pmaxx <= 0:
            # Entirely negative
            result_parts.append(shift_normal(part))
        elif pminx >= 0:
            # Entirely positive
            result_parts.append(part)
        else:
            # Crosses prime meridian normally
            left = part.intersection(LEFT_BOX)
            right = part.intersection(RIGHT_BOX)
            result_parts.append(shift_left_piece(left))
            result_parts.append(right)

    return unary_union(result_parts)

# --------------------------
# Load shapefile
# Note: this shapefile is in EPSG:4326, so the antimeridian is at -180/180, not 0/360
# --------------------------

from pathlib import Path
SHAPEFILE_ROOT = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2024_07_29')
ADMIN_SHP_FILENAME_TEMPLATE = f"{SHAPEFILE_ROOT}/lbd_standard_admin_{{admin_level}}{{simple_suffix}}.shp"
admin_level = 0
simplified_suffix = ''

# Load original
shp_path = ADMIN_SHP_FILENAME_TEMPLATE.format(
    admin_level=admin_level,
    simple_suffix=simplified_suffix
)
gdf_original = gpd.read_file(shp_path)

# --------------------------
# Reproject geometries
# --------------------------
gdf_shifted = gdf_original.copy()
gdf_shifted['geometry'] = gdf_shifted['geometry'].apply(reproject_geometry)

# Save results
# gdf_shifted.to_parquet('shapefile_0_360.parquet')
# gdf_original.to_parquet('shapefile_-180_180.parquet')

# Plot results for visual check
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Plot
# fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# gdf_original.plot(ax=axes[0], color='steelblue', edgecolor='white', linewidth=0.3)
# axes[0].set_title('Original (-180 to 180)', fontsize=14)
# axes[0].set_xlabel('Longitude')
# axes[0].set_ylabel('Latitude')
# axes[0].set_xlim(-180, 180)

# gdf_shifted.plot(ax=axes[1], color='coral', edgecolor='white', linewidth=0.3)
# axes[1].set_title('Shifted (0 to 360)', fontsize=14)
# axes[1].set_xlabel('Longitude')
# axes[1].set_ylabel('Latitude')
# axes[1].set_xlim(0, 360)

# plt.tight_layout()
# plt.savefig('shapefile_comparison.png', dpi=150, bbox_inches='tight')
# plt.show()