"""Utilities for shifting GeoDataFrames between -180/180 and 0/360 longitude conventions."""

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

LEFT_BOX = box(-180, -90, 0, 90)
RIGHT_BOX = box(0, -90, 180, 90)
_BUFFER = 1e-4


def _shift_left_piece(geom):
    """Shift a piece clipped to [-180, 0] to [180, 360]. Uses <= 0 so coords at exactly 0 shift to 360."""
    def shift_coords(coords):
        return [(x + 360 if x <= 0 else x, y) for x, y in coords]
    def shift_poly(poly):
        return Polygon(shift_coords(poly.exterior.coords),
                       [shift_coords(r.coords) for r in poly.interiors])
    if geom.geom_type == "Polygon":
        return shift_poly(geom)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([shift_poly(p) for p in geom.geoms])
    return geom


def _shift_normal(geom):
    """Shift a geometry entirely in negative space (maxx <= 0) to positive space."""
    def shift_coords(coords):
        return [(x + 360 if x < 0 else x, y) for x, y in coords]
    def shift_poly(poly):
        return Polygon(shift_coords(poly.exterior.coords),
                       [shift_coords(r.coords) for r in poly.interiors])
    if geom.geom_type == "Polygon":
        return shift_poly(geom)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([shift_poly(p) for p in geom.geoms])
    return geom


def _reproject_geometry_to_360(geom) -> object:
    """Convert a single geometry from -180/180 to 0/360 longitude space."""
    minx, _, maxx, _ = geom.bounds

    if minx >= 0:
        return geom

    if maxx <= 0:
        return _shift_normal(geom)

    # Mixed: handle part by part
    parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    result_parts = []

    for part in parts:
        pminx, _, pmaxx, _ = part.bounds

        if pminx < -180 + _BUFFER and pmaxx > 180 - _BUFFER:
            # Spans almost the full globe (e.g. Russia at antimeridian)
            left = part.intersection(LEFT_BOX)
            right = part.intersection(RIGHT_BOX)
            left_shifted = _shift_left_piece(left)
            combined = unary_union(
                [left_shifted.buffer(_BUFFER), right.buffer(_BUFFER)]
            ).buffer(-_BUFFER)
            result_parts.append(combined)
        elif pminx < -180 + _BUFFER:
            # Antimeridian fragment on negative side
            result_parts.append(_shift_normal(part))
        elif pmaxx > 180 - _BUFFER:
            # Antimeridian fragment on positive side: keep as-is
            result_parts.append(part)
        elif pmaxx <= 0:
            result_parts.append(_shift_normal(part))
        elif pminx >= 0:
            result_parts.append(part)
        else:
            # Crosses prime meridian: split, shift left half, keep right
            left = part.intersection(LEFT_BOX)
            right = part.intersection(RIGHT_BOX)
            result_parts.append(_shift_left_piece(left))
            result_parts.append(right)

    return unary_union(result_parts)


def reproject_gdf_to_360(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert a GeoDataFrame from -180/180 to 0/360 longitude convention."""
    out = gdf.copy()
    out["geometry"] = out["geometry"].apply(_reproject_geometry_to_360)
    return out
