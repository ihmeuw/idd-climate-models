"""
Point GDAL and PROJ at this environment's data folders, and stop GDAL from
looking for files that do not exist.

Imported before rasterio is used (stages 2-4):

    import gdal_env  # noqa: F401

- GDAL_DATA / PROJ_DATA: when unset, rasterio searches for them every time it
  sets up a GDAL environment, i.e. on every resample or rasterize: two lookups
  of gdalvrt.xsd and one of proj_data, two of them misses, on the shared
  filesystem the conda env lives on. One stage 2 task did ~14,000 such lookups
  (~9,300 misses).
- GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR: on every GeoTIFF open GDAL otherwise
  probes for sidecar files (.ovr, .OVR, .aux, .AUX, .tif.aux, .tif.AUX): six
  misses per population-window read in stages 3 and 4. Our raster folders
  (stage 2 outputs, population model results) contain no sidecars.
- GDAL_DRIVER_PATH=disable: GDAL otherwise probes lib/gdalplugins for ~90
  optional driver plugins at start-up (all missing here; GTiff, MEM and VRT are
  built in).

Checked 2026-10-14: identical rasters (population windows at 100 m and 1 km,
stage 2 PAF) with and without the last two; misses on /mnt/team 24 -> 0.
"""

import os
import sys

os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "share", "gdal"))
os.environ.setdefault("PROJ_DATA", os.path.join(sys.prefix, "share", "proj"))
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_DRIVER_PATH", "disable")
