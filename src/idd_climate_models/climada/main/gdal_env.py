"""
Point GDAL and PROJ at this environment's data folders.

When GDAL_DATA / PROJ_DATA are unset, rasterio searches for them every time it
sets up a GDAL environment, which in stages 2-4 is every resample or
rasterize: two lookups of gdalvrt.xsd and one of proj_data, two of them
misses, on the shared filesystem the conda env lives on. One stage 2 task did
~14,000 such lookups (~9,300 misses). Setting both once stops the search.

Import this module before rasterio is used:

    import gdal_env  # noqa: F401
"""

import os
import sys

os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "share", "gdal"))
os.environ.setdefault("PROJ_DATA", os.path.join(sys.prefix, "share", "proj"))
