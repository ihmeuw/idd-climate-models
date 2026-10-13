"""
Tests for src/idd_climate_models/climada/main/storm_zarr.py.
Run in the CLIMADA env:
    python -m unittest tests.test_climada_storm_zarr -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import xarray as xr
from zarr.storage import LocalStore

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "idd_climate_models" / "climada" / "main"))

from storm_zarr import ChunkManifestStore, open_storm  # noqa: E402


def write_sparse_storm(store_root: Path, group: str = "storm_0000") -> Path:
    """A storm group laid out like stage 1's: 64x64 chunks, NaN fill, land-only values."""
    lats = np.round(np.arange(-20, -20 + 0.1 * 150, 0.1), 4)
    lons = np.round(np.arange(150, 150 + 0.1 * 200, 0.1), 4)
    values = np.full((150, 200), np.nan, dtype="float32")
    values[:40, :50] = 25.0      # one corner of "land": most chunks stay all-NaN
    values[130:, 180:] = 18.0
    da = xr.DataArray(values, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"], name="intensity")
    ds = da.chunk({"lat": 64, "lon": 64}).to_dataset()
    ds.attrs.update(start_date="2031-02-01", end_date="2031-02-09", storm_id=0)
    ds.to_zarr(store_root, group=group, mode="a", zarr_format=3, consolidated=False,
               encoding={"intensity": {"fill_value": np.nan, "dtype": "float32"}})
    return store_root / group


class OpenStormTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.storm = write_sparse_storm(self.tmp / "draw.zarr")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_fixture_is_sparse(self):
        written = list((self.storm / "intensity" / "c").rglob("*"))
        n_files = sum(p.is_file() for p in written)
        self.assertLess(n_files, 3 * 4)  # 3x4 chunk grid, most never written

    def test_same_data_and_attrs_as_plain_open(self):
        plain = xr.open_zarr(self.storm, consolidated=False, zarr_format=3, chunks="auto")
        fast = open_storm(self.storm, chunks="auto")
        xr.testing.assert_identical(plain.load(), fast.load())
        self.assertEqual(fast.attrs["start_date"], "2031-02-01")

    def test_missing_chunks_are_never_looked_up(self):
        requested = []
        real_get = LocalStore.get

        async def spy(self, key, prototype=None, byte_range=None):
            requested.append(key)
            return await real_get(self, key, prototype, byte_range)

        existing = {p.relative_to(self.storm).as_posix()
                    for p in self.storm.rglob("*") if p.is_file() and "/c/" in p.as_posix()}
        with mock.patch.object(LocalStore, "get", spy):
            open_storm(self.storm, chunks="auto")["intensity"].values
        chunk_reads = {k for k in requested if "/c/" in k}
        self.assertTrue(chunk_reads)
        self.assertLessEqual(chunk_reads, existing)  # only chunks that exist reach the filesystem

    def test_all_nan_array_without_chunk_dir(self):
        storm = self.tmp / "empty.zarr" / "storm_0001"
        da = xr.DataArray(np.full((10, 10), np.nan, dtype="float32"), dims=["lat", "lon"],
                          coords={"lat": np.arange(10.0), "lon": np.arange(10.0)}, name="intensity")
        da.to_dataset().to_zarr(storm.parent, group=storm.name, mode="a", zarr_format=3, consolidated=False,
                                encoding={"intensity": {"fill_value": np.nan}})
        self.assertFalse((storm / "intensity" / "c").exists())
        self.assertTrue(np.isnan(open_storm(storm)["intensity"].values).all())

    def test_store_is_read_only(self):
        self.assertTrue(ChunkManifestStore(self.storm).read_only)


if __name__ == "__main__":
    unittest.main()
