"""
Tests for src/idd_climate_models/climada/main/backfill_landfall_locations.py.

Run in the CLIMADA env:
    python -m unittest tests.test_climada_landfall_backfill -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "idd_climate_models" / "climada" / "main"))

import backfill_landfall_locations as bf  # noqa: E402
from stage1_run_config import build_run_config, log_dir_for  # noqa: E402

stage1 = bf.stage1
KEY = ("TEST-MODEL", "r1i1p1f1", "ssp585", "2030-2033", "AU", 48)
LAND = gpd.GeoDataFrame(
    {"loc_id": [101, 102], "ADM0_NAME": ["Islet", "Elsewhere"]},
    geometry=[box(155.65, -17.55, 155.95, -17.15), box(160, -20, 161, -19)],
    crs="EPSG:4326",
)


def storm_da(lats, lons, value=20.0, start_date="2031-02-01"):
    da = xr.DataArray(np.full((len(lats), len(lons)), value), coords={"lat": lats, "lon": lons},
                      dims=["lat", "lon"], name="intensity")
    da.attrs.update(start_date=start_date, storm_id=0)
    return da


NARROW = storm_da([-17.3, -17.4], [155.8])                                  # 2 x 1: empty records before the fix
FLAT = storm_da([-17.3], [155.7, 155.8])                                    # 1 x 2: IndexError before the fix
WIDE = storm_da([-17.2, -17.3, -17.4], [155.7, 155.8, 155.9], value=25.0)  # 3 x 3


class TestFixedLandfallRecords(unittest.TestCase):
    """compute_storm_landfall_locations pads 1-row / 1-column footprints (2026-10-12 fix)."""

    def test_one_column_footprint_gets_its_record(self):
        records = stage1.compute_storm_landfall_locations(NARROW, LAND)
        self.assertEqual([(r["loc_id"], r["max_wind_m_s"]) for r in records], [(101, 20.0)])

    def test_one_row_footprint_no_longer_raises(self):
        records = stage1.compute_storm_landfall_locations(FLAT, LAND)
        self.assertEqual([(r["loc_id"], r["max_wind_m_s"]) for r in records], [(101, 20.0)])

    def test_normal_footprint_unchanged_by_padding(self):
        self.assertEqual(
            stage1.compute_storm_landfall_locations(WIDE, LAND),
            [{"start_date": "2031-02-01", "loc_id": 101, "ADM0_NAME": "Islet", "max_wind_m_s": 25.0}],
        )


class TestBackfill(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "stage1_test"
        # Stage 1 as it ran before the fix: storm 0 narrow with an empty parquet (the bug),
        # storm 1 wide with a correct parquet.
        for storm, da in ((0, NARROW), (1, WIDE)):
            stage1.save_single_storm_metric(da, "intensity", storm, *KEY, save_root=self.root)
            records = [] if storm == 0 else stage1.compute_storm_landfall_locations(da, LAND)
            stage1.save_storm_landfall_locations(records, *KEY, storm, da.attrs["start_date"], save_root=self.root)
        stage1.write_draw_completion_marker(log_dir_for(self.root), *KEY, build_run_config(True, False))
        self.patches = [
            mock.patch.object(stage1, "load_global_land_gdf", return_value=LAND),
            mock.patch.object(bf, "all_draws", return_value=[KEY]),
        ]
        for p in self.patches:
            p.start()
        bf._LAND.clear()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp)

    def parquet(self, storm):
        return pd.read_parquet(stage1._landfall_parquet_path(*KEY, storm, self.root))

    def test_scan_finds_only_the_empty_one(self):
        out = bf.scan_draw((KEY, self.root))
        self.assertTrue(out["finished"])
        self.assertEqual(out["affected"], [0])
        self.assertEqual(out["anomalies"], [])

    def test_scan_skips_unfinished_draws(self):
        stage1._draw_log_path(log_dir_for(self.root), "draw_completion_markers", *KEY).unlink()
        self.assertEqual(bf.scan_draw((KEY, self.root))["affected"], [])

    def test_dry_run_writes_nothing(self):
        before = self.parquet(0)
        row = bf.fix_storm((KEY, 0, self.root, False))
        self.assertEqual((row["status"], row["n_records"]), ("would rewrite", 1))
        pd.testing.assert_frame_equal(self.parquet(0), before)

    def test_apply_rewrites_with_stage1_columns(self):
        row = bf.fix_storm((KEY, 0, self.root, True))
        self.assertEqual(row["status"], "rewritten")
        fixed, reference = self.parquet(0), self.parquet(1)
        self.assertEqual(list(fixed.columns), list(reference.columns))
        self.assertEqual(fixed[["storm_id", "loc_id", "max_wind_m_s", "start_date"]].values.tolist(),
                         [[0, 101, 20.0, "2031-02-01"]])
        # Once fixed, the scan no longer reports it.
        self.assertEqual(bf.scan_draw((KEY, self.root))["affected"], [])

    def test_main_refuses_until_stage1_is_finished(self):
        stage1._draw_log_path(log_dir_for(self.root), "draw_completion_markers", *KEY).unlink()
        with self.assertRaises(SystemExit):
            bf.main(self.root, apply=True, allow_incomplete=True, num_cores=1, report=self.tmp / "r.csv")
        with self.assertRaises(SystemExit):
            bf.main(self.root, apply=False, allow_incomplete=False, num_cores=1, report=self.tmp / "r.csv")

    def test_main_apply_end_to_end(self):
        bf.main(self.root, apply=True, allow_incomplete=False, num_cores=1, report=self.tmp / "r.csv")
        report = pd.read_csv(self.tmp / "r.csv")
        self.assertEqual(report[["storm_id", "status", "n_records"]].values.tolist(), [[0, "rewritten", 1]])
        self.assertEqual(len(self.parquet(0)), 1)
        self.assertFalse((self.root / "_landfall_backfill_tmp").exists())


    def test_jobmon_steps_end_to_end(self):
        """scan shards -> fix shards -> summarize, as the launcher wires them."""
        import json
        run_dir = self.tmp / "run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(json.dumps(
            {"save_root": str(self.root), "apply": True, "n_scan_shards": 2, "n_fix_shards": 2}))
        for i in range(2):
            bf.step_scan(run_dir, i)
        scan = bf._read_shards(run_dir, "scan", 2)
        self.assertEqual(scan[["storm_id", "kind"]].values.tolist(), [[0, "affected"]])
        for i in range(2):
            bf.step_fix(run_dir, i)
        bf.step_summarize(run_dir)
        report = pd.read_csv(run_dir / "report.csv", keep_default_na=False)
        self.assertEqual(report[["basin", "storm_id", "status", "n_records"]].values.tolist(), [["AU", 0, "rewritten", 1]])
        self.assertEqual(len(self.parquet(0)), 1)

    def test_fix_step_refuses_apply_with_unfinished_draws(self):
        import json
        stage1._draw_log_path(log_dir_for(self.root), "draw_completion_markers", *KEY).unlink()
        run_dir = self.tmp / "run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(json.dumps(
            {"save_root": str(self.root), "apply": True, "n_scan_shards": 1, "n_fix_shards": 1}))
        bf.step_scan(run_dir, 0)
        with self.assertRaises(SystemExit):
            bf.step_fix(run_dir, 0)
        self.assertEqual(len(self.parquet(0)), 0)


if __name__ == "__main__":
    unittest.main()
