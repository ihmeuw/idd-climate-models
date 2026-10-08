"""
Tests for stage 1 central-pressure re-estimation and the output-overwrite check
in src/idd_climate_models/climada/main/01_climada_intensity_main.py.

Run in the CLIMADA env:
    python -m unittest tests.test_climada_stage1_central_pressure -v
"""

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import xarray as xr
from climada.hazard.tc_tracks import _estimate_pressure

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = REPO_ROOT / "src" / "idd_climate_models" / "climada" / "main"
sys.path.insert(0, str(MAIN_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from idd_climate_models.tc_estimation import _estimate_pressure as idd_estimate_pressure  # noqa: E402
from stage1_run_config import LEGACY_RUN_CONFIG, MS_TO_KN, build_run_config  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "climada_intensity_main", MAIN_DIR / "01_climada_intensity_main.py"
)
stage1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage1)

REAL_INPUT_FILE = Path(
    "/mnt/team/rapidresponse/pub/tropical-storms/climada/input/cmip6/EC-Earth3/r1i1p1f1/"
    "ssp585/2020-2024/EP/tracks_EP_EC-Earth3_ssp585_r1i1p1f1_202001_202412_e0.nc"
)


def make_input_dataset(lon=None) -> xr.Dataset:
    """Small CLIMADA-input-shaped dataset: 3 tracks x 6 steps, NaN padding at the end."""
    rng = np.random.default_rng(0)
    shape = (3, 6)
    lat = rng.uniform(-35, 35, shape)
    if lon is None:
        lon = rng.uniform(-179, 179, shape)
    vmax_ms = rng.uniform(15, 80, shape)
    for arr in (lat, lon, vmax_ms):
        arr[:, 4:] = np.nan
    lon = np.asarray(lon, dtype=float)
    return xr.Dataset(
        coords={"n_trk": np.arange(3), "time": np.arange(6) * 3600.0},
        data_vars={
            "lon": (("n_trk", "time"), lon),
            "lat": (("n_trk", "time"), lat),
            "max_sustained_wind": (("n_trk", "time"), vmax_ms),
            # Stand-in for the stored ensemble estimate; must be discarded.
            "central_pressure": (("n_trk", "time"), np.full(shape, 900.0)),
            "environmental_pressure": (("n_trk", "time"), 1013 - 0.15 * np.abs(lat)),
            "time_step": (("n_trk", "time"), np.full(shape, 1.0)),
            "tc_years": (("n_trk",), np.array([2020, 2021, 2022])),
            "tc_month": (("n_trk",), np.array([8, 9, 10])),
            "tc_basins": (("n_trk",), np.array(["EP", "EP", "EP"])),
            "category": (("n_trk",), np.array([3, 4, 5])),
        },
        attrs={"max_sustained_wind_unit": "kn"},
    )


class TestCentralPressureReestimation(unittest.TestCase):

    def test_nan_then_estimate_equals_climada_fit_everywhere(self):
        ds = make_input_dataset()
        out = stage1.estimate_central_pressure_with_climada(ds)

        expected = _estimate_pressure(
            np.full(ds["lat"].shape, np.nan),
            ds["lat"].values,
            ds["lon"].values % 360,
            ds["max_sustained_wind"].values * MS_TO_KN,
        )
        np.testing.assert_array_equal(out["central_pressure"].values, expected)
        # Padding stays NaN; every real step gets an estimate, not the stored 900.
        self.assertTrue(np.isnan(out["central_pressure"].values[:, 4:]).all())
        self.assertTrue(np.isfinite(out["central_pressure"].values[:, :4]).all())
        self.assertFalse(np.any(out["central_pressure"].values == 900.0))

    def test_environmental_pressure_and_other_variables_unchanged(self):
        ds = make_input_dataset()
        out = stage1.estimate_central_pressure_with_climada(ds)
        for var in ("environmental_pressure", "lon", "lat", "max_sustained_wind"):
            xr.testing.assert_identical(out[var], ds[var])

    def test_lon_fit_in_0_360_like_the_pipeline(self):
        # Input files store -110; TC-risk and add_tc_params use 250.
        ds_east = make_input_dataset(lon=np.full((3, 6), 250.0))
        ds_west = ds_east.assign(lon=ds_east["lon"] - 360.0)
        expected = _estimate_pressure(
            np.full((3, 6), np.nan), ds_east["lat"].values, ds_east["lon"].values,
            ds_east["max_sustained_wind"].values * MS_TO_KN,
        )
        np.testing.assert_array_equal(
            stage1.estimate_central_pressure_with_climada(ds_west)["central_pressure"].values, expected,
        )
        np.testing.assert_allclose(
            stage1.estimate_central_pressure_with_climada(ds_east)["central_pressure"].values,
            stage1.estimate_central_pressure_with_climada(ds_west)["central_pressure"].values,
            rtol=0, atol=1e-9,
        )


class TestEstimatePressureImplementationsAgree(unittest.TestCase):

    def _assert_agree(self, lat, lon, v_kn):
        nan = np.full(np.shape(lat), np.nan)
        np.testing.assert_array_equal(
            _estimate_pressure(nan.copy(), lat, lon, v_kn),
            idd_estimate_pressure(nan.copy(), lat, lon, v_kn),
        )

    def test_agree_on_synthetic_grid(self):
        lat, lon, v = np.meshgrid(
            np.linspace(-45, 45, 19), np.linspace(-180, 180, 25), np.linspace(0, 180, 13),
        )
        v[0, 0, :] = np.nan
        self._assert_agree(lat.ravel(), lon.ravel(), v.ravel())

    @unittest.skipUnless(REAL_INPUT_FILE.exists(), "CLIMADA input file not available")
    def test_agree_on_one_real_track(self):
        with xr.open_dataset(REAL_INPUT_FILE) as ds:
            track = ds.isel(n_trk=int(np.nanargmax(ds["max_sustained_wind"].max("time").values))).load()
        self._assert_agree(
            track["lat"].values, track["lon"].values, track["max_sustained_wind"].values * MS_TO_KN,
        )


class TestWindUnitConversion(unittest.TestCase):

    def test_convert_wind_to_knots_only_when_asked(self):
        ds_track = make_input_dataset().isel(n_trk=0)
        vmax = ds_track["max_sustained_wind"].values
        default = stage1.normalize_nc_storm_for_climada(ds_track)
        converted = stage1.normalize_nc_storm_for_climada(ds_track, convert_wind_to_knots=True)
        np.testing.assert_array_equal(default["max_sustained_wind"].values, vmax)
        np.testing.assert_array_equal(converted["max_sustained_wind"].values, vmax * MS_TO_KN)
        self.assertEqual(converted.attrs["max_sustained_wind_unit"], "kn")


class TestOverwriteCheck(unittest.TestCase):

    IDS = ("TEST-MODEL", "r1i1p1f1", "ssp585", "2020-2024", "EP")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.save_root = self.tmp / "stage1_test"
        self.log_dir = stage1.log_dir_for(self.save_root)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _tree(self):
        return sorted(p.relative_to(self.tmp) for p in self.tmp.rglob("*"))

    def _make_existing_results(self, draw, run_config=None):
        """Lay down a draw's outputs; run_config=None mimics pre-run-config results."""
        paths = stage1._draw_result_paths(self.save_root, self.log_dir, *self.IDS, draw)
        intensity_store, landfall_dir = paths[0], paths[2]
        intensity_store.mkdir(parents=True)
        landfall_dir.mkdir(parents=True)
        (landfall_dir / "storm_0000.parquet").write_bytes(b"")
        if run_config is not None:
            stage1.write_draw_run_config(self.log_dir, *self.IDS, draw, run_config)

    def _main(self, **kwargs):
        source_id, variant_label, experiment_id, batch_year, basin = self.IDS
        stage1.main(
            source_id=source_id, variant_label=variant_label, experiment_id=experiment_id,
            batch_year=batch_year, basin=basin, draw_batch="0-1", num_cores=1,
            save_root=self.save_root, **kwargs,
        )

    def test_dies_and_writes_nothing_when_results_exist_and_overwrite_off(self):
        self._make_existing_results(draw=0)
        before = self._tree()
        with mock.patch.object(stage1, "run_parallel", side_effect=AssertionError("must not run")):
            with self.assertRaises(SystemExit) as cm:
                self._main(reestimate_central_pressure=True)
        self.assertEqual(cm.exception.code, 1)
        self.assertEqual(self._tree(), before)

    def test_dies_when_recorded_config_differs(self):
        self._make_existing_results(draw=1, run_config=build_run_config(True, True))
        before = self._tree()
        with mock.patch.object(stage1, "run_parallel", side_effect=AssertionError("must not run")):
            with self.assertRaises(SystemExit):
                self._main(reestimate_central_pressure=True)
        self.assertEqual(self._tree(), before)

    def test_same_config_resumes(self):
        config = build_run_config(True, False)
        self._make_existing_results(draw=0, run_config=config)
        with mock.patch.object(stage1, "run_parallel", return_value=[[], []]) as run:
            self._main(reestimate_central_pressure=True)
        run.assert_called_once()

    def test_pre_run_config_results_resume_under_default_options(self):
        self._make_existing_results(draw=0)
        with mock.patch.object(stage1, "run_parallel", return_value=[[], []]) as run:
            self._main()
        run.assert_called_once()

    def test_overwrite_okay_proceeds_and_threads_options(self):
        self._make_existing_results(draw=0)
        with mock.patch.object(stage1, "run_parallel", return_value=[[], []]) as run:
            self._main(reestimate_central_pressure=True, overwrite_okay=True)
        draw_args = run.call_args.kwargs["arg_list"]
        self.assertEqual([a[5] for a in draw_args], [0, 1])
        for a in draw_args:
            self.assertEqual(a[6:], (self.save_root, self.log_dir, True, False, True))

    def test_overwrite_removes_old_results_before_processing_draw(self):
        self._make_existing_results(draw=0)
        draw_info = (*self.IDS, 0, self.save_root, self.log_dir, True, False, True)
        # TEST-MODEL has no input file, so the draw stops right after the
        # overwrite step.
        with self.assertRaises(FileNotFoundError):
            stage1.process_single_draw(draw_info)
        paths = stage1._draw_result_paths(self.save_root, self.log_dir, *self.IDS, 0)
        self.assertFalse(paths[0].exists())
        self.assertFalse(paths[2].exists())
        self.assertEqual(
            stage1.recorded_draw_run_config(self.save_root, self.log_dir, *self.IDS, 0),
            build_run_config(True, False),
        )

    def test_legacy_config_is_default_options(self):
        self.assertEqual(LEGACY_RUN_CONFIG, build_run_config(False, False))
        self.assertEqual(
            stage1.log_dir_for(Path("/x/stage1_v3")), Path("/x/stage1_v3_log"),
        )


if __name__ == "__main__":
    unittest.main()
