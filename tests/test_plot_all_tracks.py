"""Tests for plot_all_tracks."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock

from idd_climate_models.storm_data.visualizers import plot_all_tracks
from idd_climate_models.storm_data.tc_risk import TCRiskBasinData


@pytest.fixture
def sample_track_ds():
    """Create a minimal TC Risk track dataset with 5 storms."""
    n_trk = 5
    n_pnt = 20
    rng = np.random.default_rng(42)

    lon = rng.uniform(180, 290, size=(n_trk, n_pnt))
    lat = rng.uniform(5, 40, size=(n_trk, n_pnt))
    wind = rng.uniform(10, 60, size=(n_trk, n_pnt))

    # Mark last 5 points of each track as NaN (varying track lengths)
    for i in range(n_trk):
        cutoff = n_pnt - rng.integers(0, 5)
        lon[i, cutoff:] = np.nan
        lat[i, cutoff:] = np.nan
        wind[i, cutoff:] = np.nan

    basins = np.array(["EP", "EP", "NA", "EP", "NA"])
    years = np.array([2090, 2090, 2091, 2091, 2092])
    months = np.array([6, 9, 7, 1, 3])

    ds = xr.Dataset(
        {
            "lon_trks": (["n_trk", "n_pnt"], lon),
            "lat_trks": (["n_trk", "n_pnt"], lat),
            "vmax_trks": (["n_trk", "n_pnt"], wind),
            "tc_basins": (["n_trk"], basins),
            "tc_years": (["n_trk"], years.astype(float)),
            "tc_month": (["n_trk"], months.astype(float)),
        }
    )
    return ds


@pytest.fixture
def basin_data():
    return TCRiskBasinData(
        model="CMCC-ESM2",
        variant="r1i1p1f1",
        scenario="ssp245",
        time_period="2089-2092",
        basin="EP",
    )


def _mock_read(ds):
    """Return a function that returns the dataset (simulates read_track_file)."""
    def _read(draw=0):
        return ds
    return _read


class TestPlotAllTracksFiltering:
    """Test that date, basin, and country filters work correctly."""

    def test_no_filters_returns_all(self, basin_data, sample_track_ds):
        with patch.object(basin_data, "read_track_file", _mock_read(sample_track_ds)):
            fig, ax, info = plot_all_tracks(
                basin_data, draw=0, use_cartopy=False,
            )
        assert len(info) == 5
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_basin_filter_ep(self, basin_data, sample_track_ds):
        with patch.object(basin_data, "read_track_file", _mock_read(sample_track_ds)):
            fig, ax, info = plot_all_tracks(
                basin_data, draw=0, basin_filter="EP", use_cartopy=False,
            )
        assert set(info["basin"]) == {"EP"}
        assert len(info) == 3
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_basin_filter_gl_returns_all(self, basin_data, sample_track_ds):
        with patch.object(basin_data, "read_track_file", _mock_read(sample_track_ds)):
            fig, ax, info = plot_all_tracks(
                basin_data, draw=0, basin_filter="GL", use_cartopy=False,
            )
        assert len(info) == 5
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_date_filter_start(self, basin_data, sample_track_ds):
        with patch.object(basin_data, "read_track_file", _mock_read(sample_track_ds)):
            fig, ax, info = plot_all_tracks(
                basin_data, draw=0, start_date=(2091, 1), use_cartopy=False,
            )
        # Should include storms with year>=2091 (tracks 2,3,4)
        assert all(info["year"] >= 2091)
        assert len(info) == 3
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_date_filter_end(self, basin_data, sample_track_ds):
        with patch.object(basin_data, "read_track_file", _mock_read(sample_track_ds)):
            fig, ax, info = plot_all_tracks(
                basin_data, draw=0, end_date=(2090, 12), use_cartopy=False,
            )
        assert all(info["year"] <= 2090)
        assert len(info) == 2
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_date_and_basin_combined(self, basin_data, sample_track_ds):
        with patch.object(basin_data, "read_track_file", _mock_read(sample_track_ds)):
            fig, ax, info = plot_all_tracks(
                basin_data, draw=0,
                start_date=(2091, 1), basin_filter="EP",
                use_cartopy=False,
            )
        # EP storms from 2091+: track 3 (EP, 2091, month 1)
        assert len(info) == 1
        assert info.iloc[0]["basin"] == "EP"
        assert info.iloc[0]["year"] == 2091
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_no_matches_returns_empty(self, basin_data, sample_track_ds):
        with patch.object(basin_data, "read_track_file", _mock_read(sample_track_ds)):
            fig, ax, info = plot_all_tracks(
                basin_data, draw=0, basin_filter="SI", use_cartopy=False,
            )
        assert len(info) == 0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_country_filter_requires_admin_gdf(self, basin_data, sample_track_ds):
        with patch.object(basin_data, "read_track_file", _mock_read(sample_track_ds)):
            with pytest.raises(ValueError, match="admin_gdf is required"):
                plot_all_tracks(
                    basin_data, draw=0, country_filter="any", use_cartopy=False,
                )

    def test_returns_dataframe_with_expected_columns(self, basin_data, sample_track_ds):
        with patch.object(basin_data, "read_track_file", _mock_read(sample_track_ds)):
            fig, ax, info = plot_all_tracks(
                basin_data, draw=0, use_cartopy=False,
            )
        expected_cols = {"track_index", "year", "month", "basin", "max_wind", "n_points"}
        assert expected_cols.issubset(set(info.columns))
        import matplotlib.pyplot as plt
        plt.close(fig)
