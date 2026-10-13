import gdal_env  # noqa: F401  (sets GDAL_DATA/PROJ_DATA before rasterio is used)
from pathlib import Path
from typing import NamedTuple
import xarray as xr  # type: ignore
import numpy as np  # type: ignore
import rasterra as rt  # type: ignore
import pandas as pd  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
from affine import Affine  # type: ignore
import os
import argparse
import gc
import json
import re
import time
import traceback
from rra_tools.parallel import run_parallel  # type: ignore
from storm_zarr import open_storm
from rasterra import RasterArray  # type: ignore

# Constants
ROOT_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage1_v3")
SAVE_ROOT = Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage2_v3")
STORM_DRAW_TABLE_PATH = Path("/mnt/team/rapidresponse/pub/tropical-storms/storm_draw_table.csv")
# Causes in storm_draw_table.csv; each storm draw has one RR sample per cause.
RELATIVE_RISKS = ["indirect_resp_draw", "indirect_cvd_draw"]
N_DRAWS = 100  # stage 1 draws per combo; each output is their mean

# Fixed impact duration per storm-affected pixel (days), expressed as a
# year-fraction for the PAF formula: PAF = (RR - 1) / RR * (days / 365).
# Previously materialized as a per-storm raster filled with 20.0 inside the
# storm footprint, resampled to the basin template, and indexed by mask;
# the mask reduces to `rr > 0` and the value is constant, so we just use
# the precomputed fraction directly.
IMPACT_DAYS_FRACTION = 20.0 / 365.0

class StormMeta(NamedTuple):  # type: ignore
    storm_path: Path
    start_year: int
    end_year: int
    storm_id: str

def iter_storms_metadata(draw_store: Path) -> list[StormMeta]:
    """Return a list of StormMeta for each storm in the draw, without loading full data."""
    if not draw_store.exists():
        raise FileNotFoundError(f"Draw store not found: {draw_store}")

    storm_paths = sorted(
        p for p in draw_store.iterdir() if p.is_dir() and p.name.startswith("storm_")
    )

    storms_meta = []
    for storm_path in storm_paths:
        # The dates are attributes of the storm group: read its zarr.json (one
        # file) instead of xr.open_zarr, which also reads every array's
        # metadata and the lat/lon chunks.
        with open(storm_path / "zarr.json") as f:
            attrs = json.load(f).get("attributes", {})
        start_year = pd.to_datetime(attrs["start_date"]).year
        end_year = pd.to_datetime(attrs["end_date"]).year
        storm_id = attrs.get("storm_id", storm_path.name)
        storms_meta.append(StormMeta(storm_path, start_year, end_year, storm_id))
    return storms_meta

def map_storms_to_years(storms_meta: list[StormMeta], years: list[int]):
    storms_by_year = {year: [] for year in years}
    for storm in storms_meta:
        for year in range(storm.start_year, storm.end_year + 1):
            if year in storms_by_year:
                storms_by_year[year].append(storm.storm_path)
    return storms_by_year

    
##########################################
#          Helper Functions              #
##########################################

def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.assign_coords(
        lon=(((ds.lon + 180) % 360) - 180)
    ).sortby("lon")

    return ds


def ensure_min_grid(
    da: "xr.DataArray | xr.Dataset",
    buffer_deg: float = 0.1,
) -> "xr.DataArray | xr.Dataset":
    """
    Pad a single-pixel lat/lon dimension with NaN neighbors so rasterization
    sees at least a 3-cell axis. Works for both DataArray and Dataset.
    """
    if da.lat.size == 1:
        c = float(da.lat.values[0])
        da = da.reindex(lat=[c-buffer_deg, c, c+buffer_deg], fill_value=np.nan)

    if da.lon.size == 1:
        c = float(da.lon.values[0])
        da = da.reindex(lon=[c-buffer_deg, c, c+buffer_deg], fill_value=np.nan)

    return da


def to_raster(
    ds: xr.DataArray,
    no_data_value: float | int,
    lat_col: str = "lat",
    lon_col: str = "lon",
    crs: str = "EPSG:4326",
    expected_res: float | None = None,
) -> rt.RasterArray:
    lat, lon = ds[lat_col].data, ds[lon_col].data

    dlat = (lat[1:] - lat[:-1]).mean()
    dlon = (lon[1:] - lon[:-1]).mean()

    # The affine transform assumes a regular grid at the stage 1 resolution.
    # Stage 1 outputs written before the grid fix are evenly spaced at the
    # WRONG step, so check the step itself, not just uniformity.
    if expected_res is not None:
        target_lat, target_lon = expected_res, expected_res
    else:
        target_lat, target_lon = abs(dlat), abs(dlon)
    if not (
        np.allclose(np.abs(np.diff(lat)), target_lat, atol=1e-6)
        and np.allclose(np.abs(np.diff(lon)), target_lon, atol=1e-6)
    ):
        raise ValueError(
            f"to_raster requires coords uniformly spaced at "
            f"{expected_res if expected_res is not None else 'a constant step'}; "
            f"got lat steps {np.unique(np.round(np.diff(lat), 6))}, "
            f"lon steps {np.unique(np.round(np.diff(lon), 6))}"
        )

    # 🔑 detect latitude direction
    lat_increasing = lat[1] > lat[0]

    if lat_increasing:
        # south → north → flip required
        data = ds.data[::-1]
        y_origin = lat[-1]
    else:
        # already north → south → no flip
        data = ds.data
        y_origin = lat[0]

    transform = Affine(
        a=dlon,
        b=0.0,
        c=lon[0],
        d=0.0,
        e=-abs(dlat),
        f=y_origin,
    )

    return rt.RasterArray(
        data=data,
        transform=transform,
        crs=crs,
        no_data_value=no_data_value,
    )

def knots_to_ms(knots):
    """
    Convert wind speed from knots to meters per second.
    
    Parameters:
    -----------
    knots : float, array-like, or xarray.DataArray
        Wind speed in knots
        
    Returns:
    --------
    float, array-like, or xarray.DataArray
        Wind speed in meters per second
        
    Notes:
    ------
    Conversion factor: 1 knot = 0.514444 m/s
    """
    return knots * 0.514444

class RRInterpolator(NamedTuple):
    """
    Pre-built relative-risk interpolation context, shared across all storms
    in a single draw. Built once at the top of `process_single_draw` so
    every storm in that draw reuses the same `interp1d` callable and bounds
    instead of rebuilding them.
    """
    interp: object       # interp1d callable over windspeed_ms -> RR
    min_ms: float        # lowest windspeed in the RR table (m/s)
    max_ms: float        # highest windspeed in the RR table (m/s)
    max_rr: float        # RR value at the highest windspeed (used as a cap)
    sample_name: str     # column name in rr_samples_df, kept for metadata


def build_rr_interpolator(
    rr_samples_df: pd.DataFrame,
    sample_name: str,
) -> RRInterpolator:
    """Construct the per-draw RR interpolation context."""
    windspeed_ms = knots_to_ms(rr_samples_df["windspeed"].values)
    rr_values = rr_samples_df[sample_name].values

    interp = interp1d(
        windspeed_ms,
        rr_values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    return RRInterpolator(
        interp=interp,
        min_ms=float(windspeed_ms.min()),
        max_ms=float(windspeed_ms.max()),
        max_rr=float(rr_values[np.argmax(windspeed_ms)]),
        sample_name=sample_name,
    )


def rr_from_windspeed(windspeed_ms: np.ndarray, rr_ctx: RRInterpolator) -> np.ndarray:
    """
    Relative risk for windspeeds (m/s) from a pre-built RR interpolation
    context. Pixels outside the RR table's windspeed range get 0 (below min)
    or the table's max RR (above max); NaN windspeeds get 0. Same dtype as the
    input.
    """
    rr = np.zeros_like(windspeed_ms)

    above_max_mask = windspeed_ms > rr_ctx.max_ms
    interpolation_mask = (windspeed_ms >= rr_ctx.min_ms) & (windspeed_ms <= rr_ctx.max_ms)

    if np.any(above_max_mask):
        rr[above_max_mask] = rr_ctx.max_rr

    if np.any(interpolation_mask):
        rr[interpolation_mask] = rr_ctx.interp(windspeed_ms[interpolation_mask])

    return rr

def generate_basin_template_raster(basin, res=0.1, buffer_deg=5.0):
    basin_bounds = {
        'EP': ['180E', '0N', '290E', '60N'],
        'NA': ['260E', '0N', '360E', '60N'],
        'NI': ['30E',  '0N', '100E', '50N'],
        'SI': ['20E',  '45S', '100E', '0S'],
        'AU': ['100E', '45S', '180E', '0S'],
        'SP': ['180E', '45S', '250E', '0S'],
        'WP': ['100E', '0N', '180E', '60N'],
    }

    def parse_coord(c):
        match = re.match(r"([0-9\.]+)([ENWS])", c)
        val, hemi = match.groups()
        val = float(val)
        if hemi == 'S': val = -val
        if hemi == 'W': val = 360 - val
        return val

    lon_min, lat_min, lon_max, lat_max = [parse_coord(c) for c in basin_bounds[basin]]

    # Apply buffer
    lon_min -= buffer_deg
    lon_max += buffer_deg
    lat_min -= buffer_deg
    lat_max += buffer_deg

    # --- Normalize NA basin to -180 → 180 ---
    if basin == "NA":
        lon_min = ((lon_min + 180) % 360) - 180
        lon_max = ((lon_max + 180) % 360) - 180

        # ensure monotonic bounds
        if lon_max <= lon_min:
            lon_max += 360


    # Number of rows/cols
    n_cols = int(np.ceil((lon_max - lon_min) / res))
    n_rows = int(np.ceil((lat_max - lat_min) / res))

    # Create empty data array
    data = np.zeros((n_rows, n_cols), dtype=np.float32)

    # Create affine transform: from array index (col,row) to geographic coords
    # Affine: (scale_x, 0, x_min, 0, scale_y, y_max)
    # scale_y is negative because row index increases downward
    transform = Affine(res, 0, lon_min, 0, -res, lat_max)

    # Wrap as RasterArray
    raster = RasterArray(data=data,
                         transform=transform,
                         crs="EPSG:4326",
                         no_data_value=np.nan
                         )
    return raster


##########################################
#             Read in Data               #
##########################################

def get_draw_zarr_path(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    draw: int,
    metric: str,
) -> Path | None:
    """
    Locate draw-level storm Zarr store produced by Stage 1.
    Returns None if the draw produced no storms.
    """
    start_year, end_year = batch_year.split("-")
    draw_text = "" if draw == 0 else f"_e{draw - 1}"

    metrics_allowed = ["intensity", "exposure_hours", "days_impact"]
    if metric not in metrics_allowed:
        raise ValueError(f"Invalid metric: {metric}. Allowed: {metrics_allowed}")
    
    draw_store = (
        ROOT_PATH
        / source_id
        / variant_label
        / experiment_id
        / batch_year
        / basin
        / metric
        / f"{metric}_{basin}_{source_id}_{experiment_id}_{variant_label}_{start_year}01_{end_year}12{draw_text}.zarr"
    )

    if not draw_store.exists():
        return None   # ← key change - return none for 0 impact processing

    return draw_store

def load_relative_risk_df(relative_risk: str,root: Path = Path("/mnt/team/rapidresponse/pub/tropical-storms/data/relative_risk_samples/")):

    if relative_risk == "indirect_resp_draw":
        relative_risk_df = pd.read_csv(root / f"rd_rr_samples.csv")
    elif relative_risk == "indirect_cvd_draw":
        relative_risk_df = pd.read_csv(root / f"cvd_rr_samples.csv")
    
    return relative_risk_df


##########################################
#          Save Yearly-Basin Raster      #
##########################################

def save_raster(
    raster_data: np.ndarray,
    template_raster: rt.RasterArray,
    storm_draw: str,
    source_id: str,
    variant_label: str,
    sample_name: str,
    relative_risk: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    year: int,
    metric: str,  # "raw_paf" or "raw_rr"
    save_root: Path = SAVE_ROOT,
    max_retries: int = 3,
    retry_delay: float = 1.0,
):
    """
    Generic function to save raster data as GeoTIFF, with retries.

    Parameters
    ----------
    raster_data : np.ndarray
        2D array to save.
    template_raster : rt.RasterArray
        Raster template to copy CRS, transform, and no_data_value.
    metric : str
        Metric name, e.g., "raw_paf" or "raw_rr".

    """
    raster_data = raster_data.astype(np.float32)

    save_dir = save_root / storm_draw / source_id / variant_label / experiment_id / batch_year / str(year) / basin / metric
    save_dir.mkdir(parents=True, exist_ok=True)

    start_year, end_year = batch_year.split("-")
    filename = (
        f"draw_mean_{metric}_{storm_draw}_{relative_risk}_{sample_name}_{basin}_{source_id}_"
        f"{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}.tif"
    )
    save_path = save_dir / filename

    raster_array = rt.RasterArray(
        data=raster_data,
        transform=template_raster.transform,
        crs=template_raster.crs,
        no_data_value=template_raster.no_data_value,
    )

    # Retry loop for robust saving
    for attempt in range(max_retries):
        try:
            raster_array.to_file(
                save_path,
                driver="GTiff",
                compress="deflate",
                predictor=3,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            print(f"Saved {metric} raster as TIFF: {save_path}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Save failed for {save_path}, retrying in {retry_delay}s ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to save {save_path} after {max_retries} attempts") from e

    return save_path


def _is_valid_raster(path: Path) -> bool:
    try:
        _ = rt.load_raster(path)
        return True
    except Exception:
        return False   


def get_year_status(
    storm_draw: str,
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    relative_risk: str,
    sample_name: str,
    save_root: Path = SAVE_ROOT,
):
    start_year, end_year = batch_year.split("-")
    years = range(int(start_year), int(end_year) + 1)

    valid_years = []
    invalid_years = []

    for year in years:
        save_dir = (
            save_root / storm_draw / source_id / variant_label /
            experiment_id / batch_year / str(year) / basin / "raw_paf"
        )

        filename = (
            f"draw_mean_raw_paf_{storm_draw}_{relative_risk}_{sample_name}_{basin}_{source_id}_"
            f"{experiment_id}_{variant_label}_{start_year}01_{end_year}12_{year}.tif"
        )

        path = save_dir / filename

        if not path.exists():
            invalid_years.append(year)
            continue

        if path.stat().st_size < 1024 or not _is_valid_raster(path):
            path.unlink(missing_ok=True)
            invalid_years.append(year)
        else:
            valid_years.append(year)

    return valid_years, invalid_years
##########################################
#          Main Stage 2 Function         #
##########################################

class Stage2Job(NamedTuple):
    """One output series of a combo: a storm draw's RR sample for one cause."""
    storm_draw: str
    relative_risk: str
    sample_name: str
    years: tuple[int, ...]  # years whose raster is missing on disk


def build_jobs(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    save_root: Path = SAVE_ROOT,
) -> list[Stage2Job]:
    """Every (storm draw x cause) of this combo that still has a missing year.

    The storm draws are those of (source_id, variant_label) in
    storm_draw_table.csv; each has one RR sample per cause.
    """
    table = pd.read_csv(STORM_DRAW_TABLE_PATH)
    table = table[(table["source_id"] == source_id) & (table["variant_label"] == variant_label)]
    jobs = []
    for row in table.sort_values("storm_draw").itertuples(index=False):
        storm_draw = f"storm_draw_{row.storm_draw:04d}"
        for relative_risk in RELATIVE_RISKS:
            sample_name = getattr(row, relative_risk)
            _, missing_years = get_year_status(
                storm_draw=storm_draw,
                source_id=source_id,
                variant_label=variant_label,
                experiment_id=experiment_id,
                batch_year=batch_year,
                basin=basin,
                relative_risk=relative_risk,
                sample_name=sample_name,
                save_root=save_root,
            )
            if missing_years:
                jobs.append(Stage2Job(storm_draw, relative_risk, sample_name, tuple(missing_years)))
    return jobs


def add_storm_paf(
    year_sums: list[np.ndarray],
    storm_ds: xr.Dataset,
    rr_ctxs: list[RRInterpolator],
    template_raster,
) -> None:
    """Add one storm's raw PAF to each job's basin-grid yearly sum (in place).

    The storm's intensity is resampled to the basin grid once and each job's
    RR curve is applied there. RR is computed pixel by pixel and the resample
    is nearest-neighbour, so this equals resampling each job's RR raster
    (checked bit for bit on 267 storms x 6-12 RR samples in NA, EP and SP).
    """
    intensity = to_raster(
        ds=storm_ds["intensity"],
        no_data_value=np.nan,
        lat_col="lat",
        lon_col="lon",
        crs="EPSG:4326",
        expected_res=template_raster.transform.a,
    ).resample_to(target=template_raster, resampling="nearest")._ndarray

    # Work in the window that holds the storm's footprint.
    finite = np.isfinite(intensity)
    if not finite.any():
        return
    rows = np.flatnonzero(finite.any(axis=1))
    cols = np.flatnonzero(finite.any(axis=0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    windspeed = intensity[r0:r1, c0:c1]

    for total, rr_ctx in zip(year_sums, rr_ctxs):
        rr = rr_from_windspeed(windspeed, rr_ctx)
        # Pixels with a positive RR contribute; NaN windspeeds have RR 0.
        mask = rr > 0
        if mask.any():
            # Every masked pixel sees a constant 20-day impact, so the
            # year-fraction collapses to IMPACT_DAYS_FRACTION.
            total[r0:r1, c0:c1][mask] += (rr[mask] - 1) / rr[mask] * IMPACT_DAYS_FRACTION


def crop_nonzero(arr: np.ndarray) -> tuple[int, int, np.ndarray] | None:
    """(row0, col0, sub-array) covering every nonzero pixel, or None if all zero.

    Adding the sub-array at (row0, col0) gives exactly the same sum as adding
    the full array, since x + 0.0 == x.
    """
    rows = np.flatnonzero(arr.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(arr.any(axis=0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return int(r0), int(c0), arr[r0:r1, c0:c1].copy()


def process_single_draw(args):
    """
    Process one stage 1 draw for every job of the combo.

    Each storm is read once: it is loaded for the first year it covers and kept
    until its last year. For every job that still needs a year, the storm's
    raw PAF is added to that job's yearly sum, in the same storm order as the
    per-job code before (so each job's sums are unchanged).

    Returns (sums, failures): sums maps (job index, year) to a cropped
    (row0, col0, array) of that draw's yearly raw PAF (all-zero sums omitted);
    failures lists (year, storm_id) for storms that raised. A year with any
    failure must not be saved; its PAF would be missing that storm.
    """
    (
        source_id,
        variant_label,
        experiment_id,
        batch_year,
        basin,
        draw,
        jobs,
        template_raster,
        rr_tables,
        years,
    ) = args

    intensity_draw_store = get_draw_zarr_path(
        source_id=source_id,
        variant_label=variant_label,
        experiment_id=experiment_id,
        batch_year=batch_year,
        basin=basin,
        draw=draw,
        metric="intensity",
    )
    # No stage 1 store, or no storms: this draw adds zero everywhere.
    if intensity_draw_store is None or not any(intensity_draw_store.iterdir()):
        print(f"⚠️ Draw {draw} has no intensity data (basin {basin}, batch {batch_year}).")
        return {}, []

    # Build each job's RR interpolation once per draw; every storm reuses it.
    rr_ctxs = [build_rr_interpolator(rr_tables[job.relative_risk], job.sample_name) for job in jobs]

    storms_meta = iter_storms_metadata(intensity_draw_store)
    storms_by_year = map_storms_to_years(storms_meta, years)
    last_year = {path: year for year in years for path in storms_by_year[year]}

    loaded: dict[Path, xr.Dataset] = {}  # storms kept for a later year
    sums: dict[tuple[int, int], tuple[int, int, np.ndarray]] = {}
    failures: list[tuple[int, str]] = []

    for year in years:
        storm_paths_in_year = storms_by_year[year]
        active = [i for i, job in enumerate(jobs) if year in job.years]
        if not storm_paths_in_year or not active:
            continue

        year_sums = {i: np.zeros_like(template_raster._ndarray, dtype=np.float32) for i in active}

        for storm_path in storm_paths_in_year:
            storm_id = storm_path.name  # fallback identifier if open fails
            try:
                storm_ds = loaded.pop(storm_path, None)
                if storm_ds is None:
                    with open_storm(storm_path, chunks="auto") as raw:
                        storm_ds = ensure_min_grid(raw.load())
                    if basin == "NA":
                        storm_ds = normalize_dataset(storm_ds)
                storm_id = storm_ds.attrs.get("storm_id", storm_path.name)

                add_storm_paf(
                    [year_sums[i] for i in active], storm_ds, [rr_ctxs[i] for i in active], template_raster
                )

                if last_year[storm_path] > year:
                    loaded[storm_path] = storm_ds
            except Exception as e:
                print(
                    f"❌ Storm {storm_id} failed in draw {draw} "
                    f"({source_id}/{variant_label}/{experiment_id}/{batch_year}/"
                    f"{basin}, year {year}): {type(e).__name__}: {e}"
                )
                traceback.print_exc()
                failures.append((year, str(storm_id)))
                continue

        for i, arr in year_sums.items():
            cropped = crop_nonzero(arr)
            if cropped is not None:
                sums[(i, year)] = cropped
        del year_sums
        gc.collect()

    print(f"Completed draw {draw} for basin {basin}, batch {batch_year}")
    return sums, failures


def main(
    source_id: str,
    variant_label: str,
    experiment_id: str,
    batch_year: str,
    basin: str,
    num_cores: int,
    save_root: Path = SAVE_ROOT,
):
    """Write the draw-mean raw PAF rasters of every storm draw x cause of one combo.

    Each stage 1 storm of the combo is read once per draw and applied to every
    job's RR sample, instead of once per (storm draw, cause) task.
    """
    jobs = build_jobs(source_id, variant_label, experiment_id, batch_year, basin, save_root)
    if not jobs:
        print(f"✅ All outputs exist for {source_id}/{variant_label}/{experiment_id}/{batch_year}/{basin}. Skipping.")
        return None

    years = sorted({year for job in jobs for year in job.years})
    print(f"{len(jobs)} job(s) (storm draw x cause) to compute; years {years}")

    # Generate basin-wide template raster once
    template_raster = generate_basin_template_raster(basin, res=0.1)
    rr_tables = {rr: load_relative_risk_df(relative_risk=rr) for rr in {job.relative_risk for job in jobs}}

    # Sums across draws, per (job index, year); allocated on first contribution.
    cumulative_paf: dict[tuple[int, int], np.ndarray] = {}
    # (draw, year, storm_id) for every storm that raised in any draw
    failed_storms: list[tuple[int, int, str]] = []

    draws = list(range(N_DRAWS))
    for batch_start in range(0, len(draws), num_cores):
        batch_draws = draws[batch_start: batch_start + num_cores]
        print(f"Starting batch {batch_draws}")

        draw_args = [
            (source_id, variant_label, experiment_id, batch_year, basin,
             draw, jobs, template_raster, rr_tables, years)
            for draw in batch_draws
        ]
        batch_results = run_parallel(
            runner=process_single_draw,
            arg_list=draw_args,
            num_cores=num_cores,
        )
        print(f"Parallel job stage done for draw_batch: {batch_draws}")

        # Add draws in draw order, as before, so each job's float sums are unchanged.
        for draw, (draw_sums, draw_failures) in zip(batch_draws, batch_results):
            failed_storms.extend((draw, year, sid) for year, sid in draw_failures)
            for key, (r0, c0, sub) in draw_sums.items():
                total = cumulative_paf.get(key)
                if total is None:
                    total = cumulative_paf[key] = np.zeros_like(template_raster._ndarray, dtype=np.float32)
                total[r0:r0 + sub.shape[0], c0:c0 + sub.shape[1]] += sub

    # Years with a failed storm are not saved for any job, so they stay
    # missing and are recomputed on the retry; the other years are saved now.
    failed_years = sorted({year for _, year, _ in failed_storms})

    saved_paths: list[Path] = []
    for i, job in enumerate(jobs):
        for year in job.years:
            if year in failed_years:
                continue
            total = cumulative_paf.get((i, year))
            if total is None:
                total = np.zeros_like(template_raster._ndarray, dtype=np.float32)
            save_path = save_raster(
                raster_data=total / N_DRAWS,
                template_raster=template_raster,
                storm_draw=job.storm_draw,
                source_id=source_id,
                variant_label=variant_label,
                sample_name=job.sample_name,
                relative_risk=job.relative_risk,
                experiment_id=experiment_id,
                batch_year=batch_year,
                basin=basin,
                year=year,
                metric="raw_paf",
                save_root=save_root,
            )
            saved_paths.append(save_path)

    # Fix permissions on the files we wrote (and their immediate parents so
    # the group can traverse). No recursive walk needed — we know exactly
    # which paths exist.
    for path in saved_paths:
        try:
            os.chmod(path, 0o664)
            os.chmod(path.parent, 0o775)
        except Exception as e:
            print(f"⚠️ Could not set permissions for {path}: {e}")

    if failed_years:
        # Non-zero exit so Jobmon marks the task failed and retries it; the
        # retry recomputes only the unsaved (failed) years.
        print(
            f"❌ {len(failed_storms)} storm(s) failed; years {failed_years} not saved. "
            f"(draw, year, storm): {failed_storms[:20]}"
        )
        raise SystemExit(1)

    print(f"Completed {len(jobs)} job(s) for basin {basin}, batch {batch_year}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIMADA stage 2: draw-mean raw PAF rasters for every storm draw x cause of one combo"
    )
    parser.add_argument("--source_id", type=str, required=True, help="Source Id")
    parser.add_argument("--variant_label", type=str, required=True, help="Variant Label")
    parser.add_argument("--experiment_id", type=str, required=True, help="Experiment Id")
    parser.add_argument("--batch_year", type=str, required=True, help="Batch Year")
    parser.add_argument("--basin", type=str, required=True, help="Basin")
    parser.add_argument("--num_cores", type=int, required=True, help="Number of cores to use for parallel processing")
    args = parser.parse_args()

    main(
        source_id=args.source_id,
        variant_label=args.variant_label,
        experiment_id=args.experiment_id,
        batch_year=args.batch_year,
        basin=args.basin,
        num_cores=args.num_cores,
    )
