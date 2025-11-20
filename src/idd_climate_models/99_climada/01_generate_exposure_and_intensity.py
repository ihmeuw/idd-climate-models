from climada.hazard import TCTracks, TropCyclone, Centroids # type: ignore
import xarray as xr  # type: ignore
import numpy as np # type: ignore
import re
from pathlib import Path
import rasterra as rt # type: ignore
import os
import datetime as dt

def read_custom_tracks(
    root_path: Path,
    model: str,
    variant: str,
    scenario: str,
    batch_year: str,
    basin: str,
    draw: int,
) -> TCTracks:
    """
    Read tc risk model outputs for CLIMADA processing.
    """
    start_year = batch_year.split("-")[0]
    end_year = batch_year.split("-")[1] 

    draw_text = f'_e{draw - 1}' if draw > 0 else ''
    track_file = f'tracks_{basin}_{model}_{scenario}_{variant}_{start_year}01_{end_year}12{draw_text}.nc'

    file_path = root_path / model / variant / scenario / batch_year / basin / track_file

    ds_custom = xr.open_dataset(file_path)

    return ds_custom

def prepare_minimal_tctracks_from_custom(ds_custom):
    """
    Convert custom synthetic tracks with time in seconds and per-track
    year/month info into a TCTracks object compatible with
    TropCyclone.from_tracks().
    """

    storms = []
    n_trk = ds_custom.sizes["n_trk"]
    raw_time = ds_custom["time"].values  # seconds since start

    # Compute time step in hours
    if len(raw_time) > 1:
        dt_hours = np.diff(raw_time).mean() / 3600.0
    else:
        dt_hours = 1.0

    # Prepare container for datetime arrays
    tracks_time = []

    # ----------------------------------------------------
    # First loop: compute datetime arrays
    # ----------------------------------------------------
    for i in range(n_trk):

        start_year = int(ds_custom["tc_years"][i].item())
        start_month = int(ds_custom["tc_month"][i].item())

        start_date = np.datetime64(f"{start_year:04d}-{start_month:02d}-01T00:00", "s")

        if raw_time.ndim == 2:
            time_seconds = raw_time[i].astype("timedelta64[s]")
        else:
            time_seconds = raw_time.astype("timedelta64[s]")

        time_dt = start_date + time_seconds
        time_dt = time_dt.astype("datetime64[h]")

        tracks_time.append(time_dt)

    # ----------------------------------------------------
    # Second loop: build each storm dataset
    # ----------------------------------------------------
    for i in range(n_trk):

        lon = ds_custom["lon_trks"][i].values
        lat = ds_custom["lat_trks"][i].values
        vmax = ds_custom["vmax_trks"][i].values

        
        cp = ds_custom["m_trks"][i].values
        env = cp + 20.0

        time_dt = tracks_time[i]

        # Trim NaNs
        valid_idx = np.isfinite(lon) & np.isfinite(lat)
        lon = lon[valid_idx]
        lat = lat[valid_idx]
        vmax = vmax[valid_idx]

        # Needs changes
        cp = cp[valid_idx]
        env = env[valid_idx]


        time_dt = time_dt[valid_idx]
        n_time = len(lon)

        if n_time == 0:
            continue  # skip empty storms safely

        lon = ((lon + 180) % 360) - 180
        lat = np.clip(lat, -90, 90)

        vmax_max = vmax.max().item()
        category = (
            0 if vmax_max < 33 else
            1 if vmax_max < 43 else
            2 if vmax_max < 50 else
            3 if vmax_max < 58 else
            4 if vmax_max < 70 else
            5
        )

        ds = xr.Dataset(
            coords={"time": time_dt},
            data_vars={
                "lon": (("time",), lon),
                "lat": (("time",), lat),
                "max_sustained_wind": (("time",), vmax),
                "central_pressure": (("time",), cp),
                "environmental_pressure": (("time",), env),
                "basin": (("time",), np.repeat(ds_custom["tc_basins"][i].item(), n_time)),
                "radius_max_wind": (("time",), np.zeros(n_time)),
                "radius_oci": (("time",), np.zeros(n_time)),
                "time_step": (("time",), np.full(n_time, dt_hours)),
            },
            attrs={
                "name": f"CUSTOM_{i}",
                "sid": int(i),
                "id_no": int(i),
                "category": category,
                "orig_event_flag": True,
                "data_provider": "custom",
                "max_sustained_wind_unit": "kn",
                "central_pressure_unit": "mb",
            }
        )

        storms.append(ds)

    return TCTracks(data=storms)

    
def normalize_lon(lon: float) -> float:
    """Normalize longitude to [-180, 180] range."""
    lon = ((lon + 180) % 360) - 180
    return lon

def generate_basin_centroids(basin: str, res: float = 0.1) -> "Centroids":
    """
    Generate Centroids for a specific tropical cyclone basin.
    """

    # Dictionary of basin bounds
    basin_bounds = {
        'EP': ['180E', '0N', '290E', '60N'],
        'NA': ['260E', '0N', '360E', '60N'],
        'NI': ['30E', '0N', '100E', '50N'],
        'SI': ['20E', '45S', '100E', '0S'],
        'AU': ['100E', '45S', '180E', '0S'],
        'SP': ['180E', '45S', '250E', '0S'], # Original SA - possible mismatch
        'WP': ['100E', '0N', '180E', '60N'],
        'GL': ['0E', '90S', '360E', '90N']
    }

    if basin not in basin_bounds:
        raise ValueError(f"Basin '{basin}' not recognized. Available: {list(basin_bounds.keys())}")

    def parse_coord(coord_str: str) -> float:
        """Convert coordinate string with direction to float degrees."""
        match = re.match(r"([0-9\.]+)([ENWS])", coord_str)
        if not match:
            raise ValueError(f"Invalid coordinate string: {coord_str}")
        val, dir_ = match.groups()
        val = float(val)
        if dir_ in ['W', 'S']:
            val = -val
        return val

    lon_min, lat_min, lon_max, lat_max = [parse_coord(c) for c in basin_bounds[basin]]

    # Normalize longitudes to [-180, 180]
    lon_min = normalize_lon(lon_min)
    lon_max = normalize_lon(lon_max)

    # Expand upper bounds by resolution to include last grid cell
    lon_max += res
    lat_max += res

    # Create Centroids for the basin
    centroids = Centroids.from_pnt_bounds((lon_min, lat_min, lon_max, lat_max), res=res)

    return centroids

def split_tracks_by_category(tc_tracks: TCTracks) -> dict[str, TCTracks]:
    """
    Split a TCTracks object into multiple TCTracks subsets based on the 'category' attribute.
    """
    category_severity_map = {
        "severe": [4, 5],
        "moderate": [1, 2, 3],
        "tropical": [0],
    }

    results = {}

    # Extract track list
    tracks = tc_tracks.data

    for label, cats in category_severity_map.items():
        # Select only those tracks with matching category
        filtered = [tr for tr in tracks if getattr(tr, "category", None) in cats]

        if not filtered:
            print(f"⚠️ No events found for '{label}' categories {cats}")
            continue

        # Create a new TCTracks object containing only these storms
        new_tc = TCTracks()
        new_tc.data = filtered
        results[label] = new_tc

    return results

def generate_severity_tracks(tc_tracks: TCTracks) -> tuple[TCTracks | None, TCTracks | None, TCTracks | None]:
    """
    Generate separate TCTracks objects for each severity category.
    """
    split_tracks = split_tracks_by_category(tc_tracks)
    tropical_tracks = split_tracks.get("tropical")
    moderate_tracks = split_tracks.get("moderate")
    severe_tracks = split_tracks.get("severe")

    return tropical_tracks, moderate_tracks, severe_tracks

def generate_hazard_per_track(tc_tracks: TCTracks, centroids: Centroids) -> TropCyclone:
    """
    Generate CLIMADA TropCyclone hazard object from TCTracks and Centroids.
    """

    haz = TropCyclone.from_tracks(tc_tracks, centroids=centroids, store_windfields=True)

    return haz

def generate_speed_per_storm(haz: TropCyclone, centroids: Centroids, tc_tracks: TCTracks) -> list[xr.DataArray]:
    """
    Generate per-storm wind speed DataArrays for a list of tropical cyclones.
    """

    lat = np.unique(centroids.coord[:, 0])
    lon = np.unique(centroids.coord[:, 1])
    lat = np.sort(lat)

    storm_list_speed = []

    for i, event in enumerate(tc_tracks.data):
        storm_name = event.name  # <-- Grab the storm name directly
        times = event.time  # array of timesteps
        wf = haz.windfields[i].toarray()  # shape: (time, n_centroids, 2)

        n_time = len(times)
        n_lat = len(lat)
        n_lon = len(lon)

        try:
            wf_reshaped = wf.reshape(n_time, n_lat, n_lon, 2)
        except ValueError:
            print(f"⚠️ Skipping storm {storm_name} due to shape mismatch")
            continue

        # Preserve timestep if it exists
        timestep = getattr(event, "time_step", None)
        coords = {"time": times, "lat": np.flip(lat), "lon": lon, "dir": ["u", "v"]}
        if timestep is not None:
            coords["time_step"] = ("time", np.array(timestep))


        da = xr.DataArray(
            wf_reshaped,
            coords=coords,
            dims=["time", "lat", "lon", "dir"],
            name=f"{storm_name}_windfields"
        )

        # Compute wind speed
        da_speed = np.sqrt(da.isel(dir=0)**2 + da.isel(dir=1)**2)
        da_speed.name = storm_name  # <-- Name the DataArray after the storm
        da_speed.attrs.update({
            "description": f"Storm {storm_name} wind speed",
            "units": "m/s",
            "storm_name": storm_name,
            "category": getattr(event, "category", None)
        })

        storm_list_speed.append(da_speed)

        # Free memory
        del wf, wf_reshaped, da, da_speed

    return storm_list_speed


def generate_intensity_per_storm(haz: TropCyclone, centroids: Centroids, tc_tracks: TCTracks) -> list[xr.DataArray]:
    # Extract lat/lon from centroids
    lat = np.unique(centroids.coord[:, 0])
    lon = np.unique(centroids.coord[:, 1])
    lat = np.sort(lat)

    storm_list_intensity = []

    # Loop over storms
    for i, event in enumerate(tc_tracks.data):

        storm_name = event.name
        storm_start_date = event.time.min().astype("datetime64[D]").item().isoformat()
        storm_end_date = event.time.max().astype("datetime64[D]").item().isoformat()
        storm_basin = np.unique(event.basin.values)[0]  # single string

        # --- Extract 1D intensity for this event ---
        try:
            intensity_1d = haz.intensity.toarray()[i, :]  # shape: (n_cells,)
        except Exception as e:
            print(f"⚠️ Failed to extract intensity for storm {storm_name}: {e}")
            continue

        # Expected shape = (n_lat * n_lon)
        n_lat = len(lat)
        n_lon = len(lon)

        if intensity_1d.size != (n_lat * n_lon):
            print(f"⚠️ Skipping storm {storm_name} due to shape mismatch:")
            print(f"   expected: {n_lat*n_lon}, got: {intensity_1d.size}")
            continue

        # Reshape into lat/lon grid
        intensity_2d = intensity_1d.reshape(n_lat, n_lon)

        # Flip latitude for correct map orientation
        intensity_2d = np.flip(intensity_2d, axis=0)

        # Build DataArray
        da_intensity = xr.DataArray(
            intensity_2d,
            coords={"lat": np.flip(lat), "lon": lon},
            dims=["lat", "lon"],
            name=f"{storm_name}_intensity",
            attrs={
                "description": f"Storm {storm_name} intensity (max 3-sec gust wind speed)",
                "units": "m/s",
                "storm_name": storm_name,
                "category": getattr(event, "category", None),
                "start_date": storm_start_date,
                "end_date": storm_end_date,
                "basin": storm_basin,
            },
        )

        storm_list_intensity.append(da_intensity)

        # Free memory
        del intensity_1d, intensity_2d, da_intensity

    return storm_list_intensity


def compute_daily_exposure_from_storm_list(
    storm_list: list[xr.DataArray],
    wind_threshold: float = 18.0,
) -> dict[str, xr.DataArray]:
    """
    Compute daily exposure rasters (hours) from a list of per-storm wind speed DataArrays.
    Uses the per-timestep duration from each storm's 'time_step' data variable.
    Mimics the reference code by using Xarray broadcasting and resample.
    """

    daily_exposures = {}

    for storm_da in storm_list:
        if "time_step" not in storm_da.coords:
            raise ValueError(f"'time_step' coordinate not found in storm {storm_da.name}")

        # --- Compute mask: 1 if speed >= threshold, else 0 ---
        mask = xr.where(storm_da >= wind_threshold, 1.0, 0.0)

        # --- Multiply mask by timestep duration ---
        exposure = mask * storm_da["time_step"]

        # --- Resample by day and sum to get daily exposure ---
        daily = exposure.resample(time="D").sum(dim="time")

        # --- Add each day to output dict ---
        for day in daily.time.values:
            day_str = str(np.datetime64(day, "D"))
            daily_exposures[day_str] = daily.sel(time=day)

    return daily_exposures


def generate_exposure_for_severity_from_storms(
    tc_tracks: TCTracks,
    basin: str,
    resolution: float,
) -> dict[str, dict[str, xr.DataArray]]:
    """
    Generate daily exposure rasters from per-storm wind speed datasets for each severity category.

    Handles cases where no tracks exist for a given severity by inserting an empty dict.
    """

    # Step 1: Generate centroids for the basin
    centroids = generate_basin_centroids(basin=basin, res=resolution)

    # Step 2: Generate severity-specific tracks
    tropical_tracks, moderate_tracks, severe_tracks = generate_severity_tracks(tc_tracks)
    severity_map = {
        "tropical": tropical_tracks,
        "moderate": moderate_tracks,
        "severe": severe_tracks,
    }

    severity_daily_exposures = {}
    severity_storm_intensities = {}

    # Step 3: Iterate over severity levels
    for severity_name, severity_track in severity_map.items():

        # Handle None or empty tracks
        if severity_track is None or len(severity_track.data) == 0:
            print(f"⚠️ No tracks found for {severity_name} — inserting empty entry.")
            # severity_daily_exposures[severity_name] = {}  # empty dict
            continue

        print(f"🌀 Processing {severity_name.upper()} cyclones ({len(severity_track.data)} tracks)...")

        # Step 3.2: Generate hazards
        haz = generate_hazard_per_track(severity_track, centroids)

        # Step 3.2: Generate per-storm wind speed DataArrays
        storm_list = generate_speed_per_storm(haz, centroids, severity_track)

        # Step 3.3: Generate per-storm intensity DataArrays
        storm_list_intensity = generate_intensity_per_storm(haz, centroids, severity_track)

        # Step 3.4: Compute daily exposure rasters using the storm list
        daily_exposures = compute_daily_exposure_from_storm_list(storm_list)

        severity_daily_exposures[severity_name] = daily_exposures
        severity_storm_intensities[severity_name] = storm_list_intensity

    return severity_daily_exposures, severity_storm_intensities


def save_daily_exposure_rasters(
        severity_daily_exposures: dict, 
        output_dir: Path,
        model: str,
        variant: str,
        scenario: str,
        basin: str,
        draw: int | str,
        ):
    """
    Save daily exposure rasters for each severity level to GeoTIFF files using rioxarray
    """

    save_dir = output_dir / model / variant / scenario / basin / str(draw)

    for severity, day_dict in severity_daily_exposures.items():
        severity_dir = save_dir / "daily_exposure" / severity
        severity_dir.mkdir(exist_ok=True, parents=True)

        for day_str, da in day_dict.items():
            da = da.astype(np.float32)

            # Skip empty rasters
            if da.sum().item() == 0:
                continue

            # Assign proper CRS if missing
            if da.rio.crs is None:
                da = da.rio.write_crs("EPSG:4326", inplace=True)

            # Ensure proper names for rioxarray (lat->y, lon->x)
            da_rio = da.rename({"lat": "y", "lon": "x"})

            # Save GeoTIFF
            save_name = f"{day_str}_exposure_hours"
            out_path = severity_dir / f"{save_name}.tif"
            da_rio.rio.to_raster(out_path)

            # Optional: set file permissions
            os.chmod(out_path, 0o775)
            print(f"💾 Saved: {out_path}")

def save_intensity_per_storm_rasters(
        severity_storm_intensities: dict,
        output_dir: Path,
        model: str,
        variant: str,
        scenario: str,
        basin: str,
        draw: int | str,
        ):
    """
    Save per-storm intensity rasters for each severity level to GeoTIFF files using rioxarray
    """
    save_dir = output_dir / model / variant / scenario / basin / str(draw)

    for severity, storm_list in severity_storm_intensities.items():
        severity_dir = save_dir / "intensity" / severity
        severity_dir.mkdir(exist_ok=True, parents=True)

        for da in storm_list:
            da = da.astype(np.float32)

            # Skip empty rasters
            if da.sum().item() == 0:
                continue

            # Assign proper CRS if missing
            if da.rio.crs is None:
                da = da.rio.write_crs("EPSG:4326", inplace=True)

            # Ensure proper names for rioxarray (lat->y, lon->x)
            da_rio = da.rename({"lat": "y", "lon": "x"})

            # Save GeoTIFF
            start_date = da.attrs.get("start_date").split("T")[0]
            end_date = da.attrs.get("end_date").split("T")[0]
            basin = da.attrs.get("basin")
            storm_name = f"{basin}_{start_date}_{end_date}"
            
            out_path = severity_dir / f"{storm_name}_intensity.tif"
            da_rio.rio.to_raster(out_path)

            # Optional: set file permissions
            os.chmod(out_path, 0o775)
            print(f"💾 Saved intensity raster: {out_path}")

def generate_and_save_daily_exposure_rasters(
    root_path: Path,
    model: str,
    variant: str,
    scenario: str,
    batch_year: str,
    basin: str,
    draw: int | str,
    resolution: float,
    output_dir: Path,
):
    """
    Generate and save daily exposure rasters for each severity category.
    """

    # Step 1: Read in custom tracks from tc_risk model
    ds_custom = read_custom_tracks(
        root_path,
        model,
        variant,
        scenario,
        batch_year,
        basin,
        draw,
    )

    # Step 2: Prepare TCTracks object
    tc_tracks = prepare_minimal_tctracks_from_custom(ds_custom)


    # Generate daily exposure rasters
    severity_daily_exposures, severity_storm_intensities = generate_exposure_for_severity_from_storms(
        tc_tracks,
        basin,
        resolution,
    )

    # Save rasters to disk
    save_daily_exposure_rasters(
        severity_daily_exposures, 
        output_dir,
        model,
        variant,
        scenario,
        basin,
        draw,
    )

    save_intensity_per_storm_rasters(
        severity_storm_intensities,
        output_dir,
        model,
        variant,
        scenario,
        basin,
        draw,
    )

