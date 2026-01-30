import argparse
import sys
import os
import subprocess
from pathlib import Path
from typing import Any, Optional
import xarray as xr
import numpy as np

import idd_climate_models.constants as rfc
from idd_climate_models.add_tc_params import add_parameters_to_dataset
from idd_climate_models.climate_file_functions import get_dir, get_track_path

def create_track_specific_files(ds_multi, args):
    """
    Splits the multi-track dataset, applies corrections, trims NaNs, and saves 
    each individual track as a NetCDF file with a descriptive name.
    
    Args:
        ds_multi (xr.Dataset): The input multi-track dataset.
        args (Any): The arguments object containing output directory information.
    """
    output_directory = get_dir(args, source=False)

    n_trk = ds_multi.sizes["n_trk"]
    raw_time = ds_multi["time"].values  # 1D array of seconds since start

    # Compute time step in hours (constant for all tracks)
    dt_hours = np.diff(raw_time).mean() / 3600.0 if len(raw_time) > 1 else 1.0

    print(f"Processing {n_trk} tracks and saving to {output_directory}...")

    # Loop through each track
    for i in range(n_trk):
        # 1. Get Track Metadata and Data
        start_year = int(ds_multi["tc_years"][i].item())
        start_month = int(ds_multi["tc_month"][i].item())
        basin = ds_multi["tc_basins"][i].item()
        
        # Select 1D arrays for the current track
        lon = ds_multi["lon_trks"][i].values
        lat = ds_multi["lat_trks"][i].values
        cp = ds_multi["central_pressure"][i].values
        env = ds_multi["environmental_pressure"][i].values
        vmax = ds_multi["vmax_trks"][i].values

        # 2. Calculate Absolute Time
        start_date = np.datetime64(f"{start_year:04d}-{start_month:02d}-01T00:00", "s")
        time_seconds = raw_time.astype("timedelta64[s]")
        time_dt = start_date + time_seconds
        time_dt = time_dt.astype("datetime64[h]")
        
        # 3. Apply Corrections (Lon/Lat)
        lon = ((lon + 180) % 360) - 180
        lat = np.clip(lat, -90, 90)

        # 4. CRITICAL STEP: Trim NaNs and Calculate Category
        valid_idx = np.isfinite(lon) & np.isfinite(lat)
        
        valid_lon = lon[valid_idx]
        valid_lat = lat[valid_idx]
        valid_vmax = vmax[valid_idx]
        valid_cp = cp[valid_idx]
        valid_env = env[valid_idx]
        valid_time_dt = time_dt[valid_idx]
        n_time = len(valid_lon)
        
        if n_time == 0:
            continue  # Skip empty storms

        vmax_max = valid_vmax.max().item() if valid_vmax.size > 0 else 0 
        category = (
            0 if vmax_max < 33 else 1 if vmax_max < 43 else 
            2 if vmax_max < 50 else 3 if vmax_max < 58 else 
            4 if vmax_max < 70 else 5
        )

        # 5. Build the Individual Track Dataset (Perfectly Trimmed)
        track_ds = xr.Dataset(
            # The trimmed absolute datetime is now the coordinate
            coords={"time": valid_time_dt}, 
            data_vars={
                "lon": (("time",), valid_lon),
                "lat": (("time",), valid_lat),
                "max_sustained_wind": (("time",), valid_vmax),
                "central_pressure": (("time",), valid_cp),
                "environmental_pressure": (("time",), valid_env),
                "basin": (("time",), np.repeat(basin, n_time)),
                "time_step": (("time",), np.full(n_time, dt_hours)),
            },
            attrs={
                "name": f"{basin}_{args.draw}_{start_year}_{i:04d}",
                "sid": int(i),
                "id_no": int(i),
                "category": category,
                "data_provider": "custom",
                "max_sustained_wind_unit": "kn",
            }
        )

        # 6. Save the File
        # Format: BASIN_YEAR_TRACKID.nc
        filename = f'tracks_{args.basin}_{args.model}_{args.scenario}_{args.variant}_{start_year}_d{args.draw}_t{i:04d}.nc'
        output_path = Path(output_directory) / filename
        
        save_dataset(track_ds, output_path)
        
    print("Finished creating individual track files.")

def augment_and_save_track_file(args: Any) -> None:
    """Enhances a tc-risk track file with additional parameters for CLIMADA."""
    
    input_track_path = get_track_path(args, source=True)
    output_track_path = get_track_path(args, source=False)
    
    print(f"Reading track file: {input_track_path}")
    ds = xr.open_dataset(input_track_path)
    
    ds = add_parameters_to_dataset(
        ds,
        args
    )
    create_track_specific_files(ds, args)

    ds.close()
    print("Enhancement complete.")

def main():
    parser = argparse.ArgumentParser(description='Process tc-risk output for climada input')
    parser.add_argument('--input_data_type', type=str, required=True, default='tc_risk')
    parser.add_argument('--input_io_data_type', type=str, required=True, default='output')
    parser.add_argument('--output_data_type', type=str, required=True, default='climada')
    parser.add_argument('--output_io_data_type', type=str, required=True, default='input')
    parser.add_argument('--data_source', type=str, required=True, default='cmip6')
    parser.add_argument('--model', type=str, required=True, default='ACCESS-CM2')
    parser.add_argument('--variant', type = str, required=True, default='r1i1p1f1')
    parser.add_argument('--scenario', type=str, required=True, default='historical')
    parser.add_argument('--time_period', type=str, required=True, default='1970-1989')
    parser.add_argument("--basin", type=str, required=True, help="Tropical cyclone basin")
    parser.add_argument("--draw", type=int, required=True, help="Draw number or 'all'")
    parser.add_argument('--use_ensemble', type=bool, default=True)
    parser.add_argument('--env_pressure_method', type=str, default='standard')
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    augment_and_save_track_file(args)

if __name__ == "__main__":
    main()