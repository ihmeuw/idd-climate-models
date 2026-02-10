import argparse
import sys
import os
from pathlib import Path
from typing import Any, Optional
import xarray as xr
import numpy as np

import idd_climate_models.constants as rfc
from idd_climate_models.add_tc_params import add_parameters_to_dataset
from idd_climate_models.climate_file_functions import get_dir, get_track_path
from idd_climate_models.zarr_functions import verify_zarr_integrity

# --- Helper function for setting permissions on input files ---
def set_input_file_permissions(file_path: Path, mode: int = 0o775):
    """
    Sets permissions on input .nc file before reading.
    Handles both files and directories if the path points to a directory-based format.
    """
    try:
        if file_path.is_file():
            os.chmod(str(file_path), mode)
            print(f"Set permissions {oct(mode)} on input file: {file_path}")
        elif file_path.is_dir():
            # For directory-based formats (like zarr), set recursively
            os.chmod(str(file_path), mode)
            for root, dirs, files in os.walk(file_path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), mode)
                for f in files:
                    os.chmod(os.path.join(root, f), mode)
            print(f"Set permissions {oct(mode)} recursively on input directory: {file_path}")
    except Exception as e:
        print(f"Warning: Could not set permissions on {file_path}: {e}")
        # Don't fail - just warn and continue

# --- Helper function for setting permissions on output zarr (unchanged) ---
def set_zarr_permissions_recursive(zarr_path: Path, mode: int = 0o775):
    """Recursively sets permissions (mode) for all files and directories in the Zarr store."""
    
    # 1. Set permissions on the root Zarr directory
    os.chmod(str(zarr_path), mode)
    
    # 2. Walk through the Zarr store and set permissions on all files/subdirs
    for dirpath, dirnames, filenames in os.walk(zarr_path):
        # Set permissions for subdirectories
        for dirname in dirnames:
            os.chmod(Path(dirpath) / dirname, mode)
        
        # Set permissions for files
        for filename in filenames:
            os.chmod(Path(dirpath) / filename, mode)

# --- Core Processing Function ---

def create_storm_list_file(ds_multi, args):
    """
    Processes and saves each individual track as a Zarr Group within a single 
    Zarr store, preserving the list-of-datasets structure.
    """
    n_trk = ds_multi.sizes["n_trk"]
    raw_time = ds_multi["time"].values
    dt_hours = np.diff(raw_time).mean() / 3600.0 if len(raw_time) > 1 else 1.0

    storms = []
    # Loop through each track and create the independent Dataset objects
    for i in range(n_trk):
        if np.isnan(ds_multi["tc_years"][i].item()) or np.isnan(ds_multi["tc_month"][i].item()):
             continue
        
        # 1. Get Metadata & Data
        start_year = int(ds_multi["tc_years"][i].item())
        start_month = int(ds_multi["tc_month"][i].item())
        basin = ds_multi["tc_basins"][i].item()
        
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
        
        # 3. Apply Corrections
        lon = ((lon + 180) % 360) - 180
        lat = np.clip(lat, -90, 90)

        # 4. Trim NaNs and Calculate Category
        valid_idx = np.isfinite(lon) & np.isfinite(lat)
        valid_lon, valid_lat, valid_vmax, valid_cp, valid_env, valid_time_dt = (
            arr[valid_idx] for arr in [lon, lat, vmax, cp, env, time_dt]
        )
        n_time = len(valid_lon)
        
        if n_time == 0:
            continue

        vmax_max = valid_vmax.max().item() if valid_vmax.size > 0 else 0 
        category = 0 if vmax_max < 33 else 1 if vmax_max < 43 else 2 if vmax_max < 50 else 3 if vmax_max < 58 else 4 if vmax_max < 70 else 5

        # 5. Build the Individual Track Dataset
        track_ds = xr.Dataset(
            coords={"time": valid_time_dt, "storm_id": i}, 
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
        storms.append(track_ds)
    
    
    # --- 6. Save the Storms List to Zarr Groups ---
    output_zarr_path = get_track_path(args, source=False, extension=".zarr")
    
    try:
        print(f"Saving {len(storms)} storms to Zarr at: {output_zarr_path}")

        # Save first dataset (creates the zarr store)
        first_ds = storms[0]
        first_ds.to_zarr(
            output_zarr_path,
            group=f'storm_{first_ds.storm_id.item():04d}', 
            mode='w',
            consolidated=True
        )

        # Append remaining datasets as separate groups
        for ds in storms[1:]:
            group_name = f'storm_{ds.storm_id.item():04d}'
            ds.to_zarr(
                output_zarr_path,
                group=group_name,
                mode='a'
            )
        
        print(f"Successfully saved {len(storms)} storms to Zarr.")
        
        # Set Permissions
        set_zarr_permissions_recursive(output_zarr_path, 0o775)
        print("Output file permissions set to 775 recursively.")

    except Exception as e:
        print(f"Error during Zarr saving: {e}")
        raise

# Modified augment function to set input permissions
def augment_and_save_track_file(args):
    """Enhanced version with input permission setting and verification."""
    
    input_track_path = get_track_path(args, source=True)
    
    # SET PERMISSIONS ON INPUT FILE BEFORE READING
    print(f"Setting permissions on input file: {input_track_path}")
    set_input_file_permissions(input_track_path, 0o775)
    
    print(f"Reading track file: {input_track_path}")
    ds = xr.open_dataset(input_track_path)
    
    ds = add_parameters_to_dataset(ds, args)
    create_storm_list_file(ds, args)
    
    ds.close()
    print("Enhancement complete.")
    
    # VERIFY THE OUTPUT
    output_zarr_path = get_track_path(args, source=False, extension=".zarr")
    verify_zarr_integrity(input_track_path, output_zarr_path, args)


# Helper function to check if a zarr file exists and is valid
def check_zarr_exists_and_valid(output_path: Path, args, draw: int) -> bool:
    """
    Checks if the output zarr file exists and appears to be complete.
    Returns True if the draw has already been successfully processed.
    """
    if not output_path.exists():
        return False
    
    try:
        # Try to open the zarr store to verify it's not corrupted
        import zarr
        store = zarr.open(str(output_path), mode='r')
        
        # Check if there are any storm groups (basic validation)
        groups = list(store.group_keys())
        if len(groups) == 0:
            print(f"  ⚠️  Zarr exists but has no storm groups - will reprocess")
            return False
        
        # Optional: More thorough validation
        # Check if we can read the first group without errors
        first_group = groups[0]
        test_group = store[first_group]
        
        print(f"  ✓ Found existing valid zarr with {len(groups)} storm groups")
        return True
        
    except Exception as e:
        print(f"  ⚠️  Zarr exists but appears corrupted ({e}) - will reprocess")
        return False


# NEW: Process all draws for a basin with resume capability
def process_all_draws_for_basin(args):
    """
    Processes all draws for a given basin/model/variant/scenario/time_period combination.
    This replaces the single-draw processing to reduce job count.
    
    RESUME CAPABILITY: Checks for existing zarr files and skips already-completed draws.
    """
    print("=" * 80)
    print(f"Processing ALL draws for:")
    print(f"  Model: {args.model}")
    print(f"  Variant: {args.variant}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Time Period: {args.time_period}")
    print(f"  Basin: {args.basin}")
    print(f"  Number of draws: {args.num_draws}")
    print("=" * 80)
    
    # First pass: identify which draws need processing
    print("\n" + "─" * 80)
    print("CHECKING EXISTING OUTPUTS")
    print("─" * 80)
    
    draws_to_process = []
    draws_already_done = []
    
    for draw in range(args.num_draws):
        # Create temporary args to get the output path
        draw_args = argparse.Namespace(**vars(args))
        draw_args.draw = draw
        
        output_zarr_path = get_track_path(draw_args, source=False, extension=".zarr")
        
        if check_zarr_exists_and_valid(output_zarr_path, args, draw):
            draws_already_done.append(draw)
            print(f"Draw {draw:3d}: Already complete - SKIPPING")
        else:
            draws_to_process.append(draw)
            print(f"Draw {draw:3d}: Not found or incomplete - will process")
    
    print("\n" + "─" * 80)
    print(f"Resume Summary:")
    print(f"  Total draws: {args.num_draws}")
    print(f"  Already complete: {len(draws_already_done)}")
    print(f"  To process: {len(draws_to_process)}")
    print("─" * 80)
    
    # If everything is already done, exit successfully
    if len(draws_to_process) == 0:
        print("\n✅ All draws already completed! Nothing to do.")
        return
    
    # Second pass: process only the draws that need it
    successful_draws = len(draws_already_done)  # Count pre-existing as successful
    failed_draws = []
    
    for idx, draw in enumerate(draws_to_process, 1):
        print(f"\n{'─' * 80}")
        print(f"Processing draw {draw} ({idx}/{len(draws_to_process)} remaining)")
        print(f"{'─' * 80}")
        
        # Create a copy of args with the current draw number
        draw_args = argparse.Namespace(**vars(args))
        draw_args.draw = draw
        
        try:
            augment_and_save_track_file(draw_args)
            successful_draws += 1
            print(f"✅ Draw {draw} completed successfully")
        except Exception as e:
            failed_draws.append((draw, str(e)))
            print(f"❌ Draw {draw} failed: {e}")
            # Continue processing other draws even if one fails
            continue
    
    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total draws: {args.num_draws}")
    print(f"Previously completed: {len(draws_already_done)}")
    print(f"Newly processed: {len(draws_to_process)}")
    print(f"Newly successful: {len(draws_to_process) - len(failed_draws)}")
    print(f"Failed: {len(failed_draws)}")
    print(f"Overall completion: {successful_draws}/{args.num_draws} ({100*successful_draws/args.num_draws:.1f}%)")
    
    if failed_draws:
        print("\nFailed draws:")
        for draw, error in failed_draws:
            print(f"  Draw {draw}: {error}")
        
        # Exit with error if any draws failed
        print(f"\n❌ Job failed: {len(failed_draws)} draw(s) unsuccessful")
        print("💡 You can re-run this job and it will only process the failed draws")
        sys.exit(1)
    else:
        print("\n✅ All draws processed successfully!")


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
    
    # Modified: draw is now optional, num_draws is for batch processing
    parser.add_argument("--draw", type=int, required=False, help="Single draw number (for backward compatibility)")
    parser.add_argument("--num_draws", type=int, required=False, help="Process all draws from 0 to num_draws-1")
    
    parser.add_argument('--use_ensemble', type=bool, default=True)
    parser.add_argument('--env_pressure_method', type=str, default='standard')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--verify', type=bool, default=True)
    args = parser.parse_args()

    # Determine which mode to run in
    if args.num_draws is not None:
        # Batch mode: process all draws
        process_all_draws_for_basin(args)
    elif args.draw is not None:
        # Single draw mode (backward compatibility)
        augment_and_save_track_file(args)
    else:
        print("❌ ERROR: Must specify either --draw (single) or --num_draws (batch)")
        sys.exit(1)


if __name__ == "__main__":
    main()
