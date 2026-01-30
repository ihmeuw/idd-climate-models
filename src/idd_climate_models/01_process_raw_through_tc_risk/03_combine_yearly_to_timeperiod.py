"""
Combine yearly processed files into time-period files for TC-risk.

This script is called by the orchestrator as Level 3 of the workflow.
It:
  1. Reads yearly processed files from PROCESSED_DATA_PATH
  2. Combines them for the specified time period
  3. Regrids to target grid if needed
  4. Writes output to TC_RISK_INPUT_PATH

Since yearly files are already NaN-filled, this is just concatenation + optional regridding.
"""

import argparse
import sys
from pathlib import Path
import xarray as xr

import idd_climate_models.constants as rfc
from idd_climate_models.climate_file_functions import (
    should_regrid,
    regrid_with_cdo,
)
from idd_climate_models.validation_functions import is_monthly


def main():
    parser = argparse.ArgumentParser(
        description='Combine yearly processed files into time-period files for TC-risk'
    )
    parser.add_argument('--data_source', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--time_period', required=True)
    parser.add_argument('--variable', required=True)
    parser.add_argument('--grid', required=True)
    parser.add_argument('--frequency', required=True)
    parser.add_argument('--delete_yearly', type=lambda x: str(x).lower() == 'true', default=False,
                        help='Delete yearly files after combining')

    args = parser.parse_args()

    # --- 1. DEFINE PATHS AND SETTINGS ---
    TARGET_GRID = 'r360x180'  # 1x1 global grid

    # Target: TC-risk input (combined files go here)
    target_dir = (
        rfc.TC_RISK_INPUT_PATH / args.data_source / args.model / args.variant /
        args.scenario / args.time_period
    )

    # Source: Check for yearly files in variable subfolder first (new location),
    # then fall back to PROCESSED_DATA_PATH (legacy location)
    source_dir_new = target_dir / args.variable  # New: TC_RISK_INPUT_PATH/.../time_period/variable/
    source_dir_legacy = (
        rfc.PROCESSED_DATA_PATH / args.data_source / args.model / args.variant /
        args.scenario / args.variable / args.grid / args.frequency
    )

    # Use new location if it exists, otherwise fall back to legacy
    if source_dir_new.exists() and list(source_dir_new.glob("*.nc")):
        source_dir = source_dir_new
        print(f"  Using yearly files from variable subfolder: {source_dir}")
    else:
        source_dir = source_dir_legacy
        print(f"  Using yearly files from processed data path: {source_dir}")

    print(f"\n{'='*80}")
    print(f"Combining yearly files for {args.variable}/{args.frequency}")
    print(f"Model: {args.model}/{args.variant}")
    print(f"Scenario: {args.scenario}")
    print(f"Time period: {args.time_period}")
    print(f"Source dir: {source_dir}")
    print(f"Target dir: {target_dir}")
    print(f"{'='*80}\n")

    target_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. PARSE TIME PERIOD ---
    bin_start_str, bin_end_str = args.time_period.split('-')
    bin_start = int(bin_start_str)
    bin_end = int(bin_end_str)

    # --- 3. FIND YEARLY FILES IN RANGE ---
    if not source_dir.exists():
        print(f"  Source directory does not exist: {source_dir}")
        sys.exit(1)

    # Find yearly files matching the time period
    yearly_files = []
    for year in range(bin_start, bin_end + 1):
        # Try different filename patterns
        patterns = [
            f"{args.variable}_{args.frequency}_{args.model}_{args.scenario}_{args.variant}_{args.grid}_{year}*.nc",
            f"{args.variable}*{year}*.nc",
        ]
        for pattern in patterns:
            matches = list(source_dir.glob(pattern))
            if matches:
                yearly_files.extend(matches)
                break

    yearly_files = sorted(set(yearly_files))

    if not yearly_files:
        print(f"  No yearly files found for {args.variable} in {bin_start}-{bin_end}")
        print(f"  Looked in: {source_dir}")
        sys.exit(1)

    print(f"  Found {len(yearly_files)} yearly files")

    # --- 4. LOAD AND COMBINE FILES ---
    print(f"  Loading and combining files...")

    try:
        # Open all yearly files - they're already NaN-filled so just concat
        ds = xr.open_mfdataset(
            yearly_files,
            combine='by_coords',
            engine='netcdf4'
        )

        # Verify time range
        actual_years = sorted(set(ds.time.dt.year.values))
        print(f"  Combined data for years: {actual_years[0]}-{actual_years[-1]} ({len(actual_years)} years)")
        print(f"  Time steps: {len(ds.time)}")

    except Exception as e:
        print(f"  Failed to load files: {e}")
        sys.exit(1)

    # --- 5. CHECK IF REGRIDDING IS NEEDED ---
    needs_regridding = should_regrid(str(yearly_files[0]), target_grid=TARGET_GRID)

    # Set output grid label
    output_grid = 'gr' if needs_regridding else args.grid

    # Construct date strings based on frequency
    if is_monthly(args.frequency):
        date_start = f"{bin_start}01"
        date_end = f"{bin_end}12"
    else:
        date_start = f"{bin_start}0101"
        date_end = f"{bin_end}1231"

    # Construct output filename
    output_filename = (
        f"{args.variable}_{args.frequency}_{args.model}_{args.scenario}_"
        f"{args.variant}_{output_grid}_{date_start}-{date_end}.nc"
    )
    output_path = target_dir / output_filename

    # --- 6. WRITE OUTPUT ---
    print(f"  Writing output...")

    # Load into memory for writing (yearly files are small enough)
    ds = ds.compute()

    # Drop time_bnds if present (causes encoding issues)
    if 'time_bnds' in ds:
        ds = ds.drop_vars('time_bnds')

    # Encoding for compression
    encoding = {
        var: {
            'zlib': True,
            'complevel': 7,
            'shuffle': True,
        }
        for var in ds.data_vars
    }

    if needs_regridding:
        print(f"  Will regrid from {args.grid} to {output_grid} ({TARGET_GRID})")

        # Write to temporary file first
        temp_path = target_dir / f"temp_{output_filename}"
        ds.to_netcdf(temp_path, encoding=encoding, engine='netcdf4')
        ds.close()

        # Regrid using CDO
        success = regrid_with_cdo(temp_path, output_path, TARGET_GRID)

        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

        if not success:
            print(f"  Failed to regrid to {TARGET_GRID}")
            sys.exit(1)

        print(f"  Successfully regridded to {output_filename}")

    else:
        print(f"  Files already on target grid, no regridding needed")
        ds.to_netcdf(output_path, encoding=encoding, engine='netcdf4')
        ds.close()
        print(f"  Successfully wrote {output_filename}")

    # --- 7. DELETE YEARLY FILES IF REQUESTED ---
    if args.delete_yearly:
        print(f"  Deleting {len(yearly_files)} yearly files...")
        for f in yearly_files:
            try:
                f.unlink()
            except Exception as e:
                print(f"    Warning: could not delete {f}: {e}")

        # If source was a variable subfolder, try to remove the empty folder
        if source_dir == source_dir_new and source_dir.exists():
            try:
                source_dir.rmdir()  # Only removes if empty
                print(f"  Removed empty variable subfolder: {source_dir}")
            except OSError:
                pass  # Folder not empty or other error, ignore

        print(f"  Yearly files deleted.")

    # --- 8. SUMMARY ---
    print(f"\n{'='*80}")
    print(f"Completed {args.variable}/{args.frequency}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
