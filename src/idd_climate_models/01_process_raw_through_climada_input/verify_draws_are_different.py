"""
Verify that different draw files contain different data.

This checks that .nc files (or .zarr files) for the same basin/model/scenario/time_period
are actually different from each other, ensuring no duplicate data generation.
"""

import sys
import argparse
from pathlib import Path
import xarray as xr
import numpy as np
import idd_climate_models.constants as rfc


def get_draw_suffix(draw_num):
    """Get the filename suffix for a given draw number."""
    if draw_num == 0:
        return ''
    else:
        return f'_e{draw_num - 1}'


def get_nc_path(data_source, model, variant, scenario, time_period, basin, draw_num):
    """Get path to .nc file for a specific draw."""
    output_path = rfc.TC_RISK_OUTPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    time_parts = time_period.split('-')
    time_start_str = f'{int(time_parts[0]):04d}01'
    time_end_str = f'{int(time_parts[1]):04d}12'
    
    base_pattern = f'tracks_{basin}_{model}_{scenario}_{variant}_{time_start_str}_{time_end_str}'
    suffix = get_draw_suffix(draw_num)
    
    return output_path / f'{base_pattern}{suffix}.nc'


def get_zarr_path(data_source, model, variant, scenario, time_period, basin, draw_num):
    """Get path to .zarr file for a specific draw."""
    output_path = rfc.CLIMADA_INPUT_PATH / data_source / model / variant / scenario / time_period / basin
    
    time_parts = time_period.split('-')
    time_start_str = f'{int(time_parts[0]):04d}01'
    time_end_str = f'{int(time_parts[1]):04d}12'
    
    base_pattern = f'tracks_{basin}_{model}_{scenario}_{variant}_{time_start_str}_{time_end_str}'
    suffix = get_draw_suffix(draw_num)
    
    return output_path / f'{base_pattern}{suffix}.zarr'


def compare_datasets(ds1, ds2, draw1, draw2):
    """
    Compare two xarray datasets and return whether they're different.
    
    Returns:
        tuple: (are_different: bool, difference_summary: str)
    """
    differences = []
    
    # Variables that are expected to be identical (metadata)
    EXPECTED_IDENTICAL = {'tc_basins', 'tc_years', 'year', 'basin', 'month'}
    
    # Variables that should differ (actual storm data)
    MUST_DIFFER = {'lon_trks', 'lat_trks', 'vmax_trks', 'v_trks', 'm_trks'}
    
    # Compare dimensions (use sizes to avoid FutureWarning)
    if ds1.sizes != ds2.sizes:
        differences.append(f"Different dimensions: {ds1.sizes} vs {ds2.sizes}")
    
    # Compare variables
    common_vars = set(ds1.data_vars) & set(ds2.data_vars)
    
    critical_identical = []  # Track if critical variables are identical
    
    for var in common_vars:
        arr1 = ds1[var].values
        arr2 = ds2[var].values
        
        is_expected_identical = var in EXPECTED_IDENTICAL
        is_must_differ = var in MUST_DIFFER
        
        # Handle different data types appropriately
        if arr1.dtype.kind in ('U', 'S', 'O'):  # String or object types
            # For strings, use direct comparison
            if np.array_equal(arr1, arr2):
                if is_expected_identical:
                    differences.append(f"  Variable '{var}' is identical (expected)")
                else:
                    differences.append(f"⚠️  Variable '{var}' is IDENTICAL between draws {draw1} and {draw2}")
                    if is_must_differ:
                        critical_identical.append(var)
            else:
                num_different = np.sum(arr1 != arr2)
                differences.append(f"✓ Variable '{var}' differs ({num_different} elements different)")
        else:
            # For numeric types, use equal_nan
            try:
                if np.array_equal(arr1, arr2, equal_nan=True):
                    if is_expected_identical:
                        differences.append(f"  Variable '{var}' is identical (expected)")
                    else:
                        differences.append(f"⚠️  Variable '{var}' is IDENTICAL between draws {draw1} and {draw2}")
                        if is_must_differ:
                            critical_identical.append(var)
                else:
                    # Calculate some difference metrics for numeric data
                    finite_mask = np.isfinite(arr1) & np.isfinite(arr2)
                    if finite_mask.any():
                        diff = np.abs(arr1[finite_mask] - arr2[finite_mask])
                        max_diff = diff.max()
                        mean_diff = diff.mean()
                        differences.append(f"✓ Variable '{var}' differs (max: {max_diff:.4f}, mean: {mean_diff:.4f})")
                    else:
                        differences.append(f"✓ Variable '{var}' differs (no finite values to compare)")
            except TypeError:
                # Fallback for any other type issues
                if np.array_equal(arr1, arr2):
                    if is_expected_identical:
                        differences.append(f"  Variable '{var}' is identical (expected)")
                    else:
                        differences.append(f"⚠️  Variable '{var}' is IDENTICAL between draws {draw1} and {draw2}")
                        if is_must_differ:
                            critical_identical.append(var)
                else:
                    differences.append(f"✓ Variable '{var}' differs")
    
    # Only flag as problematic if critical variables are identical
    are_different = len(critical_identical) == 0
    summary = "\n  ".join(differences)
    
    return are_different, summary


def main():
    parser = argparse.ArgumentParser(
        description="Verify that different draw files contain different data"
    )
    parser.add_argument('--data_source', type=str, default='cmip6')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--time_period', type=str, required=True)
    parser.add_argument('--basin', type=str, required=True)
    parser.add_argument('--draw_start', type=int, required=True,
                        help='Starting draw number (0-249, inclusive)')
    parser.add_argument('--draw_end', type=int, required=True,
                        help='Ending draw number (0-249, inclusive)')
    parser.add_argument('--file_type', type=str, choices=['nc', 'zarr'], default='nc',
                        help='File type to check (nc or zarr)')
    parser.add_argument('--compare_all', action='store_true',
                        help='Compare all pairs (warning: O(n²) comparisons)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Verifying draws are different:")
    print(f"  {args.model}/{args.variant}/{args.scenario}/{args.time_period}/{args.basin}")
    print(f"  Draws: {args.draw_start}-{args.draw_end}")
    print(f"  File type: .{args.file_type}")
    print("=" * 80)
    
    # Get file paths
    get_path_func = get_nc_path if args.file_type == 'nc' else get_zarr_path
    
    draws = list(range(args.draw_start, args.draw_end + 1))
    file_paths = {}
    missing_files = []
    
    for draw_num in draws:
        path = get_path_func(
            args.data_source, args.model, args.variant,
            args.scenario, args.time_period, args.basin, draw_num
        )
        if path.exists():
            file_paths[draw_num] = path
        else:
            missing_files.append(draw_num)
    
    if missing_files:
        print(f"\n⚠️  Missing files for draws: {missing_files}")
        if len(file_paths) < 2:
            print("❌ Need at least 2 files to compare")
            sys.exit(1)
    
    print(f"\nFound {len(file_paths)} files to compare")
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = {}
    for draw_num, path in file_paths.items():
        try:
            if args.file_type == 'zarr':
                ds = xr.open_zarr(path)
            else:
                ds = xr.open_dataset(path)
            datasets[draw_num] = ds
            print(f"  ✓ Loaded draw {draw_num}")
        except Exception as e:
            print(f"  ❌ Failed to load draw {draw_num}: {e}")
    
    if len(datasets) < 2:
        print("\n❌ Need at least 2 valid datasets to compare")
        sys.exit(1)
    
    # Compare datasets
    print("\n" + "=" * 80)
    print("Comparing datasets...")
    print("=" * 80)
    
    draw_nums = sorted(datasets.keys())
    identical_pairs = []
    different_pairs = []
    
    if args.compare_all:
        # Compare all pairs
        for i, draw1 in enumerate(draw_nums):
            for draw2 in draw_nums[i+1:]:
                print(f"\n--- Comparing draw {draw1} vs draw {draw2} ---")
                are_different, summary = compare_datasets(
                    datasets[draw1], datasets[draw2], draw1, draw2
                )
                print(f"  {summary}")
                
                if are_different:
                    different_pairs.append((draw1, draw2))
                else:
                    identical_pairs.append((draw1, draw2))
    else:
        # Compare consecutive pairs only
        for i in range(len(draw_nums) - 1):
            draw1 = draw_nums[i]
            draw2 = draw_nums[i + 1]
            print(f"\n--- Comparing draw {draw1} vs draw {draw2} ---")
            are_different, summary = compare_datasets(
                datasets[draw1], datasets[draw2], draw1, draw2
            )
            print(f"  {summary}")
            
            if are_different:
                different_pairs.append((draw1, draw2))
            else:
                identical_pairs.append((draw1, draw2))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Datasets compared: {len(datasets)}")
    print(f"Different pairs: {len(different_pairs)}")
    print(f"Identical pairs: {len(identical_pairs)}")
    
    if identical_pairs:
        print("\n⚠️  WARNING: Found IDENTICAL pairs (possible duplicate generation):")
        for draw1, draw2 in identical_pairs:
            print(f"  - Draws {draw1} and {draw2}")
        sys.exit(1)
    else:
        print("\n✅ All compared draws are different (as expected)")
    
    # Close datasets
    for ds in datasets.values():
        ds.close()


if __name__ == '__main__':
    main()
