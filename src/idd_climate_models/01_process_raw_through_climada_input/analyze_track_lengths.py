"""
Analyze track lengths in TC-risk output files.
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


def analyze_track_lengths(ds, draw_num):
    """
    Analyze track lengths in a dataset.
    
    Returns:
        dict: Statistics about track lengths
    """
    # Get lon or lat tracks (both should have same valid data pattern)
    lon_trks = ds['lon_trks'].values
    
    # Count non-NaN values per track (along time dimension)
    track_lengths = np.sum(np.isfinite(lon_trks), axis=1)
    
    # Get time step in hours
    time_values = ds['time'].values
    dt_hours = np.diff(time_values).mean() / 3600.0 if len(time_values) > 1 else 1.0
    
    stats = {
        'draw': draw_num,
        'n_tracks': len(track_lengths),
        'min_length': int(track_lengths.min()),
        'max_length': int(track_lengths.max()),
        'mean_length': float(track_lengths.mean()),
        'median_length': float(np.median(track_lengths)),
        'timestep_hours': float(dt_hours),
        'min_duration_hours': float(track_lengths.min() * dt_hours),
        'max_duration_hours': float(track_lengths.max() * dt_hours),
        'mean_duration_hours': float(track_lengths.mean() * dt_hours),
    }
    
    return stats, track_lengths


def main():
    parser = argparse.ArgumentParser(
        description="Analyze track lengths in TC-risk output"
    )
    parser.add_argument('--data_source', type=str, default='cmip6')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--time_period', type=str, required=True)
    parser.add_argument('--basin', type=str, required=True)
    parser.add_argument('--draw', type=int, required=True,
                        help='Draw number to analyze (0-249)')
    parser.add_argument('--show_all', action='store_true',
                        help='Show individual track lengths')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Analyzing track lengths:")
    print(f"  {args.model}/{args.variant}/{args.scenario}/{args.time_period}/{args.basin}")
    print(f"  Draw: {args.draw}")
    print("=" * 80)
    
    # Get file path
    nc_path = get_nc_path(
        args.data_source, args.model, args.variant,
        args.scenario, args.time_period, args.basin, args.draw
    )
    
    if not nc_path.exists():
        print(f"\n❌ File not found: {nc_path}")
        sys.exit(1)
    
    print(f"\n✓ Loading: {nc_path.name}")
    
    # Load dataset
    ds = xr.open_dataset(nc_path)
    
    # Analyze
    stats, track_lengths = analyze_track_lengths(ds, args.draw)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("TRACK LENGTH STATISTICS")
    print("=" * 80)
    print(f"Number of tracks: {stats['n_tracks']}")
    print(f"Time step: {stats['timestep_hours']:.1f} hours")
    print(f"\n📏 File structure:")
    print(f"  Total time dimension: {ds.sizes['time']} timesteps (allocated for ALL tracks)")
    print(f"  Storage shape: ({ds.sizes['n_trk']} tracks × {ds.sizes['time']} timesteps)")
    print(f"\nActual track lengths (valid non-NaN timesteps):")
    print(f"  Min:    {stats['min_length']}")
    print(f"  Max:    {stats['max_length']}")
    print(f"  Mean:   {stats['mean_length']:.1f}")
    print(f"  Median: {stats['median_length']:.1f}")
    
    # Calculate storage efficiency
    total_allocated = ds.sizes['n_trk'] * ds.sizes['time']
    total_valid = track_lengths.sum()
    efficiency = 100 * total_valid / total_allocated
    wasted_timesteps = total_allocated - total_valid
    
    print(f"\n💾 Storage efficiency:")
    print(f"  Allocated timesteps: {total_allocated:,} (all tracks × max time)")
    print(f"  Valid timesteps:     {int(total_valid):,} (actual data)")
    print(f"  NaN padding:         {int(wasted_timesteps):,} ({100-efficiency:.1f}% of file)")
    print(f"  Efficiency:          {efficiency:.1f}%")
    
    print(f"\nTrack durations (hours):")
    print(f"  Min:    {stats['min_duration_hours']:.1f} hours ({stats['min_duration_hours']/24:.1f} days)")
    print(f"  Max:    {stats['max_duration_hours']:.1f} hours ({stats['max_duration_hours']/24:.1f} days)")
    print(f"  Mean:   {stats['mean_duration_hours']:.1f} hours ({stats['mean_duration_hours']/24:.1f} days)")
    
    # Distribution
    print(f"\nLength distribution:")
    bins = [0, 50, 100, 150, 200, 250, 300, np.inf]
    bin_labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300+']
    hist, _ = np.histogram(track_lengths, bins=bins)
    for label, count in zip(bin_labels, hist):
        if count > 0:
            pct = 100 * count / len(track_lengths)
            print(f"  {label:>10} timesteps: {count:4d} tracks ({pct:5.1f}%)")
    
    if args.show_all:
        print(f"\n" + "=" * 80)
        print("INDIVIDUAL TRACK LENGTHS")
        print("=" * 80)
        for i, length in enumerate(track_lengths):
            duration_hours = length * stats['timestep_hours']
            print(f"Track {i:4d}: {int(length):3d} timesteps ({duration_hours:6.1f} hours, {duration_hours/24:5.1f} days)")
    
    ds.close()


if __name__ == '__main__':
    main()
