"""
Verification and visualization script for TC parameters added to NetCDF files.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path


def compare_before_after(original_file, enhanced_file):
    """
    Compare original and enhanced datasets.
    
    Parameters
    ----------
    original_file : str
        Path to original NetCDF file
    enhanced_file : str
        Path to enhanced NetCDF file with parameters
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Load datasets
    ds_orig = xr.open_dataset(original_file)
    ds_enh = xr.open_dataset(enhanced_file)
    
    print("="*70)
    print("DATASET COMPARISON")
    print("="*70)
    
    print("\nOriginal variables:")
    for var in ds_orig.data_vars:
        print(f"  - {var}")
    
    print("\nNew variables added:")
    new_vars = set(ds_enh.data_vars) - set(ds_orig.data_vars)
    for var in sorted(new_vars):
        print(f"  - {var}: {ds_enh[var].attrs.get('long_name', 'N/A')}")
    
    print("\n" + "="*70)
    
    ds_orig.close()
    ds_enh.close()
    
    return new_vars


def plot_track_with_parameters(ds, year_idx=0, track_idx=0, save_path=None):
    """
    Create comprehensive visualization of a track with all parameters.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Enhanced dataset with TC parameters
    year_idx : int
        Year index to plot
    track_idx : int
        Track index to plot
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Select track
    track = ds.isel(year=year_idx, n_trk=track_idx)
    
    # Get data
    lon = track.lon_trks.values
    lat = track.lat_trks.values
    vmax = track.vmax_trks.values * 1.943844  # m/s to knots
    
    # New parameters
    p_central = track.central_pressure.values
    p_env = track.environmental_pressure.values
    p_deficit = track.pressure_deficit.values
    rmw = track.rmw.values
    roci = track.roci.values
    
    # Remove NaN
    valid = ~np.isnan(lat)
    lon = lon[valid]
    lat = lat[valid]
    vmax = vmax[valid]
    p_central = p_central[valid]
    p_env = p_env[valid]
    p_deficit = p_deficit[valid]
    rmw = rmw[valid]
    roci = roci[valid]
    
    if len(lat) == 0:
        print("No valid data for this track")
        return None
    
    # Get basin and other metadata
    basin = track.tc_basins.values.item() if 'tc_basins' in track else 'Unknown'
    year_val = ds.year.values[year_idx]
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # Create GridSpec for complex layout
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Track map colored by intensity
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(lon, lat, c=vmax, cmap='YlOrRd', s=80, 
                         edgecolors='black', linewidth=0.5, zorder=3)
    ax1.plot(lon, lat, 'k-', alpha=0.3, linewidth=1, zorder=2)
    ax1.scatter(lon[0], lat[0], marker='o', s=200, c='green', 
               edgecolors='black', linewidth=2, zorder=4, label='Start')
    ax1.scatter(lon[-1], lat[-1], marker='X', s=200, c='red',
               edgecolors='black', linewidth=2, zorder=4, label='End')
    ax1.set_xlabel('Longitude (°E)', fontsize=10)
    ax1.set_ylabel('Latitude (°N)', fontsize=10)
    ax1.set_title(f'Track - {basin} Basin, Year {year_val}', fontsize=11, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='best', fontsize=8)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Wind Speed (kt)', fontsize=9)
    
    # 2. Pressure evolution
    ax2 = fig.add_subplot(gs[0, 1])
    time_steps = np.arange(len(lat))
    ax2.plot(time_steps, p_central, 'b-', linewidth=2, label='Central Pressure')
    ax2.plot(time_steps, p_env, 'r--', linewidth=2, label='Environmental Pressure')
    ax2.fill_between(time_steps, p_central, p_env, alpha=0.2, color='gray')
    ax2.set_xlabel('Time Step', fontsize=10)
    ax2.set_ylabel('Pressure (hPa)', fontsize=10)
    ax2.set_title('Pressure Evolution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.invert_yaxis()
    
    # 3. Pressure deficit
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time_steps, p_deficit, 'purple', linewidth=2)
    ax3.fill_between(time_steps, 0, p_deficit, alpha=0.3, color='purple')
    ax3.set_xlabel('Time Step', fontsize=10)
    ax3.set_ylabel('Pressure Deficit (hPa)', fontsize=10)
    ax3.set_title('Central Pressure Deficit', fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 4. Pressure-Wind relationship
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(vmax, p_central, c=lat, cmap='viridis', 
                         s=60, edgecolors='black', linewidth=0.5, alpha=0.7)
    ax4.set_xlabel('Wind Speed (kt)', fontsize=10)
    ax4.set_ylabel('Central Pressure (hPa)', fontsize=10)
    ax4.set_title('Pressure-Wind Relationship', fontsize=11, fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.invert_yaxis()
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Latitude (°N)', fontsize=9)
    
    # 5. RMW and ROCI evolution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(time_steps, rmw, 'b-', linewidth=2, marker='o', markersize=4, label='RMW')
    ax5.plot(time_steps, roci, 'r-', linewidth=2, marker='s', markersize=4, label='ROCI')
    ax5.set_xlabel('Time Step', fontsize=10)
    ax5.set_ylabel('Radius (nm)', fontsize=10)
    ax5.set_title('Storm Size Evolution', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    # 6. Size ratio
    ax6 = fig.add_subplot(gs[1, 2])
    size_ratio = roci / rmw
    ax6.plot(time_steps, size_ratio, 'g-', linewidth=2)
    ax6.fill_between(time_steps, size_ratio, alpha=0.3, color='green')
    ax6.set_xlabel('Time Step', fontsize=10)
    ax6.set_ylabel('ROCI / RMW Ratio', fontsize=10)
    ax6.set_title('Relative Storm Size', fontsize=11, fontweight='bold')
    ax6.grid(alpha=0.3)
    ax6.axhline(y=np.mean(size_ratio), color='r', linestyle='--', 
               label=f'Mean: {np.mean(size_ratio):.1f}')
    ax6.legend(fontsize=9)
    
    # 7. Intensity metrics
    ax7 = fig.add_subplot(gs[2, 0])
    ax7_twin = ax7.twinx()
    
    line1 = ax7.plot(time_steps, vmax, 'b-', linewidth=2, label='Wind Speed')
    ax7.set_xlabel('Time Step', fontsize=10)
    ax7.set_ylabel('Wind Speed (kt)', fontsize=10, color='b')
    ax7.tick_params(axis='y', labelcolor='b')
    
    line2 = ax7_twin.plot(time_steps, p_deficit, 'r-', linewidth=2, label='Pressure Deficit')
    ax7_twin.set_ylabel('Pressure Deficit (hPa)', fontsize=10, color='r')
    ax7_twin.tick_params(axis='y', labelcolor='r')
    
    ax7.set_title('Intensity Metrics', fontsize=11, fontweight='bold')
    ax7.grid(alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax7.legend(lines, labels, loc='upper left', fontsize=9)
    
    # 8. RMW vs Latitude
    ax8 = fig.add_subplot(gs[2, 1])
    scatter = ax8.scatter(np.abs(lat), rmw, c=vmax, cmap='YlOrRd',
                         s=60, edgecolors='black', linewidth=0.5, alpha=0.7)
    ax8.set_xlabel('|Latitude| (°)', fontsize=10)
    ax8.set_ylabel('RMW (nm)', fontsize=10)
    ax8.set_title('RMW vs Latitude', fontsize=11, fontweight='bold')
    ax8.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax8)
    cbar.set_label('Wind Speed (kt)', fontsize=9)
    
    # 9. Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    stats_text = f"""
    TRACK SUMMARY
    {'='*35}
    Basin: {basin}
    Year: {year_val}
    Track Length: {len(lat)} points
    
    INTENSITY
    Max Wind: {np.max(vmax):.1f} kt
    Min Pressure: {np.min(p_central):.1f} hPa
    Max Deficit: {np.max(p_deficit):.1f} hPa
    
    SIZE
    Mean RMW: {np.mean(rmw):.1f} nm
    Mean ROCI: {np.mean(roci):.1f} nm
    Mean Ratio: {np.mean(size_ratio):.1f}
    
    LOCATION
    Lat Range: {np.min(lat):.1f}° to {np.max(lat):.1f}°
    Lon Range: {np.min(lon):.1f}° to {np.max(lon):.1f}°
    """
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Complete TC Analysis - Track {track_idx}, Year {year_val}',
                fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def generate_dataset_summary(ds, output_file=None):
    """
    Generate statistical summary of all parameters in the dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Enhanced dataset
    output_file : str, optional
        Path to save summary text file
    
    Returns
    -------
    summary : dict
        Dictionary of summary statistics
    """
    print("\n" + "="*70)
    print("DATASET PARAMETER SUMMARY")
    print("="*70)
    
    # Get all tracks
    n_years = len(ds.year)
    n_tracks = len(ds.n_trk)
    
    # Initialize collectors
    all_stats = {
        'vmax': [],
        'central_pressure': [],
        'environmental_pressure': [],
        'pressure_deficit': [],
        'rmw': [],
        'roci': [],
    }
    
    # Collect all valid data
    for var in all_stats.keys():
        if var == 'vmax':
            data = ds.vmax_trks.values.flatten() * 1.943844  # m/s to kt
        else:
            data = ds[var].values.flatten()
        
        all_stats[var] = data[~np.isnan(data)]
    
    # Print statistics
    summary = {}
    
    print("\nVARIABLE STATISTICS:")
    print("-"*70)
    
    for var, data in all_stats.items():
        if len(data) == 0:
            continue
        
        stats = {
            'count': len(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'q25': np.percentile(data, 25),
            'median': np.median(data),
            'q75': np.percentile(data, 75),
            'max': np.max(data),
        }
        
        summary[var] = stats
        
        # Units
        units = {
            'vmax': 'kt',
            'central_pressure': 'hPa',
            'environmental_pressure': 'hPa',
            'pressure_deficit': 'hPa',
            'rmw': 'nm',
            'roci': 'nm',
        }
        
        unit = units.get(var, '')
        
        print(f"\n{var.upper().replace('_', ' ')} ({unit}):")
        print(f"  Count:  {stats['count']:,}")
        print(f"  Mean:   {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Range:  [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"  IQR:    [{stats['q25']:.2f}, {stats['q75']:.2f}]")
    
    # Correlations
    print("\n" + "-"*70)
    print("KEY RELATIONSHIPS:")
    print("-"*70)
    
    # Wind-Pressure correlation
    valid_mask = ~(np.isnan(all_stats['vmax']) | np.isnan(all_stats['central_pressure']))
    if np.sum(valid_mask) > 0:
        vmax_valid = all_stats['vmax'][valid_mask[:len(all_stats['vmax'])]]
        pres_valid = all_stats['central_pressure'][valid_mask[:len(all_stats['central_pressure'])]]
        corr = np.corrcoef(vmax_valid, pres_valid)[0, 1]
        print(f"\nWind-Pressure Correlation: {corr:.3f}")
        print(f"  (Should be strongly negative, typically < -0.85)")
    
    # RMW-ROCI relationship
    valid_mask = ~(np.isnan(all_stats['rmw']) | np.isnan(all_stats['roci']))
    if np.sum(valid_mask) > 0:
        rmw_valid = all_stats['rmw'][valid_mask[:len(all_stats['rmw'])]]
        roci_valid = all_stats['roci'][valid_mask[:len(all_stats['roci'])]]
        size_ratio = roci_valid / rmw_valid
        print(f"\nROCI/RMW Ratio: {np.mean(size_ratio):.2f} ± {np.std(size_ratio):.2f}")
        print(f"  (Typical range: 3-8)")
    
    print("\n" + "="*70)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DATASET PARAMETER SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            for var, stats in summary.items():
                unit = units.get(var, '')
                f.write(f"{var.upper().replace('_', ' ')} ({unit}):\n")
                for key, val in stats.items():
                    f.write(f"  {key}: {val}\n")
                f.write("\n")
        
        print(f"\nSummary saved to: {output_file}")
    
    return summary


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TC PARAMETER VERIFICATION AND VISUALIZATION")
    print("="*70)
    
    print("\n1. COMPARE ORIGINAL AND ENHANCED DATASETS")
    print("-"*70)
    print("from verify_params import compare_before_after")
    print("")
    print("new_vars = compare_before_after(")
    print("    original_file='tracks_original.nc',")
    print("    enhanced_file='tracks_with_params.nc'")
    print(")")
    print("")
    
    print("\n2. PLOT A SINGLE TRACK WITH ALL PARAMETERS")
    print("-"*70)
    print("from verify_params import plot_track_with_parameters")
    print("import xarray as xr")
    print("")
    print("ds = xr.open_dataset('tracks_with_params.nc')")
    print("fig = plot_track_with_parameters(")
    print("    ds,")
    print("    year_idx=0,")
    print("    track_idx=0,")
    print("    save_path='track_analysis.png'")
    print(")")
    print("plt.show()")
    print("")
    
    print("\n3. GENERATE DATASET SUMMARY")
    print("-"*70)
    print("from verify_params import generate_dataset_summary")
    print("")
    print("summary = generate_dataset_summary(")
    print("    ds,")
    print("    output_file='dataset_summary.txt'")
    print(")")
    print("")
    
    print("="*70)