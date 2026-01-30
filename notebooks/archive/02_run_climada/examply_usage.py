"""
Complete example: Add TC parameters to tropical_cyclone_risk model output
and create visualizations.
"""

import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import idd_climate_models.constants as rfc
# Import the parameter estimation functions
from idd_climate_models.add_tc_params import add_parameters_to_dataset, add_tc_parameters_to_track
from idd_climate_models.verify_params import plot_track_with_parameters, generate_dataset_summary


# =============================================================================
# STEP 1: Process your file to add parameters
# =============================================================================

print("="*70)
print("STEP 1: ADDING TC PARAMETERS TO YOUR FILE")
print("="*70)

# Your file path
input_file = '/mnt/team/rapidresponse/pub/tropical-storms/tc_risk/output/cmip6/ACCESS-CM2/r1i1p1f1/historical/1970-1989/SI/tracks_SI_ACCESS-CM2_historical_r1i1p1f1_197001_198912_e9.nc'

# Add parameters to the entire file
print("\nProcessing entire file...")
output_file = add_parameters_to_dataset(
    input_file,
    use_ensemble=True,
    env_pressure_method='standard',
    verbose=True
)

print(f"\nEnhanced file created: {output_file}")

# =============================================================================
# STEP 2: Load the enhanced dataset and explore
# =============================================================================

print("\n" + "="*70)
print("STEP 2: EXPLORING THE ENHANCED DATASET")
print("="*70)

# Load the enhanced dataset
ds_enhanced = xr.open_dataset(output_file)

print("\nNew variables available:")
new_vars = [
    'central_pressure',
    'environmental_pressure', 
    'pressure_deficit',
    'rmw',
    'roci'
]

for var in new_vars:
    if var in ds_enhanced:
        print(f"  {var}: {ds_enhanced[var].attrs.get('long_name', 'N/A')}")
        print(f"    Units: {ds_enhanced[var].attrs.get('units', 'N/A')}")

# =============================================================================
# STEP 3: Analyze a specific track (the one you already processed)
# =============================================================================

print("\n" + "="*70)
print("STEP 3: ANALYZING YOUR SPECIFIC TRACK")
print("="*70)

# Select the same track you used before
track = ds_enhanced.isel(year=0, n_trk=0)

# Extract parameters
lon = track.lon_trks.values
lat = track.lat_trks.values
vmax_ms = track.vmax_trks.values
vmax_kt = vmax_ms * 1.943844  # Convert to knots

# New parameters
central_pressure = track.central_pressure.values
env_pressure = track.environmental_pressure.values
pressure_deficit = track.pressure_deficit.values
rmw = track.rmw.values
roci = track.roci.values

# Remove NaN values
valid = ~np.isnan(lat)
print(f"\nTrack has {np.sum(valid)} valid time steps")

# Print some statistics
print("\nTrack Statistics:")
print(f"  Maximum wind speed: {np.nanmax(vmax_kt):.1f} kt")
print(f"  Minimum central pressure: {np.nanmin(central_pressure):.1f} hPa")
print(f"  Maximum pressure deficit: {np.nanmax(pressure_deficit):.1f} hPa")
print(f"  Mean RMW: {np.nanmean(rmw):.1f} nm")
print(f"  Mean ROCI: {np.nanmean(roci):.1f} nm")
print(f"  ROCI/RMW ratio: {np.nanmean(roci/rmw):.2f}")

# =============================================================================
# STEP 4: Create comprehensive visualization
# =============================================================================

print("\n" + "="*70)
print("STEP 4: CREATING VISUALIZATIONS")
print("="*70)

# Create the full analysis plot
fig = plot_track_with_parameters(
    ds_enhanced,
    year_idx=0,  # 1970
    track_idx=0,
    save_path='track_analysis_complete.png'
)

plt.show()

# =============================================================================
# STEP 5: Compare with original process_track output
# =============================================================================

print("\n" + "="*70)
print("STEP 5: COMPARING WITH ORIGINAL ESTIMATION")
print("="*70)

# You can still use the original process_track function for detailed analysis
from idd_climate_models.tc_estimation import process_track

# Get the data for a single track (without NaN)
track_data = track.sel(time=valid)

results = process_track(
    lon_trks=track_data.lon_trks.values,
    lat_trks=track_data.lat_trks.values,
    vmax_trks=track_data.vmax_trks.values,
    u850_trks=track_data.u850_trks.values,
    v850_trks=track_data.v850_trks.values,
    u250_trks=track_data.u250_trks.values,
    v250_trks=track_data.v250_trks.values,
    tc_basins='SI',
    plot=True,
    verbose=True
)

# The results should match the values in the NetCDF file
print("\nVerifying consistency:")
print(f"  NetCDF central pressure range: {np.nanmin(central_pressure):.1f} - {np.nanmax(central_pressure):.1f} hPa")
print(f"  Direct estimation range: {np.min(results['estimates']['central_pressure']['mean']):.1f} - {np.max(results['estimates']['central_pressure']['mean']):.1f} hPa")

# =============================================================================
# STEP 6: Generate dataset-wide summary
# =============================================================================

print("\n" + "="*70)
print("STEP 6: DATASET-WIDE STATISTICS")
print("="*70)

summary = generate_dataset_summary(
    ds_enhanced,
    output_file='dataset_summary.txt'
)

# =============================================================================
# STEP 7: Batch process all files in a directory (optional)
# =============================================================================

print("\n" + "="*70)
print("STEP 7: BATCH PROCESSING (OPTIONAL)")
print("="*70)

print("\nTo process all files in a directory:")
print("```python")
print("from add_tc_params import batch_process_directory")
print("")
print("# Process all tracks in the SI basin")
print("output_files = batch_process_directory(")
print("    input_dir='/mnt/team/rapidresponse/pub/tropical-storms/tc_risk/output/cmip6/ACCESS-CM2/r1i1p1f1/historical/1970-1989/SI/',")
print("    output_dir='./processed_tracks/',")
print("    pattern='tracks_*.nc',")
print("    use_ensemble=True,")
print("    verbose=True")
print(")")
print("```")

# =============================================================================
# STEP 8: Save key parameters for further analysis
# =============================================================================

print("\n" + "="*70)
print("STEP 8: EXTRACTING DATA FOR ANALYSIS")
print("="*70)

# Extract all valid track data
all_tracks_data = []

for year_idx in range(len(ds_enhanced.year)):
    for track_idx in range(len(ds_enhanced.n_trk)):
        track = ds_enhanced.sel(year=year_idx, n_trk=track_idx)
        
        # Check if track has valid data
        if np.any(~np.isnan(track.lat_trks.values)):
            track_dict = {
                'year': ds_enhanced.year.values[year_idx],
                'track_id': track_idx,
                'basin': track.tc_basins.values.item() if 'tc_basins' in track else 'Unknown',
                'max_wind': np.nanmax(track.vmax_trks.values * 1.943844),
                'min_pressure': np.nanmin(track.central_pressure.values),
                'max_deficit': np.nanmax(track.pressure_deficit.values),
                'mean_rmw': np.nanmean(track.rmw.values),
                'mean_roci': np.nanmean(track.roci.values),
                'n_points': np.sum(~np.isnan(track.lat_trks.values))
            }
            all_tracks_data.append(track_dict)

print(f"\nFound {len(all_tracks_data)} valid tracks in dataset")

# Convert to pandas DataFrame for easy analysis
import pandas as pd
df_tracks = pd.DataFrame(all_tracks_data)

print("\nTrack intensity distribution:")
print(df_tracks['max_wind'].describe())

print("\nSaving track summary to CSV...")
df_tracks.to_csv('track_summary.csv', index=False)
print("Saved to: track_summary.csv")

# =============================================================================
# CLEAN UP
# =============================================================================

ds_enhanced.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

print("\nFiles created:")
print("  1. Enhanced NetCDF with TC parameters")
print("  2. track_analysis_complete.png - Full visualization")
print("  3. dataset_summary.txt - Statistical summary")
print("  4. track_summary.csv - Track-by-track data")

print("\nKey variables now available in your NetCDF files:")
print("  - central_pressure: Minimum sea-level pressure (hPa)")
print("  - environmental_pressure: Ambient pressure at storm boundary (hPa)")
print("  - pressure_deficit: P_env - P_central (hPa)")
print("  - rmw: Radius of maximum wind (nautical miles)")
print("  - roci: Radius of outermost closed isobar (nautical miles)")
print("  - *_std: Uncertainty estimates for each parameter")

print("\n" + "="*70)