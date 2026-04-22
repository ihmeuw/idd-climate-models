"""Visualization functions for TC Risk and CLIMADA storm data."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# Type hints for optional dependencies
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from .climada import ClimadaStormData, ClimadaBasinData, STORM_CATEGORIES
from .tc_risk import TCRiskBasinData, parse_basin_coordinate


# Color maps for storm categories
CATEGORY_COLORS = {
    "tropical_depression": "#808080",  # Gray
    "tropical_storm": "#00BFFF",       # Deep sky blue
    "cat1": "#FFFF00",                 # Yellow
    "cat2": "#FFA500",                 # Orange
    "cat3": "#FF0000",                 # Red
    "cat4": "#FF00FF",                 # Magenta
    "cat5": "#8B0000",                 # Dark red
}


def plot_storm_intensity(
    storm: ClimadaStormData,
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    add_colorbar: bool = True,
    title: Optional[str] = None,
    use_cartopy: bool = True,
    zoom_to_basin: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot storm intensity raster.
    
    Args:
        storm: ClimadaStormData object
        ax: Matplotlib axes (created if None)
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        add_colorbar: Whether to add a colorbar
        title: Plot title (auto-generated if None)
        use_cartopy: Whether to use cartopy for map projection
        
    Returns:
        Tuple of (figure, axes)
    """
    intensity = storm.get_intensity_array()
    
    # Create figure if needed
    if ax is None:
        if use_cartopy and HAS_CARTOPY:
            fig, ax = plt.subplots(
                figsize=(12, 8),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Set default vmax based on hurricane threshold
    if vmax is None:
        vmax = max(70, float(intensity.max().values))  # At least Cat 5 threshold
    if vmin is None:
        vmin = 0
    
    # Plot
    if use_cartopy and HAS_CARTOPY:
        im = ax.pcolormesh(
            intensity.lon,
            intensity.lat,
            intensity.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        
        # Zoom to basin bounds - now uses storm.basin_bounds directly
        if zoom_to_basin and storm.basin_bounds:
            west, east, south, north = storm.basin_bounds
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    else:
        im = ax.pcolormesh(
            intensity.lon,
            intensity.lat,
            intensity.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Zoom to basin bounds (non-cartopy)
        if zoom_to_basin and storm.basin_bounds:
            west, east, south, north = storm.basin_bounds
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
    
    # Colorbar
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Wind Speed (m/s)")
    
    # Title
    if title is None:
        title = (
            f"{storm.storm_name} - {storm.basin} - {storm.category_name.replace('_', ' ').title()}\n"
            f"{storm.start_date.strftime('%Y-%m-%d') if storm.start_date else 'Unknown'} | "
            f"Max: {storm.max_intensity:.1f} m/s"
        )
    ax.set_title(title)
    
    return fig, ax


def plot_storm_exposure_hours(
    storm: ClimadaStormData,
    ax: Optional[plt.Axes] = None,
    cmap: str = "Blues",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    add_colorbar: bool = True,
    title: Optional[str] = None,
    use_cartopy: bool = True,
    zoom_to_basin: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot storm exposure hours raster.
    
    Args:
        storm: ClimadaStormData object
        ax: Matplotlib axes (created if None)
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        add_colorbar: Whether to add a colorbar
        title: Plot title (auto-generated if None)
        use_cartopy: Whether to use cartopy for map projection
        zoom_to_basin: Whether to zoom to the storm's basin bounds
    Returns:
        Tuple of (figure, axes)
    """
    exposure = storm.get_exposure_hours_array()
    
    # Squeeze time dimension if present
    if "time" in exposure.dims:
        exposure = exposure.isel(time=0)
    
    # Create figure if needed
    if ax is None:
        if use_cartopy and HAS_CARTOPY:
            fig, ax = plt.subplots(
                figsize=(12, 8),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = float(exposure.max().values)
    
    # Plot
    if use_cartopy and HAS_CARTOPY:
        im = ax.pcolormesh(
            exposure.lon,
            exposure.lat,
            exposure.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        # Zoom to basin bounds (cartopy)
        if zoom_to_basin and storm.basin_bounds:
            west, east, south, north = storm.basin_bounds
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    else:
        im = ax.pcolormesh(
            exposure.lon,
            exposure.lat,
            exposure.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Zoom to basin bounds (non-cartopy)
        if zoom_to_basin and storm.basin_bounds:
            west, east, south, north = storm.basin_bounds
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
    
    # Colorbar
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Exposure Hours (≥17 m/s)")
    
    # Title
    if title is None:
        title = (
            f"{storm.storm_name} Exposure Hours - {storm.basin}\n"
            f"{storm.start_date.strftime('%Y-%m-%d') if storm.start_date else 'Unknown'}"
        )
    ax.set_title(title)
    
    return fig, ax


def plot_storm_dual_panel(
    storm: ClimadaStormData,
    figsize: Tuple[float, float] = (16, 6),
    use_cartopy: bool = True,
    zoom_to_basin: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot intensity and exposure hours side by side.
    
    Args:
        storm: ClimadaStormData object
        figsize: Figure size
        use_cartopy: Whether to use cartopy for map projection
        
    Returns:
        Tuple of (figure, (ax_intensity, ax_exposure))
    """
    if use_cartopy and HAS_CARTOPY:
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    plot_storm_intensity(storm, ax=ax1, use_cartopy=use_cartopy, zoom_to_basin=zoom_to_basin)
    plot_storm_exposure_hours(storm, ax=ax2, use_cartopy=use_cartopy, zoom_to_basin=zoom_to_basin)
    
    fig.suptitle(
        f"{storm.model} {storm.variant} - {storm.scenario} - {storm.time_period}",
        fontsize=12,
        y=1.02,
    )
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_basin_storm_summary(
    basin_data: ClimadaBasinData,
    figsize: Tuple[float, float] = (14, 10),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot summary of storms in a basin: count by year and category distribution.
    
    Args:
        basin_data: ClimadaBasinData object
        figsize: Figure size
        
    Returns:
        Tuple of (figure, axes array)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get storm data
    storms_by_year = basin_data.get_storms_by_year()
    storms_by_category = basin_data.get_storms_by_category()
    
    # Panel 1: Storm count by year
    ax1 = axes[0, 0]
    years = sorted(storms_by_year.keys())
    counts = [len(storms_by_year[y]) for y in years]
    ax1.bar(years, counts, color="steelblue", edgecolor="black")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Storms")
    ax1.set_title("Storms by Year")
    
    # Panel 2: Storm count by category
    ax2 = axes[0, 1]
    categories = list(STORM_CATEGORIES.keys())
    cat_counts = [len(storms_by_category.get(cat, [])) for cat in categories]
    colors = [CATEGORY_COLORS.get(cat, "gray") for cat in categories]
    ax2.bar(
        [c.replace("_", "\n") for c in categories],
        cat_counts,
        color=colors,
        edgecolor="black",
    )
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Number of Storms")
    ax2.set_title("Storms by Category")
    
    # Panel 3: Max intensity distribution
    ax3 = axes[1, 0]
    max_intensities = [storm.max_intensity for storm in basin_data.iter_storms()]
    ax3.hist(max_intensities, bins=20, color="steelblue", edgecolor="black")
    ax3.axvline(33, color="red", linestyle="--", label="Cat 1 threshold")
    ax3.axvline(50, color="darkred", linestyle="--", label="Cat 3 threshold")
    ax3.set_xlabel("Max Wind Speed (m/s)")
    ax3.set_ylabel("Count")
    ax3.set_title("Max Intensity Distribution")
    ax3.legend()
    
    # Panel 4: Hurricanes by year
    ax4 = axes[1, 1]
    hurricane_counts = []
    for year in years:
        hurricanes = [s for s in storms_by_year[year] if s.max_intensity >= 33]
        hurricane_counts.append(len(hurricanes))
    ax4.bar(years, hurricane_counts, color="orangered", edgecolor="black")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Number of Hurricanes")
    ax4.set_title("Hurricanes (Cat 1+) by Year")
    
    fig.suptitle(
        f"{basin_data.basin_name} ({basin_data.basin}) - {basin_data.model} {basin_data.variant}\n"
        f"{basin_data.scenario} - {basin_data.time_period} - Draw {basin_data.draw}",
        fontsize=12,
    )
    
    plt.tight_layout()
    return fig, axes


# =============================================================================
# TC RISK TRACK VISUALIZATION FUNCTIONS
# =============================================================================

try:
    import geopandas as gpd
    from shapely.geometry import box as shapely_box
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


def plot_tc_track(
    basin_data: TCRiskBasinData,
    draw: int,
    track_index: int = 0,
    admin_gdf: Optional["gpd.GeoDataFrame"] = None,
    buffer_deg: float = 5.0,
    windspeed_threshold: float = 34.0,
    min_marker_size: float = 4.0,
    max_marker_size: float = 12.0,
    figsize: Tuple[float, float] = (16, 10),
    show_bboxes: bool = True,
    use_cartopy: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a single TC track from TC Risk output.
    
    Args:
        basin_data: TCRiskBasinData object
        draw: Draw number to visualize
        track_index: Which track/storm to plot (0-indexed)
        admin_gdf: Optional admin boundary GeoDataFrame (clipped to view)
        buffer_deg: Buffer in degrees around data extent
        windspeed_threshold: Wind speed (m/s) above which markers are filled
        min_marker_size: Minimum marker size
        max_marker_size: Maximum marker size
        figsize: Figure size
        show_bboxes: Show bounding boxes for basin and track
        use_cartopy: Use cartopy for map projection
        
    Returns:
        Tuple of (figure, axes)
    """
    if not HAS_CARTOPY and use_cartopy:
        raise ImportError("cartopy is required for map plotting")
    
    # Read track data
    ds = basin_data.read_track_file(draw)
    
    # Extract track data (TC Risk uses lon_trks, lat_trks, vmax_trks)
    track_lon = ds['lon_trks'].isel(n_trk=track_index).values
    track_lat = ds['lat_trks'].isel(n_trk=track_index).values
    track_wind = ds['vmax_trks'].isel(n_trk=track_index).values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(track_lon) | np.isnan(track_lat))
    track_lon = track_lon[valid_mask]
    track_lat = track_lat[valid_mask]
    track_wind = track_wind[valid_mask]
    
    # Convert longitudes from 0-360 to -180/180 format
    track_lon = np.where(track_lon > 180, track_lon - 360, track_lon)
    
    ds.close()
    
    # Get basin bounds
    basin_bounds = basin_data.basin_bounds
    if basin_bounds:
        basin_lon_min, basin_lon_max, basin_lat_min, basin_lat_max = basin_bounds
    else:
        basin_lon_min = track_lon.min() - 10
        basin_lon_max = track_lon.max() + 10
        basin_lat_min = track_lat.min() - 10
        basin_lat_max = track_lat.max() + 10
    
    # Calculate view extent based on track, not full basin (basin can be huge)
    view_lon_min = max(track_lon.min() - buffer_deg, -180)
    view_lon_max = min(track_lon.max() + buffer_deg, 180)
    view_lat_min = max(track_lat.min() - buffer_deg, -90)
    view_lat_max = min(track_lat.max() + buffer_deg, 90)
    
    # Create figure
    if use_cartopy:
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        ax.set_extent([view_lon_min, view_lon_max, view_lat_min, view_lat_max], 
                      crs=ccrs.PlateCarree())
        
        # Add background features
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.coastlines(resolution='50m', linewidth=0.8)
        
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(view_lon_min, view_lon_max)
        ax.set_ylim(view_lat_min, view_lat_max)
        transform = None
    
    # Plot admin boundaries if provided
    if admin_gdf is not None and HAS_GEOPANDAS:
        bbox = shapely_box(view_lon_min, view_lat_min, view_lon_max, view_lat_max)
        admin_clipped = admin_gdf.clip(bbox)
        
        if use_cartopy:
            admin_clipped.boundary.plot(ax=ax, transform=transform,
                                        linewidth=1.0, edgecolor='darkgreen', alpha=0.8)
        else:
            admin_clipped.boundary.plot(ax=ax, linewidth=1.0, edgecolor='darkgreen', alpha=0.8)
    
    # Plot bounding boxes
    if show_bboxes and HAS_GEOPANDAS:
        # Basin bounds
        if basin_bounds:
            basin_rect = shapely_box(basin_lon_min, basin_lat_min, basin_lon_max, basin_lat_max)
            if use_cartopy:
                ax.add_geometries([basin_rect], crs=ccrs.PlateCarree(),
                                  facecolor='none', edgecolor='blue', linewidth=2, linestyle=':')
            
        # Track bounds
        track_rect = shapely_box(track_lon.min(), track_lat.min(), track_lon.max(), track_lat.max())
        if use_cartopy:
            ax.add_geometries([track_rect], crs=ccrs.PlateCarree(),
                              facecolor='none', edgecolor='orange', linewidth=2, linestyle='-.')
    
    # Draw track line
    plot_kwargs = {'transform': transform} if use_cartopy else {}
    ax.plot(track_lon, track_lat, '-', color='darkred', linewidth=2, alpha=0.6, 
            zorder=10, **plot_kwargs)
    
    # Scale marker sizes based on windspeed
    if len(track_wind) > 0 and track_wind.max() > track_wind.min():
        wind_norm = (track_wind - track_wind.min()) / (track_wind.max() - track_wind.min())
        marker_sizes = min_marker_size + wind_norm * (max_marker_size - min_marker_size)
    else:
        marker_sizes = np.full_like(track_wind, (min_marker_size + max_marker_size) / 2)
    
    # Plot track points
    for lon, lat, wind, size in zip(track_lon, track_lat, track_wind, marker_sizes):
        if wind >= windspeed_threshold:
            ax.plot(lon, lat, 'o', color='darkred', markersize=size,
                    markeredgewidth=0.5, markeredgecolor='black', zorder=11, **plot_kwargs)
        else:
            ax.plot(lon, lat, 'o', markerfacecolor='none', markeredgecolor='darkred',
                    markersize=size, markeredgewidth=1.5, zorder=11, **plot_kwargs)
    
    # Mark start and end
    if len(track_lon) > 0:
        ax.plot(track_lon[0], track_lat[0], 'g*', markersize=15, zorder=12, **plot_kwargs)
        ax.plot(track_lon[-1], track_lat[-1], 'r^', markersize=12, zorder=12, **plot_kwargs)
    
    # Gridlines
    if use_cartopy:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Title
    ax.set_title(
        f"{basin_data.basin_name} - {basin_data.model}/{basin_data.variant}\n"
        f"{basin_data.scenario} - {basin_data.time_period} - Draw {draw} - Track {track_index}",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig, ax


def plot_tc_track_dual_panel(
    basin_data: TCRiskBasinData,
    draw: int,
    track_index: int = 0,
    admin_gdf: Optional["gpd.GeoDataFrame"] = None,
    figsize: Tuple[float, float] = (20, 8),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot TC track with map and wind speed time series side by side.
    
    Args:
        basin_data: TCRiskBasinData object
        draw: Draw number
        track_index: Track index
        admin_gdf: Optional admin boundaries
        figsize: Figure size
        
    Returns:
        Tuple of (figure, (map_ax, timeseries_ax))
    """
    if not HAS_CARTOPY:
        raise ImportError("cartopy is required for this plot")
    
    # Read track data
    track_data = basin_data.get_track_data(draw, track_index)
    lon, lat, wind = track_data['lon'], track_data['lat'], track_data['wind']
    
    # Calculate extent from track data (not basin bounds - they can be huge)
    buffer = 5.0
    west = max(lon.min() - buffer, -180)
    east = min(lon.max() + buffer, 180)
    south = max(lat.min() - buffer, -90)
    north = min(lat.max() + buffer, 90)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Map panel
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax1.coastlines(resolution='50m', linewidth=0.8)
    
    # Plot admin if provided
    if admin_gdf is not None and HAS_GEOPANDAS:
        bbox = shapely_box(west, south, east, north)
        admin_clipped = admin_gdf.clip(bbox)
        admin_clipped.boundary.plot(ax=ax1, transform=ccrs.PlateCarree(),
                                    linewidth=0.8, edgecolor='darkgreen', alpha=0.6)
    
    # Plot track colored by wind speed
    # (lon, lat, wind already extracted above)
    
    # Draw line
    ax1.plot(lon, lat, '-', color='gray', linewidth=1, alpha=0.5, 
             transform=ccrs.PlateCarree(), zorder=10)
    
    # Color points by wind speed
    scatter = ax1.scatter(lon, lat, c=wind, cmap='YlOrRd', s=30,
                          vmin=0, vmax=max(70, wind.max()),
                          transform=ccrs.PlateCarree(), zorder=11,
                          edgecolors='black', linewidths=0.3)
    plt.colorbar(scatter, ax=ax1, shrink=0.8, label='Wind Speed (m/s)')
    
    # Start/end markers
    ax1.plot(lon[0], lat[0], 'g*', markersize=15, transform=ccrs.PlateCarree(), zorder=12)
    ax1.plot(lon[-1], lat[-1], 'r^', markersize=12, transform=ccrs.PlateCarree(), zorder=12)
    
    gl = ax1.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    ax1.set_title("Track Map", fontsize=12)
    
    # Time series panel
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Plot wind speed over track points
    points = np.arange(len(wind))
    ax2.plot(points, wind, '-o', color='darkred', markersize=4, linewidth=1.5)
    
    # Add category thresholds
    ax2.axhline(17, color='blue', linestyle='--', alpha=0.5, label='Tropical Storm (17 m/s)')
    ax2.axhline(33, color='orange', linestyle='--', alpha=0.5, label='Cat 1 (33 m/s)')
    ax2.axhline(50, color='red', linestyle='--', alpha=0.5, label='Cat 3 (50 m/s)')
    ax2.axhline(70, color='darkred', linestyle='--', alpha=0.5, label='Cat 5 (70 m/s)')
    
    ax2.set_xlabel("Track Point")
    ax2.set_ylabel("Max Wind Speed (m/s)")
    ax2.set_title("Wind Speed Along Track", fontsize=12)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Figure title
    fig.suptitle(
        f"{basin_data.basin_name} - {basin_data.model}/{basin_data.variant} - "
        f"{basin_data.scenario} - {basin_data.time_period}\nDraw {draw}, Track {track_index}",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_draw_summary(
    basin_data: TCRiskBasinData,
    draw: int,
    figsize: Tuple[float, float] = (14, 10),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot summary of all storms in a single draw.
    
    Args:
        basin_data: TCRiskBasinData object
        draw: Draw number
        figsize: Figure size
        
    Returns:
        Tuple of (figure, axes)
    """
    ds = basin_data.read_track_file(draw)
    n_storms = ds.dims.get('n_trk', 0)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Collect stats for all storms
    max_winds = []
    track_lengths = []
    
    for i in range(n_storms):
        wind = ds['vmax_trks'].isel(n_trk=i).values
        lon = ds['lon_trks'].isel(n_trk=i).values
        
        valid = ~np.isnan(wind)
        if valid.any():
            max_winds.append(wind[valid].max())
            track_lengths.append(valid.sum())
    
    ds.close()
    
    max_winds = np.array(max_winds)
    track_lengths = np.array(track_lengths)
    
    # Panel 1: Max wind distribution
    ax1 = axes[0, 0]
    ax1.hist(max_winds, bins=20, color='steelblue', edgecolor='black')
    ax1.axvline(33, color='red', linestyle='--', label='Cat 1 (33 m/s)')
    ax1.axvline(50, color='darkred', linestyle='--', label='Cat 3 (50 m/s)')
    ax1.set_xlabel("Max Wind Speed (m/s)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Max Wind Distribution (n={n_storms})")
    ax1.legend()
    
    # Panel 2: Category breakdown
    ax2 = axes[0, 1]
    categories = ['TD\n(<17)', 'TS\n(17-33)', 'Cat 1-2\n(33-50)', 'Cat 3-5\n(≥50)']
    counts = [
        np.sum(max_winds < 17),
        np.sum((max_winds >= 17) & (max_winds < 33)),
        np.sum((max_winds >= 33) & (max_winds < 50)),
        np.sum(max_winds >= 50),
    ]
    colors = ['gray', 'blue', 'orange', 'red']
    ax2.bar(categories, counts, color=colors, edgecolor='black')
    ax2.set_ylabel("Count")
    ax2.set_title("Storms by Category")
    
    # Panel 3: Track length distribution
    ax3 = axes[1, 0]
    ax3.hist(track_lengths, bins=20, color='green', edgecolor='black')
    ax3.set_xlabel("Track Length (points)")
    ax3.set_ylabel("Count")
    ax3.set_title("Track Length Distribution")
    
    # Panel 4: Wind vs Track Length
    ax4 = axes[1, 1]
    ax4.scatter(track_lengths, max_winds, alpha=0.6, c='steelblue', edgecolors='black', linewidths=0.3)
    ax4.set_xlabel("Track Length (points)")
    ax4.set_ylabel("Max Wind Speed (m/s)")
    ax4.set_title("Max Wind vs Track Length")
    ax4.axhline(33, color='red', linestyle='--', alpha=0.5)
    
    fig.suptitle(
        f"{basin_data.basin_name} - {basin_data.model}/{basin_data.variant}\n"
        f"{basin_data.scenario} - {basin_data.time_period} - Draw {draw}",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig, axes


def plot_all_tracks(
    basin_data: TCRiskBasinData,
    draw: int,
    start_date: Optional[Tuple[int, int]] = None,
    end_date: Optional[Tuple[int, int]] = None,
    basin_filter: Optional[str] = None,
    admin_gdf: Optional["gpd.GeoDataFrame"] = None,
    country_filter: Optional[Union[str, List[str]]] = None,
    admin_code_col: str = "ADM0_CODE",
    admin_name_col: str = "ADM0_NAME",
    figsize: Tuple[float, float] = (18, 10),
    use_cartopy: bool = True,
    wind_cmap: str = "YlOrRd",
    wind_vmin: float = 0,
    wind_vmax: float = 70,
    track_alpha: float = 0.6,
    point_size: float = 8,
) -> Tuple[plt.Figure, plt.Axes, "pd.DataFrame"]:
    """
    Plot all storm tracks from a draw on a single map with flexible filtering.

    Filters:
        - Date range: keep only storms whose genesis (tc_years, tc_month)
          falls within [start_date, end_date].
        - Basin: keep only storms whose tc_basins value matches basin_filter.
          "GL" or None means no basin filter.
        - Country: requires admin_gdf.
          * None  -> plot all storms (no country filter)
          * "any" -> plot only storms that intersect any admin polygon
          * list of country codes -> plot only storms that intersect those countries

    Args:
        basin_data: TCRiskBasinData used to load the track file.
        draw: Draw number.
        start_date: (year, month) inclusive start, or None for no lower bound.
        end_date: (year, month) inclusive end, or None for no upper bound.
        basin_filter: Basin code to keep (e.g. "EP"). None or "GL" keeps all.
        admin_gdf: GeoDataFrame of admin boundaries (needed for country_filter
            and for drawing borders on the map).
        country_filter: None (all storms), "any" (storms hitting any country),
            or a list of admin codes (storms hitting those countries).
        admin_code_col: Column in admin_gdf that holds country codes.
        admin_name_col: Column in admin_gdf that holds country names.
        figsize: Figure size.
        use_cartopy: Use cartopy projection.
        wind_cmap: Colormap for wind speed.
        wind_vmin: Min wind for colormap.
        wind_vmax: Max wind for colormap.
        track_alpha: Alpha for track lines.
        point_size: Marker size for track points.

    Returns:
        (fig, ax, track_info) where track_info is a DataFrame summarising each
        plotted track (index, year, month, basin, max_wind, and optionally
        countries_hit).
    """
    import pandas as pd

    if use_cartopy and not HAS_CARTOPY:
        raise ImportError("cartopy is required for map plotting")
    if country_filter is not None and admin_gdf is None:
        raise ValueError("admin_gdf is required when country_filter is set")

    # Normalise country_filter
    if isinstance(country_filter, str) and country_filter.lower() != "any":
        country_filter = [country_filter]

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    ds = basin_data.read_track_file(draw)
    n_storms = ds.sizes.get("n_trk", 0)

    lon_all = ds["lon_trks"].values          # (n_trk, n_pnt)
    lat_all = ds["lat_trks"].values
    wind_all = ds["vmax_trks"].values
    basins = ds["tc_basins"].values           # (n_trk,)
    years = ds["tc_years"].values.astype(int)
    months = ds["tc_month"].values.astype(int)
    ds.close()

    # Defer longitude conversion until we know which tracks survive filters.
    # lon_all stays in its native space (typically 0-360) for now.

    # ------------------------------------------------------------------
    # Filter: date range
    # ------------------------------------------------------------------
    keep = np.ones(n_storms, dtype=bool)

    if start_date is not None:
        sy, sm = start_date
        keep &= (years > sy) | ((years == sy) & (months >= sm))
    if end_date is not None:
        ey, em = end_date
        keep &= (years < ey) | ((years == ey) & (months <= em))

    # ------------------------------------------------------------------
    # Filter: basin
    # ------------------------------------------------------------------
    if basin_filter is not None and basin_filter.upper() != "GL":
        keep &= np.array([b.strip() == basin_filter for b in basins])

    candidate_indices = np.where(keep)[0]

    # ------------------------------------------------------------------
    # Filter: country intersection
    # ------------------------------------------------------------------
    track_countries: Dict[int, List[str]] = {}

    if country_filter is not None and HAS_GEOPANDAS:
        surviving = []
        for idx in candidate_indices:
            lon = lon_all[idx]
            lat = lat_all[idx]
            wind = wind_all[idx]
            valid = ~(np.isnan(lon) | np.isnan(lat))
            if not valid.any():
                continue

            # Spatial join needs -180/180 to match shapefile CRS
            lon_180 = np.where(lon[valid] > 180, lon[valid] - 360, lon[valid])
            pts = gpd.GeoDataFrame(
                {"wind": wind[valid]},
                geometry=gpd.points_from_xy(lon_180, lat[valid]),
                crs="EPSG:4326",
            )
            joined = gpd.sjoin(pts, admin_gdf[[admin_code_col, "geometry"]],
                               how="inner", predicate="within")
            if joined.empty:
                continue

            hit_codes = joined[admin_code_col].unique().tolist()
            if isinstance(country_filter, list):
                if not any(c in country_filter for c in hit_codes):
                    continue
            track_countries[idx] = hit_codes
            surviving.append(idx)

        candidate_indices = np.array(surviving, dtype=int)
    else:
        candidate_indices = np.asarray(candidate_indices)

    # ------------------------------------------------------------------
    # Compute view extent and choose map centre
    # ------------------------------------------------------------------
    if len(candidate_indices) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No tracks match the filters",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        return fig, ax, pd.DataFrame()

    # Always convert to -180/180 first
    lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)

    all_lon = np.concatenate([
        lon_all[i][~np.isnan(lon_all[i])] for i in candidate_indices
    ])
    all_lat = np.concatenate([
        lat_all[i][~np.isnan(lat_all[i])] for i in candidate_indices
    ])

    # Choose coordinate space by basin, not by data heuristic:
    #   NA → -180/180, zoom to storm bbox
    #   GL → -180/180, fixed world extent
    #   everything else → 0-360, zoom to storm bbox
    effective_basin = basin_data.basin
    if basin_filter and basin_filter.upper() != "GL":
        effective_basin = basin_filter.upper()

    use_360 = effective_basin not in ("NA", "GL")

    if use_360:
        lon_all = np.where(lon_all < 0, lon_all + 360, lon_all)
        all_lon = np.where(all_lon < 0, all_lon + 360, all_lon)

    buffer = 5.0
    if effective_basin == "GL":
        west, east, south, north = -180.0, 180.0, -90.0, 90.0
    else:
        west = float(all_lon.min() - buffer)
        east = float(all_lon.max() + buffer)
        south = max(float(all_lat.min() - buffer), -90.0)
        north = min(float(all_lat.max() + buffer), 90.0)

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    central_longitude = 180.0 if use_360 else 0.0

    if use_cartopy:
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
        data_crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
        ax.set_extent([west - central_longitude, east - central_longitude, south, north], crs=proj)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        ax.coastlines(resolution="50m", linewidth=0.8)
        transform = data_crs
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        transform = None

    plot_kw: Dict[str, Any] = {}
    if transform is not None:
        plot_kw["transform"] = transform

    # Admin borders
    if admin_gdf is not None and HAS_GEOPANDAS:
        if use_360:
            from .geo_utils import reproject_gdf_to_360
            admin_plot = reproject_gdf_to_360(admin_gdf)
        else:
            admin_plot = admin_gdf
        bbox = shapely_box(west, south, east, north)
        admin_clipped = admin_plot.clip(bbox)
        admin_clipped.boundary.plot(
            ax=ax, linewidth=0.8, edgecolor="darkgreen", alpha=0.6,
            **({"transform": transform} if transform else {}),
        )

    # ------------------------------------------------------------------
    # Plot tracks
    # ------------------------------------------------------------------
    cmap = plt.get_cmap(wind_cmap)
    norm = plt.Normalize(vmin=wind_vmin, vmax=wind_vmax)

    for idx in candidate_indices:
        lon = lon_all[idx]
        lat = lat_all[idx]
        wind = wind_all[idx]
        valid = ~(np.isnan(lon) | np.isnan(lat))
        if not valid.any():
            continue

        lon_v, lat_v, wind_v = lon[valid], lat[valid], wind[valid]

        ax.plot(lon_v, lat_v, "-", color="gray", linewidth=0.5,
                alpha=track_alpha * 0.5, zorder=5, **plot_kw)
        ax.scatter(lon_v, lat_v, c=wind_v, cmap=cmap, norm=norm,
                   s=point_size, edgecolors="none", zorder=6, **plot_kw)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.8, label="Max Wind Speed (m/s)")

    if use_cartopy:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    n_plotted = len(candidate_indices)
    date_str = ""
    if start_date:
        date_str += f" from {start_date[0]}-{start_date[1]:02d}"
    if end_date:
        date_str += f" to {end_date[0]}-{end_date[1]:02d}"

    country_str = ""
    if country_filter == "any":
        country_str = ", hitting any country"
    elif isinstance(country_filter, list):
        country_str = f", hitting {country_filter}"

    basin_str = basin_filter if (basin_filter and basin_filter != "GL") else "all basins"
    ax.set_title(
        f"{basin_data.model}/{basin_data.variant} - {basin_data.scenario} - "
        f"{basin_data.time_period} - Draw {draw}\n"
        f"{n_plotted} tracks in {basin_str}{date_str}{country_str}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Build summary DataFrame
    # ------------------------------------------------------------------
    rows = []
    for idx in candidate_indices:
        wind = wind_all[idx]
        valid_wind = wind[~np.isnan(wind)]
        row = {
            "track_index": int(idx),
            "year": int(years[idx]),
            "month": int(months[idx]),
            "basin": basins[idx].strip() if isinstance(basins[idx], str) else basins[idx],
            "max_wind": float(valid_wind.max()) if len(valid_wind) > 0 else np.nan,
            "n_points": int((~np.isnan(lon_all[idx])).sum()),
        }
        if idx in track_countries:
            row["countries_hit"] = track_countries[idx]
        rows.append(row)

    track_info = pd.DataFrame(rows)
    return fig, ax, track_info