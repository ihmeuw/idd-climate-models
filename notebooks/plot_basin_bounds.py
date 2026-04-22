#!/ihme/homes/bcreiner/miniconda/envs/idd-climate-models/bin/python
"""
Plot global map with basin boundaries.

This script creates a visualization of the tropical cyclone basin boundaries
on a global map with colored oceans.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from idd_climate_models.constants import basin_dict


def parse_coordinate(coord_str):
    """
    Parse coordinate string like '180E', '45S', '0N' to numeric value.
    
    Parameters
    ----------
    coord_str : str
        Coordinate string with direction (E/W for longitude, N/S for latitude)
    
    Returns
    -------
    float
        Numeric coordinate value (positive East/North, negative West/South)
    """
    value = float(coord_str[:-1])
    direction = coord_str[-1]
    
    if direction in ['W', 'S']:
        value = -value
    
    return value


def get_basin_coords(basin_bounds):
    """
    Convert basin bounds from string format to numeric coordinates.
    
    Parameters
    ----------
    basin_bounds : list
        List of 4 strings: [lon_west, lat_south, lon_east, lat_north]
    
    Returns
    -------
    tuple
        (lon_min, lon_max, lat_min, lat_max)
    """
    lon_min = parse_coordinate(basin_bounds[0])
    lat_min = parse_coordinate(basin_bounds[1])
    lon_max = parse_coordinate(basin_bounds[2])
    lat_max = parse_coordinate(basin_bounds[3])
    
    return lon_min, lon_max, lat_min, lat_max


def plot_basin_boundaries(width=20, lat_range=(-50, 50), save_path=None):
    """
    Create a global map showing all basin boundaries.
    
    Parameters
    ----------
    width : float, optional
        Width of the figure in inches (default: 20)
    lat_range : tuple, optional
        Latitude range to display as (lat_min, lat_max) (default: (-50, 50))
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot.
    """
    # Define colors for each basin (excluding GL which is global)
    basin_colors = {
        'EP': '#FF6B6B',      # Coral red
        'NA': '#4ECDC4',      # Turquoise
        'NI': '#FF8C00',      # Dark orange
        'SI': '#95E1D3',      # Mint
        'AU': '#F38181',      # Light coral
        'SP': '#AA96DA',      # Purple
        'WP': '#FCBAD3',      # Pink
    }
    
    # Calculate dimensions for proper aspect ratio
    lon_span = 360  # Full longitude range
    lat_span = lat_range[1] - lat_range[0]
    map_aspect_ratio = lon_span / lat_span  # width/height in degrees
    
    # Define fixed heights for title and legend (in inches)
    title_height = 0.8
    legend_height = 1.2
    
    # Calculate map height based on width and aspect ratio
    map_height = width / map_aspect_ratio
    
    # Total figure height
    total_height = title_height + map_height + legend_height
    
    # Create figure with calculated dimensions
    fig = plt.figure(figsize=(width, total_height))
    
    # Create GridSpec with explicit height ratios
    gs = gridspec.GridSpec(3, 1, figure=fig, 
                          height_ratios=[title_height, map_height, legend_height],
                          hspace=0.05)
    
    # Create title axis
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Tropical Cyclone Basin Boundaries', 
                 ha='center', va='center', fontsize=18, fontweight='bold',
                 transform=ax_title.transAxes)
    
    # Create map axis with PlateCarree projection
    ax = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
    
    # Set the extent based on lat_range
    ax.set_extent([-180, 180, lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    
    # Add ocean color
    ax.add_feature(cfeature.OCEAN, facecolor='#A6B6DC', zorder=0)
    
    # Add land with a light color
    ax.add_feature(cfeature.LAND, facecolor='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5, zorder=1)
    
    # Add coastlines
    ax.coastlines(resolution='110m', linewidth=0.8, color='#666666', zorder=2)
    
    # Add country borders
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#999999', zorder=2)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot each basin boundary (skip GL as it's the global extent)
    for basin_code, basin_info in basin_dict.items():
        if basin_code == 'GL':
            continue
            
        basin_bounds = basin_info['basin_bounds']
        lon_min, lon_max, lat_min, lat_max = get_basin_coords(basin_bounds)
        
        # Handle longitude wrapping (e.g., for basins crossing the dateline)
        width_deg = lon_max - lon_min
        if width_deg < 0:  # Crosses prime meridian or dateline
            width_deg += 360
        
        height_deg = lat_max - lat_min
        
        # Get color for this basin
        color = basin_colors.get(basin_code, '#999999')
        
        # Draw rectangle for basin boundary
        rect = Rectangle(
            (lon_min, lat_min),
            width_deg,
            height_deg,
            linewidth=3,
            edgecolor=color,
            facecolor='none',
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            zorder=4
        )
        ax.add_patch(rect)
        
        # Add label at the center of the basin
        center_lon = lon_min + width_deg / 2
        center_lat = lat_min + height_deg / 2
        
        # Handle longitude wrapping for label position
        if center_lon > 180:
            center_lon -= 360
        
        # Only add label if basin center is within the lat_range
        if lat_range[0] <= center_lat <= lat_range[1]:
            ax.text(
                center_lon,
                center_lat,
                f"{basin_code}\n{basin_info['name']}",
                transform=ccrs.PlateCarree(),
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold',
                color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, alpha=0.8, linewidth=2),
                zorder=5
            )
    
    # Create legend axis
    ax_legend = fig.add_subplot(gs[2])
    ax_legend.axis('off')
    
    # Create legend elements
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor=basin_colors[code], 
                      label=f"{code}: {basin_dict[code]['name']}", linewidth=3)
        for code in basin_colors.keys()
    ]
    
    # Add legend to the legend axis
    legend = ax_legend.legend(handles=legend_elements, loc='upper center', 
                             ncol=4, fontsize=10, framealpha=0.9,
                             bbox_to_anchor=(0.5, 1.0))
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax


if __name__ == '__main__':
    # You can specify a save path or leave it None to display
    # Customize width and lat_range as needed
    output_path = Path(__file__).parent / 'basin_boundaries_map.png'
    plot_basin_boundaries(width=20, lat_range=(-50, 50), save_path=output_path)
