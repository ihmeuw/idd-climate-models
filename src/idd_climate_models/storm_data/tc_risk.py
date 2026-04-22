"""TC Risk model output data classes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from idd_climate_models.constants import basin_dict, NUM_DRAWS
from .base import ModelVariant


def parse_basin_coordinate(coord_str: str) -> float:
    """
    Convert coordinate string like '260E' or '45S' to numeric value.
    
    Args:
        coord_str: Coordinate string with direction suffix (e.g., '260E', '45S')
        
    Returns:
        Numeric coordinate value (-180 to 180 for lon, -90 to 90 for lat)
    """
    coord_str = coord_str.strip()
    value = float(coord_str[:-1])
    direction = coord_str[-1].upper()
    
    if direction in ['W', 'S']:
        value = -value
    
    # Convert longitude from 0-360 to -180-180 if needed
    # Use >= 180 to handle the date line (180E should become -180)
    if direction in ['E', 'W'] and value >= 180:
        value = value - 360
    
    return value


@dataclass
class TCRiskBasinData:
    """
    Manages TC Risk output data for a single basin within a model/variant/scenario/time_period.
    
    Handles path building, data loading, and draw management for TC Risk netCDF outputs.
    """
    
    model: str
    variant: str
    scenario: str
    time_period: str
    basin: str
    base_path: Path = field(default=Path("/mnt/team/rapidresponse/pub/tropical-storms/tc-risk/output"))
    
    # Loaded data (populated lazily)
    _exposure_ds: Optional[xr.Dataset] = field(default=None, repr=False)
    _intensity_ds: Optional[xr.Dataset] = field(default=None, repr=False)
    _frequency_ds: Optional[xr.Dataset] = field(default=None, repr=False)
    _impact_ds: Optional[xr.Dataset] = field(default=None, repr=False)
    
    # Current draw selection
    _current_draw: Optional[int] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Convert base_path to Path if string."""
        self.base_path = Path(self.base_path)
    
    @property
    def basin_name(self) -> str:
        """Return full basin name from basin code."""
        basin_info = basin_dict.get(self.basin, {})
        if isinstance(basin_info, dict):
            return basin_info.get('name', self.basin)
        return basin_info if basin_info else self.basin
    
    @property
    def data_dir(self) -> Path:
        """Return the directory containing data files for this configuration."""
        return self.base_path / self.model / self.variant / self.scenario / self.time_period / self.basin
    
    def _build_filename(self, data_type: str, draw: Optional[int] = None) -> str:
        """Build filename for a specific data type and draw."""
        start_year = self.time_period.split('-')[0]
        end_year = self.time_period.split('-')[1]
        
        draw_suffix = f"_e{draw - 1}" if draw and draw > 0 else ""
        
        return f"{data_type}_{self.basin}_{self.model}_{self.scenario}_{self.variant}_{start_year}01_{end_year}12{draw_suffix}.nc"
    
    def get_path(self, data_type: str, draw: Optional[int] = None) -> Path:
        """Get full path to a data file."""
        filename = self._build_filename(data_type, draw)
        return self.data_dir / data_type / filename
    
    def list_available_draws(self, data_type: str = "intensity") -> List[int]:
        """List available draw numbers for a data type."""
        data_dir = self.data_dir / data_type
        if not data_dir.exists():
            return []
        
        draws = [0]  # Draw 0 has no suffix
        for f in data_dir.glob(f"{data_type}_{self.basin}_*_e*.nc"):
            # Extract draw number from _eN suffix
            stem = f.stem
            if "_e" in stem:
                try:
                    draw_num = int(stem.split("_e")[-1]) + 1
                    draws.append(draw_num)
                except ValueError:
                    pass
        
        return sorted(draws)
    
    def load_data(self, data_type: str, draw: Optional[int] = None) -> xr.Dataset:
        """Load a specific data type, optionally for a specific draw."""
        path = self.get_path(data_type, draw)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return xr.open_dataset(path)
    
    def set_draw(self, draw: int) -> None:
        """Set the current draw and clear cached data."""
        self._current_draw = draw
        self._exposure_ds = None
        self._intensity_ds = None
        self._frequency_ds = None
        self._impact_ds = None
    
    def get_exposure(self, draw: Optional[int] = None) -> xr.Dataset:
        """Get exposure data, using cached version if available."""
        draw = draw if draw is not None else self._current_draw
        if self._exposure_ds is None or draw != self._current_draw:
            self._exposure_ds = self.load_data("exposure", draw)
            self._current_draw = draw
        return self._exposure_ds
    
    def get_intensity(self, draw: Optional[int] = None) -> xr.Dataset:
        """Get intensity data, using cached version if available."""
        draw = draw if draw is not None else self._current_draw
        if self._intensity_ds is None or draw != self._current_draw:
            self._intensity_ds = self.load_data("intensity", draw)
            self._current_draw = draw
        return self._intensity_ds
    
    def get_frequency(self, draw: Optional[int] = None) -> xr.Dataset:
        """Get frequency data, using cached version if available."""
        draw = draw if draw is not None else self._current_draw
        if self._frequency_ds is None or draw != self._current_draw:
            self._frequency_ds = self.load_data("frequency", draw)
            self._current_draw = draw
        return self._frequency_ds
    
    def get_impact(self, draw: Optional[int] = None) -> xr.Dataset:
        """Get impact data, using cached version if available."""
        draw = draw if draw is not None else self._current_draw
        if self._impact_ds is None or draw != self._current_draw:
            self._impact_ds = self.load_data("impact", draw)
            self._current_draw = draw
        return self._impact_ds
    
    # =========================================================================
    # Basin bounds property
    # =========================================================================
    
    @property
    def basin_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Return basin bounds as (west, east, south, north) floats.
        
        Parses the string format from basin_dict and converts to numeric values.
        """
        basin_info = basin_dict.get(self.basin, {})
        if not isinstance(basin_info, dict):
            return None
            
        raw_bounds = basin_info.get('basin_bounds')
        if not raw_bounds or len(raw_bounds) != 4:
            return None
        
        west = parse_basin_coordinate(raw_bounds[0])
        south = parse_basin_coordinate(raw_bounds[1])
        east = parse_basin_coordinate(raw_bounds[2])
        north = parse_basin_coordinate(raw_bounds[3])
        
        return (west, east, south, north)
    
    @property
    def basin_bounds_raw(self) -> Optional[List[str]]:
        """Return raw basin bounds strings from basin_dict."""
        basin_info = basin_dict.get(self.basin, {})
        if isinstance(basin_info, dict):
            return basin_info.get('basin_bounds')
        return None
    
    # =========================================================================
    # Track file methods (TC Risk output track files)
    # =========================================================================
    
    @property 
    def track_base_path(self) -> Path:
        """Base path for track files (may differ from intensity/exposure path)."""
        return Path("/mnt/team/rapidresponse/pub/tropical-storms/tc_risk/output/cmip6")
    
    @property
    def track_dir(self) -> Path:
        """Directory containing track files for this configuration."""
        return self.track_base_path / self.model / self.variant / self.scenario / self.time_period / self.basin
    
    def _build_track_filename(self, draw: int = 0) -> str:
        """Build track filename for a specific draw."""
        start_year = self.time_period.split('-')[0]
        end_year = self.time_period.split('-')[1]
        
        # Draw 0 has no suffix, others have _eN where N = draw
        draw_suffix = f"_e{draw}" if draw > 0 else ""
        
        return f"tracks_{self.basin}_{self.model}_{self.scenario}_{self.variant}_{start_year}01_{end_year}12{draw_suffix}.nc"
    
    def get_track_path(self, draw: int = 0) -> Path:
        """Get full path to a track file for a specific draw."""
        filename = self._build_track_filename(draw)
        return self.track_dir / filename
    
    def check_track_exists(self, draw: int = 0) -> bool:
        """Check if a track file exists for a specific draw.

        For GL with no global file, returns True if any constituent basin
        has a track file for the given draw.
        """
        if self.get_track_path(draw).exists():
            return True
        if self.basin != "GL":
            return False
        return any(
            TCRiskBasinData(
                model=self.model, variant=self.variant,
                scenario=self.scenario, time_period=self.time_period,
                basin=code, base_path=self.base_path,
            ).get_track_path(draw).exists()
            for code, info in basin_dict.items()
            if isinstance(info, dict) and info.get("most_detailed", False)
        )
    
    def list_available_track_draws(self, max_draws: Optional[int] = None) -> List[int]:
        """
        List available draw numbers that have track files.
        
        Args:
            max_draws: Maximum number of draws to check (default: NUM_DRAWS from constants)
            
        Returns:
            List of available draw numbers (0-indexed)
        """
        if max_draws is None:
            max_draws = NUM_DRAWS
        
        available = []
        for draw in range(max_draws):
            if self.check_track_exists(draw):
                available.append(draw)
        
        return available
    
    def read_track_file(self, draw: int = 0) -> xr.Dataset:
        """
        Read TC risk track file for a specific draw.

        If basin is 'GL' and no global file exists, concatenates all
        most_detailed basins along n_trk. Basins with different max track
        lengths are NaN-padded via join='outer'.
        """
        file_path = self.get_track_path(draw)

        if self.basin != "GL" or file_path.exists():
            if not file_path.exists():
                raise FileNotFoundError(f"Track file not found: {file_path}")
            return xr.open_dataset(file_path)

        # GL with no global file: concatenate constituent basins
        constituent_basins = [
            code for code, info in basin_dict.items()
            if isinstance(info, dict) and info.get("most_detailed", False)
        ]
        datasets = []
        for code in constituent_basins:
            basin_obj = TCRiskBasinData(
                model=self.model,
                variant=self.variant,
                scenario=self.scenario,
                time_period=self.time_period,
                basin=code,
                base_path=self.base_path,
            )
            path = basin_obj.get_track_path(draw)
            if path.exists():
                datasets.append(xr.open_dataset(path))

        if not datasets:
            raise FileNotFoundError(
                f"No track files found for any basin: "
                f"{self.model}/{self.variant}/{self.scenario}/{self.time_period} draw {draw}"
            )

        combined = xr.concat(datasets, dim="n_trk", join="outer")
        for ds in datasets:
            ds.close()
        return combined
    
    def count_storms(self, draw: int = 0) -> int:
        """
        Count number of storms in a track file.
        
        Args:
            draw: Draw number
            
        Returns:
            Number of storms (tracks) in the file
        """
        ds = self.read_track_file(draw)
        n_storms = ds.dims.get('n_trk', 0)
        ds.close()
        return n_storms
    
    def get_track_data(self, draw: int = 0, track_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Extract track data for a specific storm.
        
        Args:
            draw: Draw number
            track_index: Index of the track/storm (0-indexed)
            
        Returns:
            Dictionary with lon, lat, wind, time arrays (NaN values removed)
        """
        ds = self.read_track_file(draw)
        
        # Extract data for specific track
        # TC Risk output uses lon_trks, lat_trks, vmax_trks
        lon = ds['lon_trks'].isel(n_trk=track_index).values
        lat = ds['lat_trks'].isel(n_trk=track_index).values
        wind = ds['vmax_trks'].isel(n_trk=track_index).values
        
        # Get time coordinate (not per-track, it's a coordinate)
        time = None
        if 'time' in ds.coords:
            time = ds['time'].values
        
        ds.close()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(lon) | np.isnan(lat))
        lon = lon[valid_mask]
        lat = lat[valid_mask]
        wind = wind[valid_mask] if wind is not None else None
        
        # Convert longitudes from 0-360 to -180/180 format
        lon = np.where(lon > 180, lon - 360, lon)
        
        result = {
            'lon': lon,
            'lat': lat,
            'wind': wind,
        }
        
        if time is not None:
            result['time'] = time[valid_mask]
        
        return result


@dataclass
class TCRiskModelVariant(ModelVariant):
    """
    Extends ModelVariant to manage TC Risk data across time periods and basins.
    
    Contains a dictionary of TCRiskBasinData objects for each basin.
    """
    
    time_periods: List[str] = field(default_factory=list)
    basins: Dict[str, TCRiskBasinData] = field(default_factory=dict)
    base_path: Path = field(default=Path("/mnt/team/rapidresponse/pub/tropical-storms/tc-risk/output"))
    
    def __post_init__(self):
        """Convert base_path to Path if string."""
        self.base_path = Path(self.base_path)
    
    def add_basin(
        self,
        basin: str,
        scenario: str,
        time_period: str,
    ) -> TCRiskBasinData:
        """Add a basin data object for a specific scenario and time period."""
        key = f"{basin}_{scenario}_{time_period}"
        
        basin_data = TCRiskBasinData(
            model=self.model,
            variant=self.variant,
            scenario=scenario,
            time_period=time_period,
            basin=basin,
            base_path=self.base_path,
        )
        
        self.basins[key] = basin_data
        
        # Track unique values
        if scenario not in self.scenarios:
            self.scenarios.append(scenario)
        if time_period not in self.time_periods:
            self.time_periods.append(time_period)
        
        return basin_data
    
    def get_basin(
        self,
        basin: str,
        scenario: str,
        time_period: str,
    ) -> Optional[TCRiskBasinData]:
        """Retrieve a basin data object."""
        key = f"{basin}_{scenario}_{time_period}"
        return self.basins.get(key)
    
    def list_basins(self) -> List[str]:
        """List all unique basin codes."""
        return list(set(key.split("_")[0] for key in self.basins.keys()))