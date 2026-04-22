"""CLIMADA model output data classes."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator

import xarray as xr

from idd_climate_models.constants import basin_dict
from .base import ModelVariant


# Wind speed thresholds for storm categories (m/s)
# Based on Saffir-Simpson scale converted to m/s
STORM_CATEGORIES = {
    "tropical_depression": (0, 17),      # < 17 m/s (< 39 mph)
    "tropical_storm": (17, 33),          # 17-32 m/s (39-73 mph)
    "cat1": (33, 43),                    # 33-42 m/s (74-95 mph)
    "cat2": (43, 50),                    # 43-49 m/s (96-110 mph)
    "cat3": (50, 58),                    # 50-57 m/s (111-129 mph)
    "cat4": (58, 70),                    # 58-69 m/s (130-156 mph)
    "cat5": (70, float("inf")),          # >= 70 m/s (>= 157 mph)
}


def classify_storm_category(max_wind_speed: float) -> str:
    """
    Classify a storm based on maximum wind speed.
    
    Args:
        max_wind_speed: Maximum wind speed in m/s
        
    Returns:
        Storm category string
    """
    for category, (min_speed, max_speed) in STORM_CATEGORIES.items():
        if min_speed <= max_wind_speed < max_speed:
            return category
    return "unknown"


def is_hurricane(max_wind_speed: float) -> bool:
    """Check if storm qualifies as hurricane (Cat 1+)."""
    return max_wind_speed >= 33


def is_major_hurricane(max_wind_speed: float) -> bool:
    """Check if storm qualifies as major hurricane (Cat 3+)."""
    return max_wind_speed >= 50


@dataclass
class ClimadaStormData:
    """
    Represents a single storm's CLIMADA output data.
    
    Contains intensity and exposure hours xarray datasets for one storm.
    """
    
    storm_name: str
    basin: str
    model: str
    variant: str
    scenario: str
    time_period: str
    draw: int
    intensity_path: Path
    exposure_hours_path: Path
    
    # Lazy-loaded datasets
    _intensity_ds: Optional[xr.Dataset] = field(default=None, repr=False)
    _exposure_hours_ds: Optional[xr.Dataset] = field(default=None, repr=False)
    
    # Cached attributes
    _attrs_loaded: bool = field(default=False, repr=False)
    _start_date: Optional[datetime] = field(default=None, repr=False)
    _end_date: Optional[datetime] = field(default=None, repr=False)
    _category: Optional[int] = field(default=None, repr=False)
    _storm_id: Optional[int] = field(default=None, repr=False)
    
    @property
    def basin_info(self) -> dict:
        """Return full basin info dict from constants."""
        return basin_dict.get(self.basin, {})
    
    @property
    def basin_name(self) -> str:
        """Return full basin name."""
        return self.basin_info.get('name', self.basin)
    
    @property
    def basin_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Return basin bounds as (west, east, south, north) floats.
        """
        raw_bounds = self.basin_info.get('basin_bounds')
        if not raw_bounds:
            return None
        
        def parse_coord(coord_str: str) -> float:
            coord_str = coord_str.strip()
            if coord_str.endswith('E'):
                return float(coord_str[:-1])
            elif coord_str.endswith('W'):
                return -float(coord_str[:-1])
            elif coord_str.endswith('N'):
                return float(coord_str[:-1])
            elif coord_str.endswith('S'):
                return -float(coord_str[:-1])
            else:
                return float(coord_str)
        
        west = parse_coord(raw_bounds[0])
        south = parse_coord(raw_bounds[1])
        east = parse_coord(raw_bounds[2])
        north = parse_coord(raw_bounds[3])
        
        if west > 180:
            west -= 360
        if east > 180:
            east -= 360
        
        return (west, east, south, north)

    @property
    def intensity(self) -> xr.Dataset:
        """Load and return intensity dataset."""
        if self._intensity_ds is None:
            self._intensity_ds = xr.open_zarr(
                self.intensity_path / self.storm_name,
                consolidated=False,
                decode_timedelta=False,
            )
            self._load_attrs_from_intensity()
        return self._intensity_ds
    
    @property
    def exposure_hours(self) -> xr.Dataset:
        """Load and return exposure hours dataset."""
        if self._exposure_hours_ds is None:
            self._exposure_hours_ds = xr.open_zarr(
                self.exposure_hours_path / self.storm_name,
                consolidated=False,
                decode_timedelta=False,
            )
        return self._exposure_hours_ds
    
    def _load_attrs_from_intensity(self) -> None:
        """Load storm attributes from intensity dataset."""
        if self._attrs_loaded or self._intensity_ds is None:
            return
        
        attrs = self._intensity_ds.attrs
        
        if "start_date" in attrs:
            self._start_date = datetime.strptime(attrs["start_date"], "%Y-%m-%d")
        if "end_date" in attrs:
            self._end_date = datetime.strptime(attrs["end_date"], "%Y-%m-%d")
        if "category" in attrs:
            self._category = int(attrs["category"])
        if "storm_id" in attrs:
            self._storm_id = int(attrs["storm_id"])
        
        self._attrs_loaded = True
    
    @property
    def start_date(self) -> Optional[datetime]:
        """Storm start date."""
        if not self._attrs_loaded:
            _ = self.intensity  # Trigger loading
        return self._start_date
    
    @property
    def end_date(self) -> Optional[datetime]:
        """Storm end date."""
        if not self._attrs_loaded:
            _ = self.intensity
        return self._end_date
    
    @property
    def year(self) -> Optional[int]:
        """Year the storm occurred (from start_date)."""
        if self.start_date:
            return self.start_date.year
        return None
    
    @property
    def category(self) -> Optional[int]:
        """Storm category from metadata."""
        if not self._attrs_loaded:
            _ = self.intensity
        return self._category
    
    @property
    def storm_id(self) -> Optional[int]:
        """Storm ID (for matching with TC Risk output)."""
        if not self._attrs_loaded:
            _ = self.intensity
        return self._storm_id
    
    @property
    def max_intensity(self) -> float:
        """Maximum wind speed across all pixels."""
        return float(self.intensity["intensity"].max().values)
    
    @property
    def category_name(self) -> str:
        """Storm category based on max intensity."""
        return classify_storm_category(self.max_intensity)
    
    def get_intensity_array(self) -> xr.DataArray:
        """Return the intensity DataArray."""
        return self.intensity["intensity"]
    
    def get_exposure_hours_array(self) -> xr.DataArray:
        """Return the exposure hours DataArray."""
        return self.exposure_hours["exposure_hours"]
    
    def close(self) -> None:
        """Close open datasets to free memory."""
        if self._intensity_ds is not None:
            self._intensity_ds.close()
            self._intensity_ds = None
        if self._exposure_hours_ds is not None:
            self._exposure_hours_ds.close()
            self._exposure_hours_ds = None
        self._attrs_loaded = False


@dataclass
class ClimadaBasinData:
    """
    Manages CLIMADA output data for a single basin/model/variant/scenario/time_period/draw.
    
    Contains multiple ClimadaStormData objects, one per storm.
    """
    
    model: str
    variant: str
    scenario: str
    time_period: str
    basin: str
    draw: int
    base_path: Path = field(default=Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage0"))
    
    # Storm data objects
    storms: Dict[str, ClimadaStormData] = field(default_factory=dict)
    _storm_list: Optional[List[str]] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Convert base_path to Path if string."""
        self.base_path = Path(self.base_path)
    
    @property
    def basin_info(self) -> dict:
        """Return full basin info dict from constants."""
        return basin_dict.get(self.basin, {})
    
    @property
    def basin_name(self) -> str:
        """Return full basin name from basin code."""
        return self.basin_info.get('name', self.basin)
    
    @property
    def basin_bounds_raw(self) -> Optional[List[str]]:
        """Return raw basin bounds strings (e.g., ['100E', '45S', '180E', '0S'])."""
        return self.basin_info.get('basin_bounds')
    
    @property
    def basin_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Return basin bounds as (west, east, south, north) floats.
        
        Returns:
            Tuple of (west, east, south, north) in degrees, or None if not found
        """
        raw_bounds = self.basin_bounds_raw
        if not raw_bounds:
            return None
        
        def parse_coord(coord_str: str) -> float:
            """Parse '100E' or '45S' into signed float."""
            coord_str = coord_str.strip()
            if coord_str.endswith('E'):
                return float(coord_str[:-1])
            elif coord_str.endswith('W'):
                return -float(coord_str[:-1])
            elif coord_str.endswith('N'):
                return float(coord_str[:-1])
            elif coord_str.endswith('S'):
                return -float(coord_str[:-1])
            else:
                return float(coord_str)
        
        west = parse_coord(raw_bounds[0])
        south = parse_coord(raw_bounds[1])
        east = parse_coord(raw_bounds[2])
        north = parse_coord(raw_bounds[3])
        
        # Handle longitude wrapping (e.g., 260E -> -100 for Atlantic)
        if west > 180:
            west -= 360
        if east > 180:
            east -= 360
        
        return (west, east, south, north)
    
    @property
    def data_dir(self) -> Path:
        """Return the directory containing data for this configuration."""
        return self.base_path / self.model / self.variant / self.scenario / self.time_period / self.basin
    
    def _build_zarr_path(self, data_type: str) -> Path:
        """Build path to zarr file for a data type (intensity or exposure_hours)."""
        start_year = self.time_period.split('-')[0]
        end_year = self.time_period.split('-')[1]
        
        draw_suffix = f"_e{self.draw - 1}" if self.draw > 0 else ""
        
        filename = f"{data_type}_{self.basin}_{self.model}_{self.scenario}_{self.variant}_{start_year}01_{end_year}12{draw_suffix}.zarr"
        
        return self.data_dir / data_type / filename
    
    @property
    def intensity_zarr_path(self) -> Path:
        """Path to intensity zarr store."""
        return self._build_zarr_path("intensity")
    
    @property
    def exposure_hours_zarr_path(self) -> Path:
        """Path to exposure hours zarr store."""
        return self._build_zarr_path("exposure_hours")
    
    def list_storms(self, refresh: bool = False) -> List[str]:
        """
        List available storm names in this zarr store.
        
        Args:
            refresh: If True, re-scan the directory
            
        Returns:
            List of storm names (e.g., ['storm_0001', 'storm_0002', ...])
        """
        if self._storm_list is not None and not refresh:
            return self._storm_list
        
        if not self.intensity_zarr_path.exists():
            self._storm_list = []
            return self._storm_list
        
        self._storm_list = sorted([
            f.name for f in self.intensity_zarr_path.iterdir()
            if f.name.startswith("storm_") and f.is_dir()
        ])
        
        return self._storm_list
    
    def validate_paths(self) -> Tuple[bool, str]:
        """
        Check that intensity and exposure_hours zarrs exist and have matching contents.
        
        Returns:
            Tuple of (is_valid, message)
        """
        if not self.intensity_zarr_path.exists():
            return False, f"Intensity zarr not found: {self.intensity_zarr_path}"
        
        if not self.exposure_hours_zarr_path.exists():
            return False, f"Exposure hours zarr not found: {self.exposure_hours_zarr_path}"
        
        intensity_storms = set(self.list_storms())
        exposure_storms = set(
            f.name for f in self.exposure_hours_zarr_path.iterdir()
            if f.name.startswith("storm_") and f.is_dir()
        )
        
        if intensity_storms != exposure_storms:
            missing_in_exposure = intensity_storms - exposure_storms
            missing_in_intensity = exposure_storms - intensity_storms
            return False, (
                f"Storm mismatch: {len(missing_in_exposure)} missing in exposure_hours, "
                f"{len(missing_in_intensity)} missing in intensity"
            )
        
        return True, f"Valid: {len(intensity_storms)} storms"
    
    def get_storm(self, storm_name: str) -> ClimadaStormData:
        """
        Get a ClimadaStormData object for a specific storm.
        
        Args:
            storm_name: Storm name (e.g., 'storm_0001')
            
        Returns:
            ClimadaStormData object
        """
        if storm_name not in self.storms:
            if storm_name not in self.list_storms():
                raise ValueError(f"Storm {storm_name} not found in {self.intensity_zarr_path}")
            
            self.storms[storm_name] = ClimadaStormData(
                storm_name=storm_name,
                basin=self.basin,
                model=self.model,
                variant=self.variant,
                scenario=self.scenario,
                time_period=self.time_period,
                draw=self.draw,
                intensity_path=self.intensity_zarr_path,
                exposure_hours_path=self.exposure_hours_zarr_path,
            )
        
        return self.storms[storm_name]
    
    def iter_storms(self) -> "Generator[ClimadaStormData, None, None]":
        """Iterate over all storms, yielding ClimadaStormData objects."""
        for storm_name in self.list_storms():
            yield self.get_storm(storm_name)
    
    def get_storms_by_year(self) -> Dict[int, List[ClimadaStormData]]:
        """Group storms by year."""
        by_year: Dict[int, List[ClimadaStormData]] = {}
        
        for storm in self.iter_storms():
            year = storm.year
            if year is not None:
                if year not in by_year:
                    by_year[year] = []
                by_year[year].append(storm)
        
        return by_year
    
    def get_storms_by_category(self) -> Dict[str, List[ClimadaStormData]]:
        """Group storms by category name."""
        by_category: Dict[str, List[ClimadaStormData]] = {}
        
        for storm in self.iter_storms():
            cat = storm.category_name
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(storm)
        
        return by_category
    
    def close_all(self) -> None:
        """Close all open storm datasets."""
        for storm in self.storms.values():
            storm.close()


@dataclass
class ClimadaModelVariant(ModelVariant):
    """
    Extends ModelVariant to manage CLIMADA data across time periods, basins, and draws.
    
    Contains a dictionary of ClimadaBasinData objects.
    """
    
    time_periods: List[str] = field(default_factory=list)
    basins: Dict[str, ClimadaBasinData] = field(default_factory=dict)
    base_path: Path = field(default=Path("/mnt/team/rapidresponse/pub/tropical-storms/climada/output/stage0"))
    
    def __post_init__(self):
        """Convert base_path to Path if string."""
        self.base_path = Path(self.base_path)
    
    def _make_key(self, basin: str, scenario: str, time_period: str, draw: int) -> str:
        """Create dictionary key for basin data."""
        return f"{basin}_{scenario}_{time_period}_{draw}"
    
    def add_basin(
        self,
        basin: str,
        scenario: str,
        time_period: str,
        draw: int = 0,
    ) -> ClimadaBasinData:
        """Add a basin data object for a specific scenario, time period, and draw."""
        key = self._make_key(basin, scenario, time_period, draw)
        
        basin_data = ClimadaBasinData(
            model=self.model,
            variant=self.variant,
            scenario=scenario,
            time_period=time_period,
            basin=basin,
            draw=draw,
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
        draw: int = 0,
    ) -> Optional[ClimadaBasinData]:
        """Retrieve a basin data object."""
        key = self._make_key(basin, scenario, time_period, draw)
        return self.basins.get(key)
    
    def list_basin_codes(self) -> List[str]:
        """List all unique basin codes."""
        return list(set(key.split("_")[0] for key in self.basins.keys()))
    
    def close_all(self) -> None:
        """Close all open datasets across all basins."""
        for basin_data in self.basins.values():
            basin_data.close_all()