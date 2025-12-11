"""
SAOCOM DEM Validation Analysis Package

This package provides utilities for validating SAOCOM InSAR-derived heights
against reference Digital Elevation Models (DEMs).

Modules:
    - utils: Utility functions for raster I/O and metadata
    - data_loading: Functions to load SAOCOM, DEM, and land cover data
    - preprocessing: Raster processing (resampling, masking, sampling)
    - calibration: Height calibration functions
    - outlier_detection: Outlier detection and filtering
    - statistics: Statistical analysis functions (NMAD, summary stats)
    - landcover: CORINE land cover processing
    - visualization: Plotting and visualization functions
    - radar_geometry: Radar geometry and shadow analysis for SAR/InSAR
    - control_points: Control point identification and quality assessment
"""

__version__ = "1.0.0"
__author__ = "SAOCOM Project Team"

# Import key functions for easy access
from .utils import read_raster_meta, load_dem_array
from .statistics_prog import nmad, calculate_nmad
from .landcover import get_clc_level1
from .calibration import calibrate_heights
from .radar_geometry import (
    calculate_local_incidence_angle,
    identify_shadow_areas,
    classify_geometric_quality
)
from .control_points import (
    identify_control_points,
    analyze_control_point_distribution,
    export_control_points
)
