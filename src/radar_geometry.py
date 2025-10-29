"""
Radar geometry and shadow analysis for SAR/InSAR data.

This module calculates radar-specific geometric effects including:
- Local incidence angle
- Radar shadow areas
- Layover areas
- Foreshortening effects
"""

import numpy as np
from scipy import ndimage


def calculate_local_incidence_angle(slope, aspect, radar_incidence=35.0,
                                    radar_azimuth=192.0):
    """
    Calculate local incidence angle accounting for terrain orientation.

    The local incidence angle is the angle between the radar look vector
    and the terrain surface normal. It determines illumination geometry
    and is critical for identifying shadow and layover areas.

    Parameters
    ----------
    slope : np.ndarray
        Terrain slope in degrees (0-90)
    aspect : np.ndarray
        Terrain aspect in degrees (0-360, 0=North, clockwise)
    radar_incidence : float, optional
        Radar incidence angle from vertical in degrees (default: 35.0)
        SAOCOM typical range: 20-50°
    radar_azimuth : float, optional
        Radar look direction in degrees (0-360, 0=North, clockwise)
        Default: 192° (descending, south-southwest looking)
        Ascending would be ~12° (north-northeast looking)

    Returns
    -------
    np.ndarray
        Local incidence angle in degrees
        - Values < 90°: Illuminated areas
        - Values > 90°: Shadow areas (radar cannot see)
        - Very small values: Potential layover (strong foreshortening)

    Notes
    -----
    Formula based on:
    cos(θ_local) = cos(θ_radar) * cos(slope) +
                   sin(θ_radar) * sin(slope) * cos(aspect - azimuth)

    where θ_radar is incidence angle, slope and aspect define terrain.

    Examples
    --------
    >>> slope = np.array([[0, 10, 30], [5, 20, 45]])
    >>> aspect = np.array([[0, 90, 180], [270, 45, 135]])
    >>> theta_local = calculate_local_incidence_angle(slope, aspect)
    >>> shadow_mask = theta_local > 90  # Shadow areas
    """
    # Convert to radians
    slope_rad = np.deg2rad(slope)
    aspect_rad = np.deg2rad(aspect)
    incidence_rad = np.deg2rad(radar_incidence)
    azimuth_rad = np.deg2rad(radar_azimuth)

    # Calculate aspect relative to radar look direction
    relative_aspect = aspect_rad - azimuth_rad

    # Local incidence angle formula
    cos_theta_local = (np.cos(incidence_rad) * np.cos(slope_rad) +
                       np.sin(incidence_rad) * np.sin(slope_rad) *
                       np.cos(relative_aspect))

    # Clip to valid range for arccos
    cos_theta_local = np.clip(cos_theta_local, -1, 1)

    # Convert to degrees
    theta_local = np.rad2deg(np.arccos(cos_theta_local))

    return theta_local


def identify_shadow_areas(local_incidence, shadow_threshold=90.0):
    """
    Identify radar shadow areas based on local incidence angle.

    Shadow occurs when the local incidence angle exceeds 90°, meaning
    the radar beam cannot reach the surface (blocked by terrain).

    Parameters
    ----------
    local_incidence : np.ndarray
        Local incidence angle in degrees
    shadow_threshold : float, optional
        Threshold in degrees for shadow (default: 90.0)

    Returns
    -------
    np.ndarray
        Boolean mask: True for shadow areas, False for illuminated

    Examples
    --------
    >>> shadow_mask = identify_shadow_areas(theta_local)
    >>> shadow_fraction = np.nanmean(shadow_mask)
    """
    return local_incidence >= shadow_threshold


def identify_layover_areas(local_incidence, layover_threshold=20.0):
    """
    Identify potential layover areas based on local incidence angle.

    Layover occurs when slopes face toward the radar and are steep,
    causing severe foreshortening where top of slope appears before base.

    Parameters
    ----------
    local_incidence : np.ndarray
        Local incidence angle in degrees
    layover_threshold : float, optional
        Threshold in degrees below which layover is likely (default: 20.0)

    Returns
    -------
    np.ndarray
        Boolean mask: True for potential layover areas

    Notes
    -----
    Layover is most severe when local incidence < radar incidence angle
    and slopes are steep and facing toward radar.

    Examples
    --------
    >>> layover_mask = identify_layover_areas(theta_local, threshold=15)
    """
    return local_incidence <= layover_threshold


def calculate_foreshortening_factor(local_incidence):
    """
    Calculate radar foreshortening factor.

    Foreshortening is the compression of terrain features in radar
    geometry. Factor < 1 indicates compression, = 1 no distortion.

    Parameters
    ----------
    local_incidence : np.ndarray
        Local incidence angle in degrees

    Returns
    -------
    np.ndarray
        Foreshortening factor (0 to 1+)
        - ~1.0: Minimal distortion
        - <0.5: Severe foreshortening (potential layover)
        - NaN: Shadow areas

    Examples
    --------
    >>> foreshorten = calculate_foreshortening_factor(theta_local)
    >>> severe_mask = foreshorten < 0.5
    """
    # Foreshortening = sin(local_incidence) / sin(radar_incidence)
    # For simplicity, use sin(local_incidence)
    local_incidence_rad = np.deg2rad(local_incidence)
    factor = np.sin(local_incidence_rad)

    # Set shadow areas (>90°) to NaN
    factor = np.where(local_incidence >= 90, np.nan, factor)

    return factor


def classify_geometric_quality(local_incidence, slope,
                               shadow_thresh=90.0,
                               layover_thresh=20.0,
                               steep_slope_thresh=30.0):
    """
    Classify areas by radar geometric quality.

    Parameters
    ----------
    local_incidence : np.ndarray
        Local incidence angle in degrees
    slope : np.ndarray
        Terrain slope in degrees
    shadow_thresh : float, optional
        Threshold for shadow (default: 90°)
    layover_thresh : float, optional
        Threshold for layover (default: 20°)
    steep_slope_thresh : float, optional
        Threshold for steep terrain (default: 30°)

    Returns
    -------
    np.ndarray
        Classification array:
        - 0: Optimal (flat to moderate slope, good illumination)
        - 1: Acceptable (moderate slope, acceptable illumination)
        - 2: Foreshortening (steep slopes facing radar)
        - 3: Shadow (radar blocked by terrain)
        - 4: Layover (severe foreshortening)

    Examples
    --------
    >>> quality = classify_geometric_quality(theta_local, slope)
    >>> quality_names = ['Optimal', 'Acceptable', 'Foreshortening',
    ...                  'Shadow', 'Layover']
    """
    # Initialize classification
    classification = np.zeros_like(local_incidence, dtype=np.int8)

    # 1. Optimal: moderate incidence, gentle slopes
    optimal_mask = (local_incidence >= 30) & (local_incidence < 60) & (slope < 15)
    classification[optimal_mask] = 0

    # 2. Acceptable: reasonable geometry
    acceptable_mask = ((local_incidence >= 25) & (local_incidence < 80) &
                      (slope < steep_slope_thresh) & ~optimal_mask)
    classification[acceptable_mask] = 1

    # 3. Foreshortening: steep slopes, moderate incidence
    foreshorten_mask = ((local_incidence >= layover_thresh) &
                       (local_incidence < shadow_thresh) &
                       (slope >= steep_slope_thresh))
    classification[foreshorten_mask] = 2

    # 4. Shadow: local incidence >= 90°
    shadow_mask = local_incidence >= shadow_thresh
    classification[shadow_mask] = 3

    # 5. Layover: very small local incidence (overrides others)
    layover_mask = local_incidence < layover_thresh
    classification[layover_mask] = 4

    return classification


def calculate_radar_brightness(local_incidence, normalize=True):
    """
    Calculate expected relative radar brightness based on geometry.

    Brightness is proportional to cos(local_incidence) for diffuse scattering.

    Parameters
    ----------
    local_incidence : np.ndarray
        Local incidence angle in degrees
    normalize : bool, optional
        Normalize to 0-1 range (default: True)

    Returns
    -------
    np.ndarray
        Relative brightness factor
        - Higher values: Brighter (better signal)
        - Lower values: Darker (worse signal)
        - Negative/NaN: Shadow

    Examples
    --------
    >>> brightness = calculate_radar_brightness(theta_local)
    >>> plt.imshow(brightness, cmap='gray')
    """
    local_incidence_rad = np.deg2rad(local_incidence)
    brightness = np.cos(local_incidence_rad)

    # Set shadow to NaN
    brightness = np.where(local_incidence >= 90, np.nan, brightness)

    if normalize:
        valid_brightness = brightness[~np.isnan(brightness)]
        if len(valid_brightness) > 0:
            min_val = np.nanmin(valid_brightness)
            max_val = np.nanmax(valid_brightness)
            if max_val > min_val:
                brightness = (brightness - min_val) / (max_val - min_val)

    return brightness


def analyze_shadow_statistics(gdf, local_incidence_col='local_incidence',
                              residual_col='residual'):
    """
    Analyze accuracy metrics stratified by shadow conditions.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with local incidence and residuals
    local_incidence_col : str, optional
        Column name for local incidence angle
    residual_col : str, optional
        Column name for residuals

    Returns
    -------
    dict
        Statistics for different illumination categories:
        - 'optimal': 30-60° local incidence
        - 'acceptable': 20-80° local incidence
        - 'steep': 80-90° (near shadow)
        - 'shadow': >90°
        - 'layover': <20°

    Examples
    --------
    >>> stats = analyze_shadow_statistics(gdf)
    >>> print(f"Shadow RMSE: {stats['shadow']['rmse']:.2f} m")
    """
    from scipy.stats import median_abs_deviation

    stats = {}

    # Define categories
    categories = {
        'optimal': (30, 60),
        'acceptable': (20, 80),
        'steep': (80, 90),
        'shadow': (90, 180),
        'layover': (0, 20)
    }

    for category, (min_angle, max_angle) in categories.items():
        mask = ((gdf[local_incidence_col] >= min_angle) &
                (gdf[local_incidence_col] < max_angle) &
                gdf[residual_col].notna())

        if mask.sum() > 0:
            residuals = gdf.loc[mask, residual_col]

            stats[category] = {
                'count': len(residuals),
                'bias': float(np.mean(residuals)),
                'rmse': float(np.sqrt(np.mean(residuals**2))),
                'nmad': float(1.4826 * median_abs_deviation(residuals, nan_policy='omit')),
                'std': float(np.std(residuals)),
                'min_incidence': float(gdf.loc[mask, local_incidence_col].min()),
                'max_incidence': float(gdf.loc[mask, local_incidence_col].max()),
                'mean_incidence': float(gdf.loc[mask, local_incidence_col].mean())
            }
        else:
            stats[category] = {
                'count': 0,
                'bias': np.nan,
                'rmse': np.nan,
                'nmad': np.nan,
                'std': np.nan,
                'min_incidence': np.nan,
                'max_incidence': np.nan,
                'mean_incidence': np.nan
            }

    return stats
