"""
Visualization functions for SAOCOM height validation analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plot_raster_with_stats(ax, data, cmap, title, cbar_label, extent,
                            hull_gdf=None, vlims=None, stats_arr=None,
                            nodata=-9999):
    """
    Plot a raster with statistics overlay and study area boundary.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : np.ndarray
        Raster data to plot
    cmap : matplotlib.colors.Colormap
        Colormap
    title : str
        Plot title
    cbar_label : str
        Colorbar label
    extent : tuple
        (xmin, xmax, ymin, ymax) for image extent
    hull_gdf : gpd.GeoDataFrame, optional
        Study area boundary to overlay
    vlims : tuple, optional
        (vmin, vmax) for color limits
    stats_arr : np.ndarray, optional
        Array for statistics calculation
    nodata : float, optional
        Nodata value (default: -9999)

    Returns
    -------
    matplotlib.image.AxesImage
        The image object

    Examples
    --------
    >>> im = plot_raster_with_stats(ax, dem_array, 'terrain', 'DEM',
    ...                              'Elevation (m)', extent, hull_gdf=hull)
    """
    ax.set_facecolor('white')

    # Mask invalid data
    disp = np.ma.masked_equal(data, nodata)

    cm = cmap.copy() if hasattr(cmap, 'copy') else plt.get_cmap(cmap)
    cm.set_bad(color='white', alpha=0)

    # Plot raster
    im_kwargs = {'cmap': cm, 'origin': 'upper', 'extent': extent}
    if vlims:
        im_kwargs.update({'vmin': vlims[0], 'vmax': vlims[1]})

    im = ax.imshow(disp, **im_kwargs)

    # Plot boundary
    if hull_gdf is not None:
        hull_gdf.boundary.plot(ax=ax, color='darkred', linewidth=2.5,
                               label='Study Area')

    # Labels and styling
    ax.set(title=title, xlabel='UTM Easting (m)', ylabel='UTM Northing (m)')
    ax.grid(True, color='black', alpha=0.3, linewidth=0.5)
    ax.tick_params(colors='black')

    # Colorbar
    cb = plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    cb.ax.yaxis.label.set_color('black')
    cb.ax.tick_params(colors='black')

    # Statistics text box
    if stats_arr is not None and stats_arr.size:
        txt = (f"Min: {np.nanmin(stats_arr):.1f}m\n"
               f"Max: {np.nanmax(stats_arr):.1f}m\n"
               f"Mean: {np.nanmean(stats_arr):.1f}m\n"
               f"Std: {np.nanstd(stats_arr):.1f}m")

        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                         edgecolor='black'))

    return im


def plot_gridded_panel(ax, data, title, cmap, extent, hull_gdf=None,
                       vmin=None, vmax=None, stats_text=None):
    """
    Plot a single gridded difference map panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : np.ndarray
        Gridded data
    title : str
        Panel title
    cmap : str or Colormap
        Colormap name or object
    extent : tuple
        (xmin, xmax, ymin, ymax)
    hull_gdf : gpd.GeoDataFrame, optional
        Boundary to overlay
    vmin, vmax : float, optional
        Color limits
    stats_text : str, optional
        Statistics text to overlay

    Examples
    --------
    >>> plot_gridded_panel(ax, diff_grid, 'SAOCOM - TINItaly',
    ...                    'RdBu_r', extent, hull_gdf=hull)
    """
    ax.set_facecolor('white')

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='white', alpha=0)

    im = ax.imshow(data, cmap=cmap, origin='upper', extent=extent,
                   vmin=vmin, vmax=vmax)

    if hull_gdf is not None:
        hull_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('UTM Easting (m)')
    ax.set_ylabel('UTM Northing (m)')
    ax.grid(True, alpha=0.3)

    plt.colorbar(im, ax=ax, label='Difference (m)', shrink=0.8)

    if stats_text:
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))


def plot_points_panel(ax, data, title, cmap, hull_gdf=None,
                      vmin=None, vmax=None, stats_text=None):
    """
    Plot a single point-based difference map panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : gpd.GeoDataFrame
        Point data with 'diff' column
    title : str
        Panel title
    cmap : str or Colormap
        Colormap
    hull_gdf : gpd.GeoDataFrame, optional
        Boundary
    vmin, vmax : float, optional
        Color limits
    stats_text : str, optional
        Statistics text

    Examples
    --------
    >>> plot_points_panel(ax, valid_pts, 'SAOCOM Points',
    ...                   'RdBu_r', hull_gdf=hull)
    """
    ax.set_facecolor('gainsboro')

    im = ax.scatter(data.geometry.x, data.geometry.y, c=data['diff'],
                    s=1, alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax)

    if hull_gdf is not None:
        hull_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('UTM Easting (m)')
    ax.set_ylabel('UTM Northing (m)')
    ax.grid(True, alpha=0.3)

    plt.colorbar(im, ax=ax, label='Difference (m)', shrink=0.8)

    if stats_text:
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))


def plot_distribution_histogram(ax, diff_series, title, metrics):
    """
    Plot residual distribution histogram with statistics.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    diff_series : pd.Series or np.ndarray
        Residual values
    title : str
        Plot title
    metrics : dict
        Dictionary with keys: n_points, mean_diff, rmse, nmad, std_diff

    Examples
    --------
    >>> metrics = {
    ...     'n_points': len(residuals),
    ...     'mean_diff': residuals.mean(),
    ...     'rmse': np.sqrt((residuals**2).mean()),
    ...     'nmad': calculate_nmad(residuals),
    ...     'std_diff': residuals.std()
    ... }
    >>> plot_distribution_histogram(ax, residuals, 'TINItaly Residuals', metrics)
    """
    ax.set_facecolor('white')

    ax.hist(diff_series, bins=20, alpha=0.75, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Zero')
    ax.axvline(metrics['mean_diff'], color='green', linestyle='-',
               label=f"Mean: {metrics['mean_diff']:+.2f}m")

    stats_text = (f"n = {metrics['n_points']:,}\n"
                  f"RMSE = {metrics['rmse']:.2f} m\n"
                  f"NMAD = {metrics['nmad']:.2f} m\n"
                  f"Std Dev = {metrics['std_diff']:.2f} m")

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('Elevation Difference (m)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_scatter_comparison(ax, x_data, y_data, x_label, y_label, title, stats):
    """
    Create 1:1 scatter plot comparing two height datasets.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x_data, y_data : array-like
        Height data to compare
    x_label, y_label : str
        Axis labels
    title : str
        Plot title
    stats : dict
        Dictionary with keys: n_points (or n_pixels), mean_diff, rmse, correlation

    Examples
    --------
    >>> stats = {
    ...     'n_points': len(x_data),
    ...     'mean_diff': (y_data - x_data).mean(),
    ...     'rmse': np.sqrt(((y_data - x_data)**2).mean()),
    ...     'correlation': np.corrcoef(x_data, y_data)[0, 1]
    ... }
    >>> plot_scatter_comparison(ax, saocom_heights, ref_heights,
    ...                          'SAOCOM', 'TINItaly', 'Comparison', stats)
    """
    ax.set_facecolor('white')

    ax.scatter(x_data, y_data, s=1, alpha=0.3, c='steelblue', label='Data Points')

    # 1:1 line
    lims = [
        np.min([x_data.min(), y_data.min()]),
        np.max([x_data.max(), y_data.max()]),
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='1:1 Line', zorder=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Statistics box
    n_key = 'n_points' if 'n_points' in stats else 'n_pixels'
    stats_text = (f"n = {stats[n_key]:,}\n"
                  f"Bias = {stats['mean_diff']:.2f} m\n"
                  f"RMSE = {stats['rmse']:.2f} m\n"
                  f"Corr (r) = {stats['correlation']:.3f}")

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='black',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                     edgecolor='black'))

    ax.set_title(title, fontweight='bold', fontsize=12, color='black')
    ax.set_xlabel(x_label, fontsize=11, color='black')
    ax.set_ylabel(y_label, fontsize=11, color='black')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_aspect('equal', 'box')
    ax.tick_params(colors='black')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')


def plot_hexbin_density(ax, x_data, y_data, x_label, y_label, title, fig=None):
    """
    Create hexbin density plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x_data, y_data : array-like
        Data to plot
    x_label, y_label : str
        Axis labels
    title : str
        Plot title
    fig : matplotlib.figure.Figure, optional
        Figure object for colorbar

    Examples
    --------
    >>> plot_hexbin_density(ax, saocom, tinitaly, 'SAOCOM (m)',
    ...                     'TINItaly (m)', 'Density Plot', fig=fig)
    """
    ax.set_facecolor('gainsboro')

    hb = ax.hexbin(x_data, y_data, gridsize=150, cmap='inferno',
                   norm=colors.LogNorm(), mincnt=1)

    if fig is not None:
        fig.colorbar(hb, ax=ax, label='Point Count')

    # 1:1 line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'r--', linewidth=2, label='1:1 line', alpha=0.9)

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.legend()
    ax.set_aspect('equal', 'box')


def plot_hist2d(ax, x_data, y_data, x_label, y_label, title, fig=None):
    """
    Create 2D histogram plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x_data, y_data : array-like
        Data to plot
    x_label, y_label : str
        Axis labels
    title : str
        Plot title
    fig : matplotlib.figure.Figure, optional
        Figure object for colorbar

    Examples
    --------
    >>> plot_hist2d(ax, saocom, copernicus, 'SAOCOM (m)',
    ...             'Copernicus (m)', '2D Histogram', fig=fig)
    """
    ax.set_facecolor('gainsboro')

    h = ax.hist2d(x_data, y_data, bins=150, cmap='inferno',
                  norm=colors.LogNorm(), cmin=1)

    if fig is not None:
        fig.colorbar(h[3], ax=ax, label='Point Count')

    # 1:1 line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'w--', linewidth=1.5, label='1:1 line', alpha=0.7)

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.legend()
    ax.set_aspect('equal', 'box')


def plot_bland_altman(ax, x_data, y_data, x_label, y_label, title, fig=None):
    """
    Create Bland-Altman plot for agreement analysis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x_data, y_data : array-like
        Two measurement methods to compare
    x_label, y_label : str
        Labels for the two methods
    title : str
        Plot title
    fig : matplotlib.figure.Figure, optional
        Figure object for colorbar

    Notes
    -----
    Bland-Altman plots show:
    - X-axis: Average of two measurements
    - Y-axis: Difference between measurements
    - Lines: Mean difference and limits of agreement (±1.96 SD)

    Examples
    --------
    >>> plot_bland_altman(ax, saocom, tinitaly, 'SAOCOM', 'TINItaly',
    ...                   'Bland-Altman Analysis', fig=fig)
    """
    # Calculate average and difference
    average = (x_data + y_data) / 2
    difference = y_data - x_data

    # Statistics
    mean_diff = np.mean(difference)
    std_diff = np.std(difference)
    limit_of_agreement = 1.96 * std_diff

    ax.set_facecolor('gainsboro')

    # Hexbin plot
    hb = ax.hexbin(average, difference, gridsize=150, cmap='viridis',
                   norm=colors.LogNorm(), mincnt=1)

    if fig is not None:
        fig.colorbar(hb, ax=ax, label='Point Count')

    # Statistical lines
    ax.axhline(0, color='white', linestyle='--', linewidth=1.5,
               label='Zero (Perfect Agreement)')
    ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2,
               label=f'Mean Diff: {mean_diff:.2f} m')
    ax.axhline(mean_diff + limit_of_agreement, color='red', linestyle='--',
               linewidth=1.5,
               label=f'Limits of Agreement (±{limit_of_agreement:.2f} m)')
    ax.axhline(mean_diff - limit_of_agreement, color='red', linestyle='--',
               linewidth=1.5)

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel(f'Average of ({x_label} and {y_label})', fontsize=11)
    ax.set_ylabel(f'Difference ({y_label} - {x_label})', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.legend()
