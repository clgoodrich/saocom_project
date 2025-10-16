"""Utility helpers for producing land-cover visualisations efficiently.

This module packages the optimised plotting loop that was previously
experimented with in notebooks.  The primary motivation for creating a
dedicated module is to make it easier to reuse the logic from scripts while
keeping a single, well-tested implementation in the repository.

Two key improvements are implemented here:

* Column names from a ``pandas.DataFrame`` are normalised before iterating
  with :meth:`DataFrame.itertuples`.  The normalisation step mirrors the way
  ``itertuples`` sanitises column names internally and prevents the
  ``AttributeError`` that was observed when columns contained spaces or other
  non-alphanumeric characters (for example ``"LC Code"``).
* Land-cover polygons are dissolved per class so that a clean outline of the
  overall class is drawn on top of the coverage/void visualisation.  This
  keeps rendering costs low while satisfying the requirement of highlighting
  the class perimeter clearly.

The public entry-point is :func:`plot_landcover_maps`, which expects the key
arrays and statistics DataFrame that are already produced elsewhere in the
project.  The function is intentionally side-effect free except for writing
PNG files to the supplied results directory.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from affine import Affine
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch
from rasterio.features import shapes
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape
from shapely.ops import unary_union


@dataclass(frozen=True)
class PlotColours:
    """Container for the colours required to render a land-cover class."""

    fill_rgb: Tuple[float, float, float]
    outline: str = "black"
    outline_width: float = 3.0


REQUIRED_STATS_COLUMNS = {
    "LC_Code",
    "LC_Name",
    "Total_km2",
    "Pct_of_Study_Area",
    "Coverage_km2",
    "Pct_Coverage",
    "Void_km2",
    "Pct_Void",
}


def _sanitise_column_name(column: str) -> str:
    """Normalise a DataFrame column so ``itertuples`` exposes predictable attributes."""

    column = column.strip()
    column = re.sub(r"\s+", "_", column)
    column = re.sub(r"[^0-9A-Za-z_]", "_", column)
    column = re.sub(r"_+", "_", column)
    return column.strip("_")


def normalise_stats_columns(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``stats_df`` with sanitised column names.

    ``DataFrame.itertuples`` silently sanitises column names.  When the caller
    subsequently accesses ``row.LC_Code`` but the original column was named
    ``"LC Code"`` (with a space), ``AttributeError`` is raised.  Sanitising the
    columns ahead of time keeps the user facing API predictable and avoids the
    runtime error.
    """

    renamed = stats_df.copy()
    renamed.columns = [_sanitise_column_name(col) for col in renamed.columns]

    missing = REQUIRED_STATS_COLUMNS.difference(renamed.columns)
    if missing:
        raise KeyError(
            "stats_df is missing required columns after normalisation: "
            f"{sorted(missing)}"
        )

    return renamed


def _bbox_window(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    return r0, r1, c0, c1


def _slice_transform(transform: Affine, r0: int, c0: int) -> Affine:
    return transform * Affine.translation(c0, r0)


def _exterior_coords(geoms: Iterable[Polygon | MultiPolygon]) -> List[np.ndarray]:
    paths: List[np.ndarray] = []
    for geom in geoms:
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            paths.append(np.asarray(geom.exterior.coords))
        else:
            paths.extend(np.asarray(poly.exterior.coords) for poly in geom.geoms if not poly.is_empty)
    return paths


def _rings_from_geom(
    geom: Polygon | MultiPolygon,
    *,
    include_holes: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return exterior (and optionally interior) rings from ``geom`` as arrays."""

    outline_paths: List[np.ndarray] = []
    hole_paths: List[np.ndarray] = []

    if geom.is_empty:
        return outline_paths, hole_paths

    if isinstance(geom, Polygon):
        outline_paths.append(np.asarray(geom.exterior.coords))
        if include_holes:
            hole_paths.extend(np.asarray(ring.coords) for ring in geom.interiors)
        return outline_paths, hole_paths

    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            if poly.is_empty:
                continue
            outline_paths.append(np.asarray(poly.exterior.coords))
            if include_holes:
                hole_paths.extend(np.asarray(ring.coords) for ring in poly.interiors)
        return outline_paths, hole_paths

    boundary = geom.boundary
    if isinstance(boundary, LineString):
        outline_paths.append(np.asarray(boundary.coords))
    elif isinstance(boundary, MultiLineString):
        outline_paths.extend(np.asarray(segment.coords) for segment in boundary.geoms)

    return outline_paths, hole_paths


def plot_landcover_maps(
    stats_df: pd.DataFrame,
    corine_10m: np.ndarray,
    void_mask: np.ndarray,
    target_transform: Affine,
    sentinel_rgb_norm: np.ndarray,
    extent: Sequence[float],
    colour_map: dict[int, Tuple[int, int, int]],
    hull_gdf,
    results_dir: Path,
    *,
    include_hole_outlines: bool = False,
) -> None:
    """Render per-class land-cover figures using the optimised plotting routine.

    Parameters
    ----------
    stats_df:
        Summary statistics for each land-cover class.  The function will create
        a copy with sanitised columns to avoid ``AttributeError`` when using
        ``itertuples``.
    corine_10m, void_mask:
        Raster arrays defining the land-cover classes and void/no-data pixels.
    target_transform:
        Affine transform mapping pixel coordinates to spatial coordinates.
    sentinel_rgb_norm:
        Background RGB array (values in ``[0, 1]``) shown underneath the vector
        outlines.
    extent:
        Sequence ``(xmin, xmax, ymin, ymax)`` used for ``imshow``.
    colour_map:
        Dictionary mapping land-cover codes to RGB tuples in the ``0–255``
        range.
    hull_gdf:
        GeoDataFrame containing the study-area boundary.  Only the boundary is
        drawn.
    results_dir:
        Destination directory for the generated PNG files.
    include_hole_outlines:
        When ``True`` interior voids are outlined using a dashed stroke.
    """

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    stats_df = normalise_stats_columns(stats_df)

    xmin_grid, xmax_grid, ymin_grid, ymax_grid = extent

    # Pre-compute the dissolved void polygons once; they are reused for each class.
    void_start = time.perf_counter()
    void_shapes = shapes(void_mask.astype(np.uint8), mask=void_mask, transform=target_transform)
    void_polys: List[Polygon] = []
    for geom, value in void_shapes:
        if value != 1:
            continue
        poly = shape(geom).buffer(0)
        if not poly.is_empty:
            void_polys.append(poly)
    void_union = unary_union(void_polys) if void_polys else None
    void_end = time.perf_counter()
    print(f"[precompute] void_union built in {void_end - void_start:.3f}s")

    pix_w = abs(target_transform.a)
    pix_h = abs(target_transform.e)
    pix_area = pix_w * pix_h
    min_area = 4 * pix_area

    for row in stats_df.itertuples(index=True):
        loop_start = time.perf_counter()
        lc_code = row.LC_Code
        class_name = row.LC_Name

        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor="white")
        ax.set_facecolor("white")
        ax.imshow(
            sentinel_rgb_norm,
            extent=[xmin_grid, xmax_grid, ymin_grid, ymax_grid],
            origin="upper",
            alpha=0.4,
            interpolation="nearest",
        )

        rgb = colour_map.get(lc_code, (128, 128, 128))
        colours = PlotColours(tuple(channel / 255.0 for channel in rgb))

        lc_mask = corine_10m == lc_code
        window = _bbox_window(lc_mask)
        if window is None:
            plt.close(fig)
            continue

        r0, r1, c0, c1 = window
        lc_mask_s = lc_mask[r0:r1, c0:c1]
        transform_s = _slice_transform(target_transform, r0, c0)

        t1 = time.perf_counter()
        class_shapes = shapes(lc_mask_s.astype(np.uint8), mask=lc_mask_s, transform=transform_s, connectivity=8)
        polygons: List[Polygon | MultiPolygon] = []
        for geom, value in class_shapes:
            if value != 1:
                continue
            poly = shape(geom).buffer(0)
            if not poly.is_empty and poly.area >= min_area:
                polygons.append(poly)
        t2 = time.perf_counter()

        outline_paths: List[np.ndarray] = []
        hole_paths: List[np.ndarray] = []
        if polygons:
            lc_union = unary_union(polygons)
            outline_paths, hole_paths = _rings_from_geom(
                lc_union, include_holes=include_hole_outlines
            )

        coverage_geoms: List[Polygon | MultiPolygon]
        void_geoms: List[Polygon | MultiPolygon]
        if void_union:
            coverage_geoms = []
            void_geoms = []
            for poly in polygons:
                void_part = poly.intersection(void_union)
                coverage_part = poly.difference(void_union)
                if not coverage_part.is_empty:
                    coverage_geoms.append(coverage_part)
                if not void_part.is_empty:
                    void_geoms.append(void_part)
        else:
            coverage_geoms = polygons
            void_geoms = []
        t3 = time.perf_counter()

        cov_paths = _exterior_coords(coverage_geoms)
        if cov_paths:
            ax.add_collection(
                PolyCollection(cov_paths, facecolors=[colours.fill_rgb], edgecolors="none", alpha=0.35)
            )
            ax.add_collection(LineCollection(cov_paths, colors=["black"], linewidths=2.0))
            ax.add_collection(LineCollection(cov_paths, colors=[colours.fill_rgb], linewidths=1.2))

        void_paths = _exterior_coords(void_geoms)
        if void_paths:
            ax.add_collection(
                PolyCollection(void_paths, facecolors=["white"], edgecolors="none", alpha=0.8)
            )
            ax.add_collection(
                LineCollection(void_paths, colors=["red"], linewidths=1.5, linestyles="--")
            )
            ax.add_collection(
                LineCollection(void_paths, colors=["gray"], linewidths=0.8, linestyles="--")
            )

        if outline_paths:
            ax.add_collection(
                LineCollection(
                    outline_paths,
                    colors=[colours.outline],
                    linewidths=colours.outline_width,
                    zorder=10,
                )
            )

        if include_hole_outlines and hole_paths:
            ax.add_collection(
                LineCollection(
                    hole_paths,
                    colors=[colours.outline],
                    linewidths=max(colours.outline_width - 1.0, 1.0),
                    linestyles="--",
                    alpha=0.7,
                    zorder=9,
                )
            )

        hull_gdf.boundary.plot(ax=ax, color="black", linewidth=3, linestyle="-")

        ax.set_title(
            (
                f"Land Cover: {class_name} (Code {lc_code})\n"
                f"Total: {row.Total_km2:.1f} km² ({row.Pct_of_Study_Area:.1f}% of study area) | "
                f"Coverage: {row.Coverage_km2:.1f} km² ({row.Pct_Coverage:.1f}%) | "
                f"Void: {row.Void_km2:.1f} km² ({row.Pct_Void:.1f}%)"
            ),
            fontweight="bold",
            fontsize=12,
            pad=15,
        )

        stats_text = (
            f"Coverage: {row.Pct_Coverage:.1f}%\n"
            f"Void: {row.Pct_Void:.1f}%\n"
            f"Area w/ data: {row.Coverage_km2:.1f} km²\n"
            f"Area w/o data: {row.Void_km2:.1f} km²"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
        )

        legend_elements = [
            Patch(
                facecolor=colours.fill_rgb,
                edgecolor="black",
                linewidth=2,
                alpha=0.35,
                label=f"{class_name} (Coverage)",
            ),
            Patch(
                facecolor="white",
                edgecolor="red",
                linewidth=1.5,
                alpha=0.8,
                label=f"{class_name} (Void/No Data)",
            ),
            Patch(
                facecolor="none",
                edgecolor="black",
                linewidth=3,
                label="Study Area Boundary",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=10,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            title="Legend",
        )

        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")
        ax.grid(True, alpha=0.3, color="gray", linewidth=0.5)
        plt.tight_layout()

        safe_name = str(class_name).replace(" ", "_").replace(",", "").replace("/", "_")
        filename = f"landcover_{lc_code}_{safe_name}_coverage_void.png"
        output_path = results_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        loop_end = time.perf_counter()
        print(
            "TIMING idx={idx}: shapes={shapes:.3f}s overlay={overlay:.3f}s draw={draw:.3f}s total={total:.3f}s -> {path}".format(
                idx=row.Index,
                shapes=t2 - t1,
                overlay=t3 - t2,
                draw=time.perf_counter() - t3,
                total=loop_end - loop_start,
                path=output_path,
            )
        )


__all__ = [
    "PlotColours",
    "plot_landcover_maps",
    "normalise_stats_columns",
]

