"""High-detail visualization of SAOCOM coverage and void statistics."""

from collections import defaultdict
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib.patches import Patch
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, Polygon, shape


def _iter_polygon_exteriors(
    geoms: Iterable[Polygon],
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield exterior coordinates from Polygon or MultiPolygon geometries."""

    for geom in geoms:
        if geom.is_empty:
            continue

        if isinstance(geom, Polygon):
            if geom.is_valid:
                yield geom.exterior.xy
        elif isinstance(geom, MultiPolygon):
            for sub_geom in geom.geoms:
                if sub_geom.is_empty or not sub_geom.is_valid:
                    continue
                yield sub_geom.exterior.xy


def calculate_landcover_statistics(
    corine_10m: np.ndarray,
    void_mask: np.ndarray,
    grid_size: float,
) -> pd.DataFrame:
    """Compute coverage and void statistics for each land cover code."""

    study_area_pixels = np.sum(corine_10m > 0)

    summary_data: List[Dict[str, float]] = []

    for lc_code in sorted(np.unique(corine_10m[corine_10m > 0])):
        lc_mask = corine_10m == lc_code
        lc_total_pixels = int(np.sum(lc_mask))

        if lc_total_pixels == 0:
            continue

        lc_void_pixels = int(np.sum(lc_mask & void_mask))
        lc_coverage_pixels = lc_total_pixels - lc_void_pixels

        total_area_km2 = lc_total_pixels * (grid_size**2) / 1e6
        coverage_area_km2 = lc_coverage_pixels * (grid_size**2) / 1e6
        void_area_km2 = lc_void_pixels * (grid_size**2) / 1e6

        summary_data.append(
            {
                "LC_Code": lc_code,
                "Total_km2": total_area_km2,
                "Coverage_km2": coverage_area_km2,
                "Void_km2": void_area_km2,
                "Pct_Coverage": 100 * lc_coverage_pixels / lc_total_pixels,
                "Pct_Void": 100 * lc_void_pixels / lc_total_pixels,
                "Pct_of_Study_Area": 100 * lc_total_pixels / study_area_pixels,
            }
        )

    stats_df = pd.DataFrame(summary_data)
    if not stats_df.empty:
        total_void_km2 = stats_df["Void_km2"].sum()
        if total_void_km2 > 0:
            stats_df["Pct_of_Total_Voids"] = (
                100 * stats_df["Void_km2"] / total_void_km2
            )
        else:
            stats_df["Pct_of_Total_Voids"] = 0.0

    return stats_df.sort_values(by="Pct_Void", ascending=False).reset_index(drop=True)


def extract_class_polygons(
    corine_10m: np.ndarray,
    void_mask: np.ndarray,
    target_transform,
) -> Dict[int, Dict[str, List[Polygon]]]:
    """Vectorize coverage and void masks for each land cover class."""

    class_polygons: Dict[int, Dict[str, List[Polygon]]] = defaultdict(
        lambda: {"coverage": [], "void": []}
    )

    unique_classes = np.unique(corine_10m[corine_10m > 0])

    for lc_code in sorted(unique_classes):
        lc_mask = corine_10m == lc_code

        coverage_mask = lc_mask & (~void_mask)
        if np.any(coverage_mask):
            coverage_shapes = shapes(
                coverage_mask.astype(np.uint8),
                mask=coverage_mask,
                transform=target_transform,
            )
            class_polygons[lc_code]["coverage"].extend(
                shape(geom) for geom, val in coverage_shapes if val == 1
            )

        void_class_mask = lc_mask & void_mask
        if np.any(void_class_mask):
            void_shapes = shapes(
                void_class_mask.astype(np.uint8),
                mask=void_class_mask,
                transform=target_transform,
            )
            class_polygons[lc_code]["void"].extend(
                shape(geom) for geom, val in void_shapes if val == 1
            )

    return class_polygons


def render_overview_map(
    corine_display: np.ndarray,
    cmap,
    norm,
    extent,
    class_polygons: Dict[int, Dict[str, List[Polygon]]],
    hull_gdf,
    corine_colors_mpl: Dict[int, tuple],
    corine_classes: Dict[int, str],
    void_mask: np.ndarray,
    study_area_mask: np.ndarray,
    grid_size: float,
    results_dir,
):
    """Render a combined land-cover overview map with void statistics."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor="white")
    ax.set_facecolor("white")

    im = ax.imshow(
        corine_display,
        cmap=cmap,
        norm=norm,
        origin="upper",
        extent=extent,
        alpha=0.2,
    )

    for lc_code, poly_dict in class_polygons.items():
        fill_color = corine_colors_mpl.get(lc_code, (0.5, 0.5, 0.5))
        for x, y in _iter_polygon_exteriors(poly_dict["coverage"]):
            ax.fill(x, y, color=fill_color, alpha=0.15, edgecolor="none", zorder=1)
            ax.plot(x, y, color=fill_color, linewidth=1.5, alpha=0.9, zorder=2)

    hull_gdf.boundary.plot(
        ax=ax, color="black", linewidth=2.5, linestyle="--", zorder=3
    )

    void_area = np.sum(void_mask) * (grid_size**2) / 1e6
    total_area = np.sum(study_area_mask) * (grid_size**2) / 1e6
    pct_void = 100 * np.sum(void_mask) / np.sum(study_area_mask)

    stats_text = (
        f"Void Area: {void_area:.2f} km²\n"
        f"Coverage Area: {total_area - void_area:.2f} km²\n"
        f"Void %: {pct_void:.1f}%\n"
        f"Coverage %: {100 - pct_void:.1f}%"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
            edgecolor="black",
        ),
        zorder=4,
    )

    legend_elements: List[Patch] = []
    for lc_code in sorted(class_polygons.keys()):
        color = corine_colors_mpl.get(lc_code, (0.5, 0.5, 0.5))
        legend_elements.append(
            mpatches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color,
                edgecolor=color,
                alpha=0.3,
                linewidth=2,
                label=f"{lc_code}: {corine_classes.get(lc_code, 'Unknown')}",
            )
        )

    legend_elements.append(
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="gray",
            linewidth=1,
            label="VOID (No SAOCOM Data)",
        )
    )

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=13,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        title="Land Cover Classes",
    )

    ax.set_title(
        "CORINE Land Cover with SAOCOM Void Zones\n(White areas = No coverage)",
        fontweight="bold",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("UTM Easting (m)", fontsize=11)
    ax.set_ylabel("UTM Northing (m)", fontsize=11)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True, alpha=0.3, color="gray", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(
        results_dir / "landcover_with_voids_outlined.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()


def render_void_bar_charts(
    stats_df: pd.DataFrame,
    corine_colors_mpl: Dict[int, tuple],
    corine_classes: Dict[int, str],
    results_dir,
):
    """Create bar charts showing void percentages and contributions."""

    if stats_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor="white")

    top_pct = stats_df.nlargest(15, "Pct_Void")
    bars1 = ax1.barh(range(len(top_pct)), top_pct["Pct_Void"])
    for i, row in enumerate(top_pct.itertuples(index=False)):
        color = corine_colors_mpl.get(row.LC_Code, (0.5, 0.5, 0.5))
        bars1[i].set_color(color)
        bars1[i].set_edgecolor("black")
        bars1[i].set_linewidth(0.5)

    ax1.set_yticks(range(len(top_pct)))
    ax1.set_yticklabels(
        [f"{row.LC_Code}: {corine_classes.get(row.LC_Code, '')[:35]}" for row in top_pct.itertuples(index=False)],
        fontsize=9,
    )
    ax1.set_xlabel("% of Land Cover Class that is Void", fontsize=11)
    ax1.set_title(
        "Land Covers with Highest Void Percentage\n(Worst Coverage Performance)",
        fontweight="bold",
        fontsize=12,
    )
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    ax1.axvline(100, color="black", linestyle="-", linewidth=2, label="100% (Total Void)", zorder=3)
    for pct in [20, 40, 60, 80]:
        ax1.axvline(pct, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=3)
        ax1.text(pct, -0.5, f"{pct}%", ha="center", fontsize=9, color="gray")

    ax1.set_xlim(0, 105)

    if "Pct_of_Total_Voids" not in stats_df.columns:
        total_void = stats_df["Void_km2"].sum()
        stats_df["Pct_of_Total_Voids"] = (
            100 * stats_df["Void_km2"] / total_void if total_void > 0 else 0.0
        )

    top_contrib = stats_df.nlargest(15, "Pct_of_Total_Voids")
    bars2 = ax2.barh(range(len(top_contrib)), top_contrib["Pct_of_Total_Voids"])
    for i, row in enumerate(top_contrib.itertuples(index=False)):
        color = corine_colors_mpl.get(row.LC_Code, (0.5, 0.5, 0.5))
        bars2[i].set_color(color)
        bars2[i].set_edgecolor("black")
        bars2[i].set_linewidth(0.5)

    ax2.set_yticks(range(len(top_contrib)))
    ax2.set_yticklabels(
        [
            f"{row.LC_Code}: {corine_classes.get(row.LC_Code, '')[:35]}"
            for row in top_contrib.itertuples(index=False)
        ],
        fontsize=9,
    )
    ax2.set_xlabel("% of Total Void Area", fontsize=11)
    ax2.set_title(
        "Land Covers Contributing Most to Total Voids\n(Largest Void Areas)",
        fontweight="bold",
        fontsize=12,
    )
    ax2.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(
        results_dir / "voids_by_landcover_charts.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()


def render_per_class_maps(
    stats_df: pd.DataFrame,
    class_polygons: Dict[int, Dict[str, List[Polygon]]],
    sentinel_rgb_norm: np.ndarray,
    extent,
    hull_gdf,
    corine_colors,
    results_dir,
):
    """Generate per-class coverage and void maps using cached polygons."""

    if stats_df.empty:
        return

    print(f"\nGenerating {len(stats_df)} land cover maps using cached polygons...")

    for row in stats_df.itertuples(index=False):
        lc_code = row.LC_Code
        class_name = row.LC_Name if "LC_Name" in stats_df.columns else str(lc_code)

        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor="white")
        ax.set_facecolor("white")

        ax.imshow(
            sentinel_rgb_norm,
            extent=extent,
            origin="upper",
            alpha=0.4,
        )

        fill_color = tuple(c / 255 for c in corine_colors.get(lc_code, (128, 128, 128)))

        for x, y in _iter_polygon_exteriors(class_polygons[lc_code]["coverage"]):
            ax.fill(x, y, color=fill_color, alpha=0.35, edgecolor="none", hatch="///")
            ax.plot(x, y, color="black", linewidth=2.0)
            ax.plot(x, y, color=fill_color, linewidth=1.2)

        for x, y in _iter_polygon_exteriors(class_polygons[lc_code]["void"]):
            ax.fill(x, y, color="white", alpha=0.8, edgecolor="none", hatch="....")
            ax.plot(x, y, color="red", linewidth=1.5, linestyle="--")
            ax.plot(x, y, color="gray", linewidth=0.8, linestyle="--")

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
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor="black",
            ),
        )

        legend_elements = [
            Patch(
                facecolor=fill_color,
                edgecolor="black",
                linewidth=2,
                alpha=0.35,
                hatch="///",
                label=f"{class_name} (Coverage)",
            ),
            Patch(
                facecolor="white",
                edgecolor="red",
                linewidth=1.5,
                alpha=0.8,
                hatch="....",
                linestyle="--",
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
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, alpha=0.3, color="gray", linewidth=0.5)
        plt.tight_layout()

        safe_name = class_name.replace(" ", "_").replace(",", "").replace("/", "_")
        filename = f"landcover_{lc_code}_{safe_name}_coverage_void.png"
        plt.savefig(
            results_dir / filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.show()
        plt.close()

        print(f"Saved: {filename}")


def generate_high_detail_outputs(
    corine_10m: np.ndarray,
    corine_display: np.ndarray,
    cmap,
    norm,
    void_mask: np.ndarray,
    study_area_mask: np.ndarray,
    sentinel_rgb_norm: np.ndarray,
    target_transform,
    extent,
    hull_gdf,
    corine_colors,
    corine_colors_mpl,
    corine_classes,
    grid_size: float,
    results_dir,
):
    """Run the full high-detail visualization workflow with caching."""

    print("Calculating statistics and vectorizing polygons (this may take a moment)...")

    stats_df = calculate_landcover_statistics(corine_10m, void_mask, grid_size)
    stats_df["LC_Name"] = stats_df["LC_Code"].map(
        lambda code: corine_classes.get(code, f"Class {code}")
    )
    column_order = [
        "LC_Code",
        "LC_Name",
        "Total_km2",
        "Coverage_km2",
        "Void_km2",
        "Pct_Coverage",
        "Pct_Void",
        "Pct_of_Study_Area",
    ]
    if "Pct_of_Total_Voids" in stats_df.columns:
        column_order.append("Pct_of_Total_Voids")
    stats_df = stats_df[column_order]

    class_polygons = extract_class_polygons(corine_10m, void_mask, target_transform)

    print("✓ Statistics prepared and polygons cached.")
    print(stats_df.to_string(index=False, float_format="%.2f"))

    render_overview_map(
        corine_display,
        cmap,
        norm,
        extent,
        class_polygons,
        hull_gdf,
        corine_colors_mpl,
        corine_classes,
        void_mask,
        study_area_mask,
        grid_size,
        results_dir,
    )

    render_void_bar_charts(stats_df, corine_colors_mpl, corine_classes, results_dir)

    render_per_class_maps(
        stats_df,
        class_polygons,
        sentinel_rgb_norm,
        extent,
        hull_gdf,
        corine_colors,
        results_dir,
    )

    print("\n✓ High-detail void analysis complete!")

