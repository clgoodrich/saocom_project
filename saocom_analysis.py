import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.transform import from_bounds, rowcol
from shapely.geometry import Point
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import seaborn as sns

TARGET_CRS = "EPSG:32632"
GRID_SIZE = 10
NODATA = -9999
COHERENCE_THRESHOLD = 0.7
MIN_SAMPLES = 50
CORINE_CLASSES = {
    111: "Continuous urban fabric",
    112: "Discontinuous urban fabric",
    121: "Industrial units",
    122: "Road and rail networks",
    123: "Port areas",
    124: "Airports",
    131: "Mineral extraction sites",
    132: "Dump sites",
    133: "Construction sites",
    141: "Green urban areas",
    142: "Sport and leisure facilities",
    211: "Non-irrigated arable land",
    212: "Permanently irrigated land",
    213: "Rice fields",
    221: "Vineyards",
    222: "Fruit trees and berry plantations",
    223: "Olive groves",
    231: "Pastures",
    241: "Annual crops associated with permanent crops",
    242: "Complex cultivation patterns",
    243: "Agriculture with natural vegetation",
    244: "Agro-forestry areas",
    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    321: "Natural grasslands",
    322: "Moors and heathland",
    323: "Sclerophyllous vegetation",
    324: "Transitional woodland/shrub",
    331: "Beaches, dunes, sands",
    332: "Bare rocks",
    333: "Sparsely vegetated areas",
    334: "Burnt areas",
    335: "Glaciers and perpetual snow",
    411: "Inland marshes",
    412: "Peat bogs",
    421: "Salt marshes",
    422: "Salines",
    423: "Intertidal flats",
    511: "Water courses",
    512: "Water bodies",
    521: "Coastal lagoons",
    522: "Estuaries",
    523: "Sea and ocean"
}

slope_bins = [0, 5, 15, 30, 90]
slope_labels = ["0-5°", "5-15°", "15-30°", ">30°"]
aspect_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
aspect_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
coherence_bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
coherence_labels = ["0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalized_mad(values):
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return np.nan
    med = np.median(vals)
    return 1.4826 * np.median(np.abs(vals - med))


def to_serializable(value):
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return None if np.isnan(val) else val
    if isinstance(value, np.ndarray):
        return [to_serializable(v) for v in value.tolist()]
    if pd.isna(value):
        return None
    return value


def compute_metrics(diff, reference=None, estimate=None):
    mask = ~np.isnan(diff)
    vals = diff[mask]
    if vals.size == 0:
        return {k: np.nan for k in [
            "n_points", "mean_diff", "median_diff", "std_diff", "rmse", "mae", "nmad", "correlation"
        ]}
    corr = np.nan
    if reference is not None and estimate is not None:
        ref = np.asarray(reference)
        est = np.asarray(estimate)
        shared = mask & ~np.isnan(ref) & ~np.isnan(est)
        if np.any(shared):
            corr = float(np.corrcoef(ref[shared], est[shared])[0, 1])
    return {
        "n_points": int(vals.size),
        "mean_diff": float(np.mean(vals)),
        "median_diff": float(np.median(vals)),
        "std_diff": float(np.std(vals, ddof=0)),
        "rmse": float(np.sqrt(np.mean(vals ** 2))),
        "mae": float(np.mean(np.abs(vals))),
        "nmad": float(normalized_mad(vals)),
        "correlation": corr
    }


def compute_group_stats(data, group_col, diff_col):
    grouped = data.dropna(subset=[group_col, diff_col]).groupby(group_col)
    stats = grouped[diff_col].agg(
        N_Points="size",
        Median_Diff_m=lambda x: float(np.median(x)),
        NMAD_m=lambda x: float(normalized_mad(x.values)),
        Mean_Diff_m=lambda x: float(np.mean(x)),
        Std_Dev_m=lambda x: float(np.std(x, ddof=0)),
        RMSE_m=lambda x: float(np.sqrt(np.mean(x ** 2)))
    ).reset_index()
    return stats


def resample_raster(path, transform, shape, target_crs, nodata, resampling):
    dst = np.full(shape, nodata, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=target_crs,
            dst_nodata=nodata,
            resampling=resampling
        )
    dst = np.where(dst == nodata, np.nan, dst)
    return dst


def sample_raster(array, transform, xs, ys):
    rows, cols = rowcol(transform, xs, ys)
    valid = (
        (rows >= 0) & (rows < array.shape[0]) &
        (cols >= 0) & (cols < array.shape[1])
    )
    values = np.full(xs.shape, np.nan, dtype=np.float32)
    values[valid] = array[rows[valid], cols[valid]]
    return values, rows, cols, valid


def compute_slope_aspect(dem):
    filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    gy, gx = np.gradient(filled, GRID_SIZE, GRID_SIZE)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360
    slope[np.isnan(dem)] = np.nan
    aspect[np.isnan(dem)] = np.nan
    return slope, aspect


def coverage_statistics(coverage, mask, corine):
    void_mask = mask & (~coverage)
    total_void = np.nan if mask.sum() == 0 else float(void_mask.sum() / mask.sum() * 100)
    records = []
    unique_codes = [c for c in np.unique(corine[mask]) if not np.isnan(c)]
    for code in unique_codes:
        code_mask = mask & (corine == code)
        if not code_mask.any():
            continue
        covered = (coverage & code_mask).sum()
        void = (code_mask & (~coverage)).sum()
        coverage_pct = float(covered / code_mask.sum() * 100)
        records.append({
            "corine_code": int(code),
            "LC_Label": CORINE_CLASSES.get(int(code), "Unknown"),
            "Covered_pct": coverage_pct,
            "Void_pct": 100 - coverage_pct,
            "n_cells": int(code_mask.sum())
        })
    return pd.DataFrame(records), total_void, void_mask


def local_void_clusters(void_mask):
    def nanmean(vals):
        valid = vals[~np.isnan(vals)]
        return np.nan if valid.size == 0 else float(valid.mean())
    return generic_filter(void_mask.astype(float), nanmean, size=5, mode="constant", cval=np.nan)


def make_plots(output_dir, saocom_gdf, lc_stats, threshold_stats, residual_grid_tinitaly, void_mask, transform):
    sns.set_theme(style="whitegrid", palette="colorblind")

    if not lc_stats.empty:
        plt.figure(figsize=(10, 6))
        order = lc_stats.sort_values("NMAD_m").head(10)["corine_code"].astype(str).tolist()
        subset = saocom_gdf.dropna(subset=["corine_code"]).copy()
        subset["corine_code"] = subset["corine_code"].astype(int).astype(str)
        sns.boxplot(data=subset[subset["corine_code"].isin(order)], x="corine_code", y="diff_tinitaly", order=order)
        plt.xlabel("CORINE code")
        plt.ylabel("Height residual (m)")
        plt.title("TINITALY residuals by dominant land cover")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "boxplot_residuals_landcover.png"), dpi=300)
        plt.close()

    if threshold_stats is not None and not threshold_stats.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(threshold_stats["coherence_threshold"], threshold_stats["RMSE_m"], marker="o")
        plt.xlabel("Coherence threshold")
        plt.ylabel("RMSE (m)")
        plt.title("RMSE vs coherence threshold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rmse_vs_coherence.png"), dpi=300)
        plt.close()

    plt.figure(figsize=(10, 8))
    clipped = np.clip(residual_grid_tinitaly, -5, 5)
    plt.imshow(clipped, cmap="coolwarm", vmin=-5, vmax=5)
    plt.colorbar(label="Residual (m)")
    plt.title("SAOCOM - TINITALY residual map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_map_tinitaly.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(void_mask, cmap="Greys")
    plt.title("SAOCOM coverage voids")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "void_zones.png"), dpi=300)
    plt.close()


def run_analysis(saocom_path, tinitaly_path, copernicus_path, corine_path, output_dir="results"):
    ensure_dir(output_dir)

    df = pd.read_csv(saocom_path)
    df = df[df["COHER"] >= COHERENCE_THRESHOLD].copy()
    geometry = gpd.GeoSeries([Point(xy) for xy in zip(df["LON2"], df["LAT2"])]).set_crs("EPSG:4326")
    saocom_gdf = gpd.GeoDataFrame(df, geometry=geometry).to_crs(TARGET_CRS)
    saocom_gdf["x_utm"] = saocom_gdf.geometry.x
    saocom_gdf["y_utm"] = saocom_gdf.geometry.y

    minx = np.floor(saocom_gdf["x_utm"].min() / GRID_SIZE) * GRID_SIZE
    maxx = np.ceil(saocom_gdf["x_utm"].max() / GRID_SIZE) * GRID_SIZE
    miny = np.floor(saocom_gdf["y_utm"].min() / GRID_SIZE) * GRID_SIZE
    maxy = np.ceil(saocom_gdf["y_utm"].max() / GRID_SIZE) * GRID_SIZE
    width = int((maxx - minx) / GRID_SIZE)
    height = int((maxy - miny) / GRID_SIZE)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    tinitaly_10m = resample_raster(tinitaly_path, transform, (height, width), TARGET_CRS, NODATA, Resampling.bilinear)
    copernicus_10m = resample_raster(copernicus_path, transform, (height, width), TARGET_CRS, NODATA, Resampling.bilinear)
    corine_10m = resample_raster(corine_path, transform, (height, width), TARGET_CRS, NODATA, Resampling.nearest)

    tin_vals, rows, cols, valid = sample_raster(tinitaly_10m, transform, saocom_gdf["x_utm"].values, saocom_gdf["y_utm"].values)
    cop_vals, _, _, _ = sample_raster(copernicus_10m, transform, saocom_gdf["x_utm"].values, saocom_gdf["y_utm"].values)
    cor_vals, _, _, _ = sample_raster(corine_10m, transform, saocom_gdf["x_utm"].values, saocom_gdf["y_utm"].values)

    saocom_gdf["tinitaly_height"] = tin_vals
    saocom_gdf["copernicus_height"] = cop_vals
    saocom_gdf["corine_code"] = cor_vals

    valid_tin = ~np.isnan(saocom_gdf["tinitaly_height"])
    valid_cop = ~np.isnan(saocom_gdf["copernicus_height"])

    offset_tinitaly = float(np.median(saocom_gdf.loc[valid_tin, "HEIGHT"] - saocom_gdf.loc[valid_tin, "tinitaly_height"]))
    offset_copernicus = float(np.median(saocom_gdf.loc[valid_cop, "HEIGHT"] - saocom_gdf.loc[valid_cop, "copernicus_height"]))

    saocom_gdf["HEIGHT_ABSOLUTE_TIN"] = saocom_gdf["HEIGHT"] - offset_tinitaly
    saocom_gdf["HEIGHT_ABSOLUTE_COP"] = saocom_gdf["HEIGHT"] - offset_copernicus

    saocom_gdf["diff_tinitaly"] = saocom_gdf["HEIGHT_ABSOLUTE_TIN"] - saocom_gdf["tinitaly_height"]
    saocom_gdf["diff_copernicus"] = saocom_gdf["HEIGHT_ABSOLUTE_COP"] - saocom_gdf["copernicus_height"]

    slope_10m, aspect_10m = compute_slope_aspect(tinitaly_10m)
    slope_vals, _, _, _ = sample_raster(slope_10m, transform, saocom_gdf["x_utm"].values, saocom_gdf["y_utm"].values)
    aspect_vals, _, _, _ = sample_raster(aspect_10m, transform, saocom_gdf["x_utm"].values, saocom_gdf["y_utm"].values)
    saocom_gdf["slope"] = slope_vals
    saocom_gdf["aspect"] = aspect_vals
    saocom_gdf["slope_bin"] = pd.cut(saocom_gdf["slope"], bins=slope_bins, labels=slope_labels, include_lowest=True, right=False)
    saocom_gdf["aspect_bin"] = pd.cut(saocom_gdf["aspect"], bins=aspect_bins, labels=aspect_labels, include_lowest=True, right=False)

    valid_elev = saocom_gdf["tinitaly_height"].dropna()
    if valid_elev.empty:
        elevation_bins = [0, 1]
    else:
        quantiles = np.unique(np.quantile(valid_elev, [0, 0.25, 0.5, 0.75, 1]))
        if quantiles.size < 2:
            elevation_bins = [quantiles[0], quantiles[0] + 1]
        else:
            elevation_bins = quantiles
    saocom_gdf["elevation_bin"] = pd.cut(saocom_gdf["tinitaly_height"], bins=elevation_bins, include_lowest=True)

    saocom_gdf["coherence_bin"] = pd.cut(
        saocom_gdf["COHER"], bins=coherence_bins, labels=coherence_labels, right=True, include_lowest=True
    )

    lc_stats = compute_group_stats(saocom_gdf, "corine_code", "diff_tinitaly")
    lc_stats["corine_code"] = lc_stats["corine_code"].astype(int)
    lc_stats["LC_Label"] = lc_stats["corine_code"].apply(lambda c: CORINE_CLASSES.get(int(c), "Unknown"))
    lc_stats = lc_stats[["corine_code", "LC_Label", "N_Points", "Median_Diff_m", "NMAD_m", "Mean_Diff_m", "Std_Dev_m", "RMSE_m"]]
    lc_height_stats_filtered = lc_stats[lc_stats["N_Points"] >= MIN_SAMPLES].copy()

    slope_stats = compute_group_stats(saocom_gdf, "slope_bin", "diff_tinitaly")
    aspect_stats = compute_group_stats(saocom_gdf, "aspect_bin", "diff_tinitaly")
    elevation_stats = compute_group_stats(saocom_gdf, "elevation_bin", "diff_tinitaly")

    coherence_stats = compute_group_stats(saocom_gdf, "coherence_bin", "diff_tinitaly")
    coherence_stats["coherence_bin"] = coherence_stats["coherence_bin"].astype(str)
    mid_map = {
        coherence_labels[i]: (coherence_bins[i] + coherence_bins[i + 1]) / 2 for i in range(len(coherence_labels))
    }
    coherence_stats["coherence_mid"] = coherence_stats["coherence_bin"].map(mid_map)
    coherence_stats = coherence_stats[[
        "coherence_bin", "coherence_mid", "N_Points", "Median_Diff_m", "NMAD_m", "Mean_Diff_m", "Std_Dev_m", "RMSE_m"
    ]]

    threshold_records = []
    for thresh in coherence_bins[:-1]:
        subset = saocom_gdf[saocom_gdf["COHER"] >= thresh]
        if subset.empty:
            continue
        metrics = compute_metrics(
            subset["diff_tinitaly"].values,
            reference=subset["tinitaly_height"].values,
            estimate=subset["HEIGHT_ABSOLUTE_TIN"].values
        )
        threshold_records.append({
            "coherence_threshold": float(thresh),
            "N_Points": metrics["n_points"],
            "Median_Diff_m": metrics["median_diff"],
            "NMAD_m": metrics["nmad"],
            "Mean_Diff_m": metrics["mean_diff"],
            "Std_Dev_m": metrics["std_diff"],
            "RMSE_m": metrics["rmse"]
        })
    threshold_stats = pd.DataFrame(threshold_records)
    if not threshold_stats.empty:
        threshold_stats = threshold_stats[[
            "coherence_threshold", "N_Points", "Median_Diff_m", "NMAD_m", "Mean_Diff_m", "Std_Dev_m", "RMSE_m"
        ]]

    coverage = np.zeros((height, width), dtype=bool)
    coverage[rows[valid], cols[valid]] = True
    study_area_mask = ~np.isnan(tinitaly_10m)
    coverage_by_lc, void_percentage_global, void_mask = coverage_statistics(coverage, study_area_mask, corine_10m)
    void_cluster_map = local_void_clusters(void_mask)

    residual_grid_tinitaly = np.full((height, width), np.nan, dtype=np.float32)
    residual_grid_copernicus = np.full((height, width), np.nan, dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(residual_grid_tinitaly, (rows[valid], cols[valid]), saocom_gdf.loc[valid, "diff_tinitaly"].values)
    np.add.at(residual_grid_copernicus, (rows[valid], cols[valid]), saocom_gdf.loc[valid, "diff_copernicus"].values)
    np.add.at(counts, (rows[valid], cols[valid]), 1)
    mask_counts = counts > 0
    residual_grid_tinitaly[mask_counts] = residual_grid_tinitaly[mask_counts] / counts[mask_counts]
    residual_grid_copernicus[mask_counts] = residual_grid_copernicus[mask_counts] / counts[mask_counts]

    saocom_tinitaly_diff = saocom_gdf["diff_tinitaly"].values
    saocom_copernicus_diff = saocom_gdf["diff_copernicus"].values
    saocom_tinitaly_metrics = compute_metrics(
        saocom_tinitaly_diff,
        reference=saocom_gdf["tinitaly_height"].values,
        estimate=saocom_gdf["HEIGHT_ABSOLUTE_TIN"].values
    )
    saocom_copernicus_metrics = compute_metrics(
        saocom_copernicus_diff,
        reference=saocom_gdf["copernicus_height"].values,
        estimate=saocom_gdf["HEIGHT_ABSOLUTE_COP"].values
    )
    ref_diff = tinitaly_10m - copernicus_10m
    ref_metrics = compute_metrics(
        ref_diff.ravel(),
        reference=copernicus_10m.ravel(),
        estimate=tinitaly_10m.ravel()
    )

    tolerance_tin = saocom_tinitaly_metrics["nmad"] if not np.isnan(saocom_tinitaly_metrics["nmad"]) else 0.0
    tolerance_cop = saocom_copernicus_metrics["nmad"] if not np.isnan(saocom_copernicus_metrics["nmad"]) else 0.0
    higher_tin = np.sum(saocom_tinitaly_diff > tolerance_tin)
    lower_tin = np.sum(saocom_tinitaly_diff < -tolerance_tin)
    equal_tin = np.sum((saocom_tinitaly_diff >= -tolerance_tin) & (saocom_tinitaly_diff <= tolerance_tin))
    higher_cop = np.sum(saocom_copernicus_diff > tolerance_cop)
    lower_cop = np.sum(saocom_copernicus_diff < -tolerance_cop)
    equal_cop = np.sum((saocom_copernicus_diff >= -tolerance_cop) & (saocom_copernicus_diff <= tolerance_cop))

    tin_total = higher_tin + equal_tin + lower_tin
    cop_total = higher_cop + equal_cop + lower_cop
    tin_confusion = pd.DataFrame({
        "Category": ["SAOCOM higher", "Within tolerance", "SAOCOM lower"],
        "Count": [int(higher_tin), int(equal_tin), int(lower_tin)]
    })
    tin_confusion["Percentage"] = tin_confusion["Count"].apply(lambda v: float(v / tin_total * 100) if tin_total else np.nan)
    cop_confusion = pd.DataFrame({
        "Category": ["SAOCOM higher", "Within tolerance", "SAOCOM lower"],
        "Count": [int(higher_cop), int(equal_cop), int(lower_cop)]
    })
    cop_confusion["Percentage"] = cop_confusion["Count"].apply(lambda v: float(v / cop_total * 100) if cop_total else np.nan)

    export_gdf = saocom_gdf.copy()
    export_gdf["corine_code"] = export_gdf["corine_code"].astype(float)
    for cat_col in ["slope_bin", "aspect_bin", "elevation_bin", "coherence_bin"]:
        if cat_col in export_gdf:
            export_gdf[cat_col] = export_gdf[cat_col].astype(str)
    export_gdf.to_file(os.path.join(output_dir, "saocom_points.gpkg"), driver="GPKG")

    lc_stats.to_csv(os.path.join(output_dir, "landcover_stats_all.csv"), index=False)
    lc_height_stats_filtered.to_csv(os.path.join(output_dir, "landcover_stats_filtered.csv"), index=False)
    slope_stats.to_csv(os.path.join(output_dir, "slope_stats.csv"), index=False)
    aspect_stats.to_csv(os.path.join(output_dir, "aspect_stats.csv"), index=False)
    elevation_stats.to_csv(os.path.join(output_dir, "elevation_stats.csv"), index=False)
    coherence_stats.to_csv(os.path.join(output_dir, "coherence_bin_stats.csv"), index=False)
    threshold_stats.to_csv(os.path.join(output_dir, "coherence_threshold_stats.csv"), index=False)
    coverage_by_lc.to_csv(os.path.join(output_dir, "coverage_by_landcover.csv"), index=False)
    tin_confusion.to_csv(os.path.join(output_dir, "confusion_tinitaly.csv"), index=False)
    cop_confusion.to_csv(os.path.join(output_dir, "confusion_copernicus.csv"), index=False)

    np.save(os.path.join(output_dir, "tinitaly_residual_grid.npy"), residual_grid_tinitaly)
    np.save(os.path.join(output_dir, "copernicus_residual_grid.npy"), residual_grid_copernicus)
    np.save(os.path.join(output_dir, "coverage_grid.npy"), coverage.astype(np.uint8))
    np.save(os.path.join(output_dir, "void_cluster_map.npy"), void_cluster_map)

    summary = {
        "offset_tinitaly": offset_tinitaly,
        "offset_copernicus": offset_copernicus,
        "saocom_tinitaly_metrics": saocom_tinitaly_metrics,
        "saocom_copernicus_metrics": saocom_copernicus_metrics,
        "ref_metrics": ref_metrics,
        "void_percentage_global": void_percentage_global
    }
    if not threshold_stats.empty:
        best_row = threshold_stats.loc[threshold_stats["RMSE_m"].idxmin()]
        summary["optimal_coherence_threshold"] = float(best_row["coherence_threshold"])
        summary["optimal_threshold_rmse"] = float(best_row["RMSE_m"])
    with open(os.path.join(output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2, allow_nan=True)

    make_plots(output_dir, saocom_gdf, lc_height_stats_filtered, threshold_stats, residual_grid_tinitaly, void_mask, transform)

    return {
        "summary": summary,
        "landcover_stats": lc_height_stats_filtered,
        "slope_stats": slope_stats,
        "aspect_stats": aspect_stats,
        "elevation_stats": elevation_stats,
        "coherence_stats": coherence_stats,
        "coherence_threshold_stats": threshold_stats,
        "coverage": coverage_by_lc
    }


__all__ = [
    "run_analysis",
    "TARGET_CRS",
    "GRID_SIZE",
    "COHERENCE_THRESHOLD"
]
