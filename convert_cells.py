import json

# Load the current notebook
with open(r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 86: Convert from plt.subplots(1, 2) to two separate figures
cell_86_new = """# Create gridded difference maps (simplified)
print("Creating gridded difference maps...")

# TINItaly grid - create from point residuals
valid_tin_pts = saocom_cleaned[saocom_cleaned['diff_tinitaly'].notna()]
if len(valid_tin_pts) > 0:
    fig1, ax1 = plt.subplots(figsize=(9, 8))
    vmin, vmax = np.percentile(valid_tin_pts['diff_tinitaly'], [2, 98])

    # Create gridded view using hexbin
    hb1 = ax1.hexbin(
        valid_tin_pts.geometry.x,
        valid_tin_pts.geometry.y,
        C=valid_tin_pts['diff_tinitaly'],
        gridsize=100,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        reduce_C_function=np.mean
    )
    plt.colorbar(hb1, ax=ax1, label='Difference (m)')

    # Add hull bounding box
    hull = saocom_cleaned.geometry.unary_union.convex_hull
    hull_gdf = gpd.GeoDataFrame(geometry=[hull], crs=saocom_cleaned.crs)
    hull_gdf.boundary.plot(ax=ax1, color='red', linewidth=2, linestyle='--', label='Study Area Hull')

    ax1.set_title('SAOCOM - TINItaly (Gridded)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle="--", color="gray")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'gridded_comparison_tinitaly.png', dpi=150, bbox_inches='tight')
    plt.show()

# Copernicus grid
valid_cop_pts = saocom_cleaned[saocom_cleaned['diff_copernicus'].notna()]
if len(valid_cop_pts) > 0:
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    vmin2, vmax2 = np.percentile(valid_cop_pts['diff_copernicus'], [2, 98])

    hb2 = ax2.hexbin(
        valid_cop_pts.geometry.x,
        valid_cop_pts.geometry.y,
        C=valid_cop_pts['diff_copernicus'],
        gridsize=100,
        cmap='RdBu_r',
        vmin=vmin2,
        vmax=vmax2,
        reduce_C_function=np.mean
    )
    plt.colorbar(hb2, ax=ax2, label='Difference (m)')
    ax2.set_title('SAOCOM - Copernicus (Gridded)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle="--", color="gray")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'gridded_comparison_copernicus.png', dpi=150, bbox_inches='tight')
    plt.show()
"""

# Cell 98: Convert from plt.subplots(2, 2) to four separate figures
cell_98_new = """# Reference DEM comparison
print("Creating reference DEM comparison...")

# Calculate difference between reference DEMs
dem_diff = tinitaly_10m - copernicus_10m
dem_diff[tinitaly_10m == -9999] = np.nan
dem_diff[copernicus_10m == -9999] = np.nan

# Calculate extent
extent = [
    target_transform.c,
    target_transform.c + target_transform.a * grid_width,
    target_transform.f + target_transform.e * grid_height,
    target_transform.f
]

# Figure 1: TINItaly DEM
fig1, ax1 = plt.subplots(figsize=(9, 8))
tin_plot = tinitaly_10m.copy()
tin_plot[tin_plot == -9999] = np.nan
im1 = ax1.imshow(tin_plot, cmap="terrain", extent=extent, origin="upper")
plt.colorbar(im1, ax=ax1, label="Elevation (m)")
ax1.set_title("TINItaly DEM (10m)", fontsize=14, fontweight="bold")
ax1.set_xlabel("UTM Easting (m)", fontsize=8)
ax1.set_ylabel("UTM Northing (m)", fontsize=8)
ax1.set_aspect("equal", adjustable="box")
ax1.set_xticks([])
ax1.set_yticks([])

# Add scale bar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar = ScaleBar(10, "m", length_fraction=0.25, location="lower right",
                    box_alpha=0.7, scale_loc="top", font_properties={"size": 8})
ax1.add_artist(scalebar)
ax1.grid(True, alpha=0.3, linestyle="--", color="gray")

plt.tight_layout()
plt.savefig(IMAGES_DIR / "reference_dem_tinitaly.png", dpi=150, bbox_inches="tight")
plt.show()

# Figure 2: Copernicus DEM
fig2, ax2 = plt.subplots(figsize=(9, 8))
cop_plot = copernicus_10m.copy()
cop_plot[cop_plot == -9999] = np.nan
im2 = ax2.imshow(cop_plot, cmap="terrain", extent=extent, origin="upper")
plt.colorbar(im2, ax=ax2, label="Elevation (m)")
ax2.set_title("Copernicus DEM (10m)", fontsize=14, fontweight="bold")
ax2.set_aspect("equal", adjustable="box")
ax2.set_xticks([])
ax2.set_yticks([])
scalebar2 = ScaleBar(10, "m", length_fraction=0.25, location="lower right",
                     box_alpha=0.7, scale_loc="top", font_properties={"size": 8})
ax2.add_artist(scalebar2)
ax2.grid(True, alpha=0.3, linestyle="--", color="gray")

plt.tight_layout()
plt.savefig(IMAGES_DIR / "reference_dem_copernicus.png", dpi=150, bbox_inches="tight")
plt.show()

# Figure 3: Difference map
if not np.all(np.isnan(dem_diff)):
    fig3, ax3 = plt.subplots(figsize=(9, 8))
    vmin, vmax = np.nanpercentile(dem_diff, [2, 98])
    im3 = ax3.imshow(dem_diff, extent=extent, origin="upper", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im3, ax=ax3, label="Difference (m)")
    ax3.set_title("TINItaly - Copernicus", fontsize=14, fontweight="bold")
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xticks([])
    ax3.set_yticks([])
    scalebar3 = ScaleBar(10, "m", length_fraction=0.25, location="lower right",
                         box_alpha=0.7, scale_loc="top", font_properties={"size": 8})
    ax3.add_artist(scalebar3)
    ax3.grid(True, alpha=0.3, linestyle="--", color="gray")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "reference_dem_difference.png", dpi=150, bbox_inches="tight")
    plt.show()

# Figure 4: Statistics panel
fig4, ax4 = plt.subplots(figsize=(9, 8))
ax4.axis("off")
stats_text = f\"\"\"Reference DEM Comparison Statistics

TINItaly:
  Resolution: 10m (native)
  Range: [{np.nanmin(tin_plot):.1f}, {np.nanmax(tin_plot):.1f}] m
  Mean: {np.nanmean(tin_plot):.1f} m

Copernicus:
  Resolution: 30m → 10m (resampled)
  Range: [{np.nanmin(cop_plot):.1f}, {np.nanmax(cop_plot):.1f}] m
  Mean: {np.nanmean(cop_plot):.1f} m

Difference (TINItaly - Copernicus):
  Mean: {np.nanmean(dem_diff):.2f} m
  Std: {np.nanstd(dem_diff):.2f} m
  NMAD: {1.4826 * np.nanmedian(np.abs(dem_diff - np.nanmedian(dem_diff))):.2f} m
  Range: [{np.nanmin(dem_diff):.2f}, {np.nanmax(dem_diff):.2f}] m
\"\"\"

ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment="center", family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig(IMAGES_DIR / "reference_dem_stats.png", dpi=150, bbox_inches="tight")
plt.show()
"""

# Cell 107: Convert from gridspec to separate figures
cell_107_new = """# Visualize PCA Results
print("Creating PCA visualizations...\\n")

# Figure 1: Scree plot - Explained variance
fig1, ax1 = plt.subplots(figsize=(8, 6))
var_exp = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(var_exp)
x_pos = np.arange(1, len(var_exp) + 1)

ax1.bar(x_pos, var_exp, alpha=0.7, color="steelblue", label="Individual")
ax1.plot(x_pos, cum_var, color="red", marker="o", linewidth=2, label="Cumulative")
ax1.axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
ax1.set_xlabel("Principal Component", fontsize=11, fontweight="bold")
ax1.set_ylabel("Explained Variance (%)", fontsize=11, fontweight="bold")
ax1.set_title("Scree Plot - Variance Explained by PCs", fontsize=12, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x_pos)

plt.tight_layout()
plt.savefig("images/pca_scree_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# Figure 2: Feature contribution to PC1 (bar plot)
fig2, ax2 = plt.subplots(figsize=(8, 6))
pc1_contrib = loadings["PC1"].abs().sort_values(ascending=True)
colors = ["red" if x < 0 else "blue" for x in loadings.loc[pc1_contrib.index, "PC1"]]
ax2.barh(range(len(pc1_contrib)), loadings.loc[pc1_contrib.index, "PC1"], color=colors, alpha=0.7)
ax2.set_yticks(range(len(pc1_contrib)))
ax2.set_yticklabels(pc1_contrib.index, fontsize=9)
ax2.set_xlabel("Loading on PC1", fontsize=11, fontweight="bold")
ax2.set_title("Feature Contributions to PC1", fontsize=12, fontweight="bold")
ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
ax2.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("images/pca_pc1_contributions.png", dpi=150, bbox_inches="tight")
plt.show()

# Figure 3: Component loadings heatmap
fig3, ax3 = plt.subplots(figsize=(12, 4))
sns.heatmap(loadings.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            cbar_kws={"label": "Loading"}, ax=ax3, vmin=-1, vmax=1)
ax3.set_title("Feature Loadings on Principal Components", fontsize=12, fontweight="bold")
ax3.set_xlabel("Feature", fontsize=11, fontweight="bold")
ax3.set_ylabel("Principal Component", fontsize=11, fontweight="bold")
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("images/pca_loadings_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# Sample data for scatter plots (max 10000 points for performance)
if len(df_pca) > 10000:
    plot_idx = np.random.choice(len(df_pca), 10000, replace=False)
    df_plot = df_pca.iloc[plot_idx]
else:
    df_plot = df_pca

# Figure 4: PC1 vs PC2 - Void zones highlighted
fig4, ax4 = plt.subplots(figsize=(8, 6))
scatter = ax4.scatter(df_plot["PC1"], df_plot["PC2"],
                      c=df_plot["is_void_zone"], cmap="RdYlGn_r",
                      alpha=0.4, s=5, vmin=0, vmax=1)
ax4.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11, fontweight="bold")
ax4.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11, fontweight="bold")
ax4.set_title("PC1 vs PC2 - Void Zones Highlighted", fontsize=12, fontweight="bold")
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label("Void Zone", fontsize=10)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["No", "Yes"])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("images/pca_pc1_vs_pc2.png", dpi=150, bbox_inches="tight")
plt.show()

# Figure 5: PC1 vs PC3 - Void zones highlighted
if n_components >= 3:
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    scatter = ax5.scatter(df_plot["PC1"], df_plot["PC3"],
                          c=df_plot["is_void_zone"], cmap="RdYlGn_r",
                          alpha=0.4, s=5, vmin=0, vmax=1)
    ax5.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=11, fontweight="bold")
    ax5.set_ylabel(f"PC3 ({var_exp[2]:.1f}%)", fontsize=11, fontweight="bold")
    ax5.set_title("PC1 vs PC3 - Void Zones Highlighted", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label("Void Zone", fontsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["No", "Yes"])
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/pca_pc1_vs_pc3.png", dpi=150, bbox_inches="tight")
    plt.show()

# Figure 6: PC2 vs PC3 - Void zones highlighted
if n_components >= 3:
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    scatter = ax6.scatter(df_plot["PC2"], df_plot["PC3"],
                          c=df_plot["is_void_zone"], cmap="RdYlGn_r",
                          alpha=0.4, s=5, vmin=0, vmax=1)
    ax6.set_xlabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=11, fontweight="bold")
    ax6.set_ylabel(f"PC3 ({var_exp[2]:.1f}%)", fontsize=11, fontweight="bold")
    ax6.set_title("PC2 vs PC3 - Void Zones Highlighted", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label("Void Zone", fontsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["No", "Yes"])
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/pca_pc2_vs_pc3.png", dpi=150, bbox_inches="tight")
    plt.show()

# Figure 7: Distribution of PC1 by void zone
fig7, ax7 = plt.subplots(figsize=(8, 6))
df_pca[df_pca["is_void_zone"] == 0]["PC1"].hist(bins=50, alpha=0.6, label="Non-void",
                                                   color="green", ax=ax7, density=True)
df_pca[df_pca["is_void_zone"] == 1]["PC1"].hist(bins=50, alpha=0.6, label="Void",
                                                   color="red", ax=ax7, density=True)
ax7.set_xlabel("PC1 Score", fontsize=11, fontweight="bold")
ax7.set_ylabel("Density", fontsize=11, fontweight="bold")
ax7.set_title("Distribution of PC1 by Void Zone Status", fontsize=12, fontweight="bold")
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("images/pca_pc1_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# Figure 8: Distribution of PC2 by void zone
fig8, ax8 = plt.subplots(figsize=(8, 6))
df_pca[df_pca["is_void_zone"] == 0]["PC2"].hist(bins=50, alpha=0.6, label="Non-void",
                                                   color="green", ax=ax8, density=True)
df_pca[df_pca["is_void_zone"] == 1]["PC2"].hist(bins=50, alpha=0.6, label="Void",
                                                   color="red", ax=ax8, density=True)
ax8.set_xlabel("PC2 Score", fontsize=11, fontweight="bold")
ax8.set_ylabel("Density", fontsize=11, fontweight="bold")
ax8.set_title("Distribution of PC2 by Void Zone Status", fontsize=12, fontweight="bold")
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("images/pca_pc2_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# Figure 9: Summary text box
fig9, ax9 = plt.subplots(figsize=(8, 6))
ax9.axis("off")
summary_text = f\"\"\"PCA Summary

Total samples: {len(df_pca):,}
Void zones: {void_count:,} ({void_pct:.1f}%)

Top 3 Features (by PC1 loading):
\"\"\"
for i, (feat, _) in enumerate(loadings["PC1"].abs().sort_values(ascending=False).head(3).items(), 1):
    summary_text += f"{i}. {feat}\\n"

summary_text += f"\\nVariance Explained:\\n"
for i in range(min(3, n_components)):
    summary_text += f"PC{i+1}: {var_exp[i]:.1f}%\\n"

ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment="center",
         family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

plt.tight_layout()
plt.savefig("images/pca_summary.png", dpi=150, bbox_inches="tight")
print(f"[OK] Saved PCA visualizations to images/ directory")
plt.show()
"""

# Update the cells
nb['cells'][86]['source'] = cell_86_new.split('\n')
nb['cells'][98]['source'] = cell_98_new.split('\n')
nb['cells'][107]['source'] = cell_107_new.split('\n')

# Save the updated notebook
with open(r'C:\users\colto\documents\github\saocom_project\saocom_analysis_clean.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully updated cells 86, 98, and 107")
print("\nCell 86: Converted plt.subplots(1,2) to 2 separate figures")
print("Cell 98: Converted plt.subplots(2,2) to 4 separate figures")
print("Cell 107: Converted gridspec to 9 separate figures")
