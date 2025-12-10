This document guides Claude Code when assisting with this repository. It preserves the outcomes we care about while allowing creative exploration in how those outcomes are achieved.

0) Modes & Priorities
Modes

Core Mode (safe, deterministic) — Use when modifying production code paths or preparing release artifacts.

Creative Mode (explore, iterate) — Use for proposing/refactoring methods that might improve speed, accuracy, or clarity. New ideas live under experiments/ unless and until they pass the Outcome Contract.

Claude defaults to Creative Mode unless the task is explicitly labeled production-critical (release, grading, publication draft). When in doubt, start creative but publish Core-compatible outputs.

Instruction Hierarchy

Outcome Contract (Section 1)

Data & Paths (Section 2)

Guardrails (Section 6)

Everything else is flexible (use judgment; ask via NEED_INFO if blocked)

Rule of thumb: If creativity threatens the Outcome Contract or data integrity, switch to Core Mode.

1) Outcome Contract (Non-negotiable)

Claude must deliver the same types of outcomes and folders no matter the method:

Primary Notebook: saocom_v3.ipynb runs top-to-bottom without reordering and completes without exceptions.

Required Artifacts (same names/locations unless agreed via CHANGE_PROPOSAL):

results/ — processed tables/grids/caches used in figures & stats

images/ — residual maps, hist/violin, Bland–Altman, 3D terrain

docs/ — documentation/slides

topography_outputs/ — slope/aspect/curvature derivatives

Core Metrics are present and interpretable for both global and stratified views:

Bias, NMAD, RMSE of residuals = SAOCOM − Reference DEM

Stratification by slope, aspect, elevation, land cover

Calibration happens before residual computation (SAOCOM heights are relative).

Outlier handling precedes final stats/plots (method is flexible).

CRS/transform correctness is explicit and validated; no silent reprojections.

If a creative change deviates (e.g., new filenames), Claude must include a CHANGE_PROPOSAL and a compatibility shim so saocom_v3.ipynb still runs unchanged.

2) Data & Paths (Authoritative)

All inputs share a common spatial extent in EPSG:4326 (WGS84).

Inputs

data/saocom_csv/ — SAOCOM point CSVs (10 m spacing), relative heights

data/copernicus.tif, data/demCOP30.tif — Copernicus DEM (30 m)

data/tinitaly/tinitaly_crop.tif — TINItaly DEM (10 m)

data/corine/, data/ground_cover/ — CORINE rasters (~30 m) + DBF lookup

data/sentinel_data/ — Sentinel-2 RGB for visualization

Outputs

results/, images/, docs/, topography_outputs/

Primary workflow: saocom_v3.ipynb (must remain runnable)

Creative freedom: You may add experiments/ and scripts/ subtrees; do not alter data/ in place.

3) Pipeline Summary (What, not How)

Claude is free to change how steps are implemented if the outcomes remain. The canonical sequence is:

Load & QC (outlier-ish coordinates, datum checks)

Geometric prep (resample to 10 m if needed, convex-hull mask, sample DEM at points)

Calibrate SAOCOM to reference (median offset or proposed variant)

Outlier handling (Isolation Forest/IQR/KNN—or a creative combo)

Land cover sampling & hierarchy (Level 1/2/3)

Residual stats (Bias/NMAD/RMSE, stratified by terrain & land cover)

Visualizations (maps/plots/Bland–Altman/3D)

Creative freedom: propose alternative estimators (e.g., Huber, Theil–Sen), robust resampling/stacking strategies, or better plots—so long as the Outcome Contract is met.

4) Creative Exploration Tracks

Track A — Canonical
Leaves existing functions/API intact. Improves internals (vectorization, memory, readability, stability).

Track B — Experimental
New methods, heuristics, or libraries go to experiments/<slug>/. Include:

README.md (what/why/how)

*_experiment.ipynb or script.py

RESULT_PARITY.json (see Section 5)

Any generated figures in experiments/<slug>/images/

Claude may propose promotion of Track B work into production only if it passes Result Parity and does not degrade usability.

5) Result Parity (Promotion Gate)

To move an experiment into production, meet these minimums versus the current Canonical run:

Functional parity: All required artifacts produced; notebook runs clean.

Metric parity or improvement:

NMAD: ≤ baseline + 2% (or lower is better)

RMSE: ≤ baseline + 2% (or lower is better)

Bias: absolute value ≤ baseline + 2% (or closer to 0 is better)

Visual parity: All expected plots exist and render without clipped extents or axis errors.

Runtime: Not > 2× baseline unless justified by material accuracy gains.

Record baseline vs candidate in experiments/<slug>/RESULT_PARITY.json.

6) Guardrails (Minimal, but Firm)

MUST

Keep CRS and Affine transform explicit; verify before writing outputs.

Use np.nan in memory for NODATA; preserve file NODATA on write.

Avoid destructive edits to data/. Write new artifacts to results/, images/, experiments/.

Calibrate before residuals; filter outliers before final stats/plots.

SHOULD (soft constraints; may deviate with rationale)

Prefer pure functions; pass crs, transform, nodata, grid_shape.

Log resampling kernels (DEM: bilinear/cubic; labels: nearest/mode).

Provide seeds for randomness in Canonical runs. In Creative runs, you may explore unseeded sweeps but record seeds for chosen candidates.

7) Flexibility Charter

Claude has explicit permission to:

Swap out outlier methods (e.g., LOF, DBSCAN) if Result Parity holds.

Improve visual storytelling (ridge plots, hexbin density, small multiples).

Add acceleration (vectorized ops, chunked raster reads, dask/polars) under experiments/ first.

Suggest new dependencies via CHANGE_PROPOSAL (with install lines + portability notes).

Refactor modules as long as public function names used by the notebook remain stable—or supply a shim and a CHANGE_PROPOSAL.

8) Communication Blocks (Structured, but Lighter)

Claude should use these blocks as needed (combine when useful):

ACTION_PLAN