# Data Folder Organization

This folder contains all input data for the SAOCOM DEM validation analysis. All data are clipped to the same spatial extent in **EPSG:4326** (WGS84).

## 📁 Folder Structure

### Root Level Files
- **demCOP30.tif** - Copernicus DEM (30m resolution, EPSG:4326)
- **Readme.txt** - Original data description

### Subfolders

#### `saocom_csv/` - SAOCOM Point Data
- **Resolution**: 10 m spacing
- **Format**: CSV
- **CRS**: EPSG:4326
- **Description**: Extracted heights using SAOCOM InSAR images
- **Note**: Heights are **relative** and require calibration to a reference point (negative values are expected)

**Files:**
- `verona_mstgraph_ASI056_weighted_Tcoh00_Bn0_202307-202507.csv` - Current SAOCOM data

#### `copernicus/` - Copernicus DEM
- **Resolution**: 30 m
- **Format**: GeoTIFF
- **CRS**: EPSG:4326
- **Description**: Copernicus Global DEM (GLO-30)

**Files:**
- `GLO30.tif` - Copernicus DEM tile

#### `tinitaly/` - TINItaly Reference DEM
- **Resolution**: 10 m
- **Format**: GeoTIFF
- **CRS**: EPSG:4326
- **Description**: High-accuracy DEM from INGV (Istituto Nazionale di Geofisica e Vulcanologia)

**Files:**
- `tinitaly_crop.tif` - Clipped TINItaly DEM
- `tinitaly_crop.tif.aux.xml` - Auxiliary metadata

#### `corine/` - CORINE Land Cover (Clipped)
- **Resolution**: ~100 m (resampled to match analysis extent)
- **Format**: GeoTIFF with DBF attribute table
- **CRS**: EPSG:4326
- **Description**: Clipped CORINE land cover classification

**Files:**
- `corine_clip.tif` - Clipped land cover raster
- `corine_clip.tif.vat.dbf` - Value Attribute Table with land cover classes
- `corine_clip.tfw`, `corine_clip.tif.aux.xml`, `corine_clip.tif.vat.cpg` - Metadata
- `U2018_CLC2018_V2020_20u1.tif` - Full CORINE 2018 dataset (197 MB)
- `U2018_CLC2018_V2020_20u1.tif.vat.dbf` - Full dataset attribute table
- Various auxiliary files (.tfw, .aux.xml, .ovr, .vat.cpg, .xml)

#### `ground_cover/` - Additional Land Cover Data
- **Resolution**: ~30 m
- **Format**: GeoTIFF with DBF attribute table
- **CRS**: EPSG:4326

**Files:**
- `land_cover_clipped.tif` - Alternative land cover classification
- `land_cover_clipped.tif.vat.dbf` - Attribute table
- `lclu_crop.tif` - Land cover/land use classification
- `lidar_crop.tif` - LiDAR-derived land cover
- Associated metadata files (.tfw, .aux.xml, .ovr, .vat.cpg, .xml)

#### `sentinel_data/` - Sentinel-2 RGB Imagery
- **Resolution**: Variable (native Sentinel-2)
- **Format**: GeoTIFF (3-band RGB)
- **CRS**: EPSG:4326
- **Description**: True-color composite for visualization

**Files:**
- `Sentinel2Views_Clip.tif` - RGB composite image
- `Sentinel2Views_Clip.tif.ovr` - Pyramid overviews
- Metadata files (.tfw, .aux.xml)

#### `backup csv/` - Archive of CSV Versions
Historical versions of SAOCOM data for reference:
- `verona_fullGraph_ASI056_weighted_Tcoh07_Bn150_202307-202507.csv` (15 MB)
- `verona_fullGraph_weighted_Tcoh07_edited.csv` (5.3 MB)
- `verona_fullGraph_weighted_Tcoh07_edited_older.csv` (4.0 MB)

## 📋 Data Summary

| Dataset | Resolution | Format | Size | Purpose |
|---------|-----------|--------|------|---------|
| SAOCOM | 10 m | CSV | 5.4 MB | Test data (InSAR heights) |
| Copernicus DEM | 30 m | GeoTIFF | 756 KB | Reference DEM |
| TINItaly DEM | 10 m | GeoTIFF | 4.8 MB | High-accuracy reference |
| CORINE (clipped) | ~100 m | GeoTIFF | 19 KB | Land cover stratification |
| CORINE (full) | ~100 m | GeoTIFF | 197 MB | Original full dataset |
| Sentinel-2 RGB | Variable | GeoTIFF | 8.0 MB | Visualization background |

## 🔧 Usage Notes

### Coordinate Reference System
- All data share **EPSG:4326** (WGS84 latitude/longitude)
- TINItaly and some files use UTM Zone 32N internally but are provided in WGS84

### Height Datum
- **SAOCOM**: Relative heights (require calibration to absolute datum)
- **Copernicus**: EGM2008 geoid
- **TINItaly**: Local Italian vertical datum

### Recommended Workflow
1. Load SAOCOM CSV from `saocom_csv/`
2. Load reference DEMs: `demCOP30.tif` (root), `tinitaly/tinitaly_crop.tif`
3. Calibrate SAOCOM to reference (median offset correction)
4. Sample land cover from `corine/` or `ground_cover/`
5. Compute residuals and stratify by terrain/land cover

### File Naming Conventions
- `*_crop.tif` - Clipped to study area extent
- `*_clip.tif` - Clipped to study area extent
- `*.vat.dbf` - Value Attribute Tables for categorical rasters
- `*.aux.xml` - GDAL auxiliary metadata
- `*.ovr` - Pyramid overviews for faster rendering
- `*.tfw` - World files (georeference information)

## 🗂️ Archived Data

Duplicate and old versions have been moved to `_archive/old_data/`:
- `duplicates/` - Files that existed in both root and subfolders
- `old_csv_versions/` - Superseded CSV files
- `reference_images/` - Original reference images (Graph.jpg, Height_rdc.jpg, TemporalCoh.jpg)
- `data.rar` - Original data archive

## 📚 References

- **Copernicus DEM**: EU Copernicus Programme (https://spacedata.copernicus.eu/)
- **TINItaly**: INGV (http://tinitaly.pi.ingv.it/)
- **CORINE**: European Environment Agency (https://land.copernicus.eu/pan-european/corine-land-cover)
- **Sentinel-2**: ESA Copernicus Programme (https://sentinel.esa.int/)
- **SAOCOM**: CONAE - Comisión Nacional de Actividades Espaciales (Argentina)
