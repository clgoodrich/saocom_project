

A short description of naming convention and contents of each folder.
All data are clipped to the same extent. Reference frame is a KML file generate based on SAOCOM data. 

1-  SAOCOM:   Resolution: 10 m, Format: csv, Spatial_crs: EPSG:4326, Description: Extracted heights using SAOCOM images,
2-  COP30:    Resolution: 30 m, Format: tif, Spatial_crs: EPSG:4326, Description: Copernicus dem,
3-  TINItaly: Resolution: 10 m, Format: tif, Spatial_crs: EPSG:4326, Description: Dem downloaded from TINItaly
4-  LCLU:     Resolution: 30 m, Format: tif, Spatial_crs: EPSG:4326, Description: CORINE landcover map.

* InSAR Heights are relative. To make them absolute, the user needs to select a reference point. That's why negative values exist in the .csv file 
* Please reproject the vertical and spatial reference system based on the desired goals of the project