from whitebox_workflows import WbEnvironment, download_sample_data
import whitebox
import os 
# Initialize WhiteboxTools
wbt = whitebox.WhiteboxTools()

# Set the environment and path to DEM
wbe = WbEnvironment()
dem_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\dem_2021.tif"

# Set up working directory, usually where your DEM file is located
working_dir = os.path.join(os.path.dirname(dem_path), "WBT_Outputs")
if not os.path.exists(working_dir):
    os.makedirs(working_dir)
wbt.set_working_dir(working_dir)


breached_dem = "breached_dem.tif"
filled_dem = "filled_dem.tif"
flow_dir = "flow_direction.tif"
flow_accum = "flow_accumulation.tif"
threshold_list = [1000000, 200000, 300000]  # Define your threshold value for stream extraction
for threshold in threshold_list:
    streams = f"streams_{int(threshold/1000)}k.tif"
    streams_vector = f"streams_{int(threshold/1000)}k.shp"
    streams_gpkg = os.path.join(working_dir, f"streams_{int(threshold/1000)}k.gpkg")
    search_dist = 10
    # Breach depressions in the DEM
    # wbt.breach_depressions_least_cost(dem_path, breached_dem, search_dist)

    # # Fill depressions in the resampled DEM
    # wbt.fill_depressions(breached_dem, filled_dem)

    # # Calculate flow direction on the filled DEM
    # wbt.d8_pointer(filled_dem, flow_dir)

    # # Calculate flow accumulation
    # wbt.d8_flow_accumulation(filled_dem, flow_accum)

    # Define streams from flow accumulation

    wbt.extract_streams(flow_accum, streams, threshold)
    wbt.raster_streams_to_vector(streams, flow_dir, streams_vector)

    #convert streams.shp to a geopackage and assign the CRS of the DEM

    import geopandas as gpd
    import rasterio
    gdf = gpd.read_file(os.path.join(working_dir, streams_vector))
    with rasterio.open(os.path.join(working_dir, dem_path)) as src:
        crs = src.crs

    #assign the CRS to the geodataframe
    gdf.crs = crs
    gdf.to_file(streams_gpkg, driver="GPKG")