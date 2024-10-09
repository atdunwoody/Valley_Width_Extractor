# dem_processor.py

from whitebox_workflows import WbEnvironment, download_sample_data
import whitebox
import os
import geopandas as gpd
import rasterio

def get_wbt_streams(
    dem,
    output_dir,
    threshold=100000,
    output_dir_base=None,
    overwrite=False,
    breach_depressions=True,
):
    """
    Processes a list of DEM files to extract streams and convert them to GeoPackage format.

    Parameters:
    - dem (str): Path to the DEM file.
    - output_dir (str): Path to the output directory.
    - threshold (int, optional): Threshold value for stream extraction from flow accumulation raster. Default is 100000.
    - output_dir_base (str, optional): Base directory for output. Defaults to the DEM file's directory.
    - overwrite (bool, optional): If True, existing output directories will be overwritten. Default is False.

    Returns:
    - None
    """

    # Initialize WhiteboxTools
    wbt = whitebox.WhiteboxTools()

    # Set the environment
    wbe = WbEnvironment()

    # Create working directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif overwrite:
        # Clear existing directory
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)

    # wbt.set_working_dir(output_dir)

    # Define output filenames

    filled_dem = os.path.join(output_dir, "filled_dem.tif")
    d8_pointer = os.path.join(output_dir, "d8_pointer.tif")
    flow_accum = os.path.join(output_dir, "flow_accumulation.tif")
    breached_dem = os.path.join(output_dir, "breached_dem.tif")
    
    # Check if the file path exists
    if not os.path.exists(breached_dem):
        wbt.breach_depressions_least_cost(dem, breached_dem, 10)
    # Fill depressions in the DEM
    if not os.path.exists(filled_dem):
        wbt.fill_depressions(dem, filled_dem)
    # Calculate flow direction on the filled DEM
    if breach_depressions and not os.path.exists(d8_pointer) and not os.path.exists(flow_accum): 
        wbt.d8_pointer(breached_dem, d8_pointer)
        wbt.d8_flow_accumulation(breached_dem, flow_accum)
    elif not os.path.exists(d8_pointer) and not os.path.exists(flow_accum):
        wbt.d8_pointer(filled_dem, d8_pointer)
        wbt.d8_flow_accumulation(filled_dem, flow_accum)
    


    streams_raster = os.path.join(output_dir, f"streams_{int(threshold/1000)}k.tif")
    streams_vector = streams_raster.replace(".tif", ".shp")
    streams_gpkg = streams_raster.replace(".tif", ".gpkg")
    if not os.path.exists(streams_raster):
        wbt.extract_streams(flow_accum, streams_raster, threshold)

    # Convert raster streams to vector
    if not os.path.exists(streams_vector):
        wbt.raster_streams_to_vector(streams_raster, d8_pointer, streams_vector)

        # Assign CRS from DEM to the GeoDataFrame and save as GeoPackage
        gdf = gpd.read_file(streams_vector)
        with rasterio.open(dem) as src:
            crs = src.crs

        gdf = gdf.to_crs(crs)
        gdf.to_file(streams_gpkg, driver="GPKG")


if __name__ == "__main__":
    dem = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\ET_middle_LIDAR_2020_1m_DEM.tin.tif",
    output_dir = r"Y:\ATD\GIS\ETF\WBT_Outputs"
    get_wbt_streams(dem, output_dir)
