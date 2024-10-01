from whitebox_workflows import WbEnvironment, download_sample_data
import whitebox
import os 
# Initialize WhiteboxTools
wbt = whitebox.WhiteboxTools()

# Set the environment and path to DEM
wbe = WbEnvironment()
dem = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\ET_low_LIDAR_2020_1m_DEM.tin.tif"

# Set up working directory, usually where your DEM file is located
working_dir = os.path.join(os.path.dirname(dem), "WBT_Outputs_Low")
os.makedirs(working_dir, exist_ok=True)
wbt.set_working_dir(working_dir)

flow_dir = "flow_direction.tif"
streams = "streams_100k.tif"
hillslopes = "hillslopes_100k.tif"
hillslopes_vector = hillslopes.replace(".tif", ".shp")


wbt.hillslopes(flow_dir, streams, hillslopes)
# Convert Hillslopes Raster to Vector Polygons
wbt.raster_to_vector_polygons(
    i=hillslopes,              # Input raster file
    output=hillslopes_vector,  # Output vector shapefile

)

import geopandas as gpd
gdf = gpd.read_file(os.path.join(working_dir, hillslopes_vector))
hillslopes_vector_out = hillslopes_vector.replace(".shp", ".gpkg")
gdf.to_file(os.path.join(working_dir, hillslopes_vector_out), driver="GPKG")

# geomorphons = "geomorphons_filled.tif"
# ridges = "ridges_filled.tif"
# wbt.geomorphons(
#     filled_dem, 
#     geomorphons, 
#     search=50, 
#     threshold=0.0, 
#     fdist=0, 
#     skip=0, 
#     forms=True, 
#     residuals=False, 
# )

# wbt.find_ridges(filled_dem, ridges, line_thin=True)