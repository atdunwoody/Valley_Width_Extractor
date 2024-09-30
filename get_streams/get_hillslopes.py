from whitebox_workflows import WbEnvironment, download_sample_data
import whitebox
import os 
# Initialize WhiteboxTools
wbt = whitebox.WhiteboxTools()

# Set the environment and path to DEM
wbe = WbEnvironment()
dem = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\dem_2021.tif"

# Set up working directory, usually where your DEM file is located
working_dir = os.path.join(os.path.dirname(dem), "WBT_Outputs")
os.makedirs(working_dir, exist_ok=True)
wbt.set_working_dir(working_dir)

breached_dem = "breached_dem.tif"
filled_dem = "filled_dem.tif"
flow_dir = "flow_direction.tif"
flow_accum = "flow_accumulation.tif"
streams = "streams_10k.tif"
streams_vector = "streams.shp"
hillslopes = "hillslopes_10k.tif"
geomorphons = "geomorphons_filled.tif"
ridges = "ridges_filled.tif"
wbt.hillslopes(flow_dir, streams, hillslopes)

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