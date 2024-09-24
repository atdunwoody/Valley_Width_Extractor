from whitebox_workflows import WbEnvironment, download_sample_data
import whitebox
import os 
# Initialize WhiteboxTools
wbt = whitebox.WhiteboxTools()

# Set the environment and path to DEM
wbe = WbEnvironment()
dem_path = r"Y:\ATD\Drone Data Processing\Metashape_Processing\Bennett\062023-062022\UW\UW_062023-062022 Exports\UW_062023-062022____062023_PostError_PCFiltered_DEM.tif"

# Set up working directory, usually where your DEM file is located
working_dir = os.path.join(os.path.dirname(dem_path), "WBT_Outputs")
if not os.path.exists(working_dir):
    os.makedirs(working_dir)
wbt.set_working_dir(working_dir)



# Fill depressions in the resampled DEM
filled_dem = "filled_dem.tif"
wbt.fill_depressions(dem_path, filled_dem)

# Calculate flow direction on the filled DEM
flow_dir = "flow_direction.tif"
wbt.d8_pointer(filled_dem, flow_dir)

# Calculate flow accumulation
flow_accum = "flow_accumulation.tif"
wbt.d8_flow_accumulation(filled_dem, flow_accum)

# Define streams from flow accumulation
streams = "streams.tif"
threshold = 80000  # Define your threshold value for stream extraction
wbt.extract_streams(flow_accum, streams, threshold)


