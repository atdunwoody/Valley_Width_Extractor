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

flow_dir = "flow_direction.tif"
unnested_basins = "unnested_basins.tif"
pour_pts = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\Watershed\pour_points.shp"

wbt.unnest_basins(
    flow_dir, 
    pour_pts, 
    unnested_basins, 

)