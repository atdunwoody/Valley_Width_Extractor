
from whitebox_workflows import WbEnvironment
import whitebox
import os
import geopandas as gpd
import rasterio




# Initialize WhiteboxTools
wbt = whitebox.WhiteboxTools()

# Set the environment
wbe = WbEnvironment()


wbt.set_working_dir(r"Y:\ATD\GIS\ETF\Watershed Stats\Unnested Watersheds\LM2")


