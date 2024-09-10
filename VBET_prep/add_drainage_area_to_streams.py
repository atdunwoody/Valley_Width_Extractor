import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import LineString
from rasterio.mask import mask

# Input file paths
input_gpkg = r"Y:\ATD\GIS\Bennett\Valley Widths\Channel Polygons\Centerlines_LSDTopo\Bennett_Centerlines_EPSG26913.gpkg"
output_gpkg = r"Y:\ATD\GIS\Bennett\Valley Widths\Channel Polygons\Centerlines_LSDTopo\Bennett_Centerlines_DA.shp"
raster_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Channel Polygons\WBT Channels\drainage_area.tif"

# Load the LineString layer
gdf = gpd.read_file(input_gpkg)

# Define buffer distance
buffer_distance = 5  # 5 meters

# Open the raster file
with rasterio.open(raster_path) as src:
    # Read CRS of the raster
    raster_crs = src.crs
    
    # Reproject the GeoDataFrame if needed
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    
    drainage_areas = []
    
    # Iterate through each LineString feature
    for idx, row in gdf.iterrows():
        line = row.geometry
        
        # Create a buffer around the line
        buffer_geom = line.buffer(buffer_distance)
        
        # Mask the raster using the buffered geometry
        out_image, out_transform = mask(src, [buffer_geom], crop=True)
        
        # Extract valid values (non-nodata) from the masked raster
        valid_data = out_image[out_image != src.nodata]
        
        if len(valid_data) > 0:
            # Calculate the maximum valid raster value
            max_value = np.max(valid_data)
        else:
            max_value = None
        
        # Append the calculated drainage area (maximum raster value)
        drainage_areas.append(max_value)
    
    # Add the new 'drainage_area' column to the GeoDataFrame
    gdf['DA'] = drainage_areas

# Save the updated GeoDataFrame to a new GeoPackage
gdf.to_file(output_gpkg, driver="ESRI Shapefile")

print(f"Drainage areas calculated and saved to {output_gpkg}.")