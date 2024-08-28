import geopandas as gpd
import rasterio
from rasterio.mask import mask
import os

def clip_raster_by_name(clipping_gpkg_path, input_raster_path, output_folder):
    # Load the clipping GeoPackage
    clipping_gdf = gpd.read_file(clipping_gpkg_path)
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each unique "Name" in the clipping GeoPackage
    for name in clipping_gdf["Name"].unique():
        # Filter the clipping GeoDataFrame by the current "Name"
        clip_gdf = clipping_gdf[clipping_gdf["Name"] == name]
        
        # Convert the filtered GeoDataFrame to a list of geometries
        clip_geometry = [geom for geom in clip_gdf.geometry]

        # Open the input raster
        with rasterio.open(input_raster_path) as src:
            # Clip the raster with the geometry
            out_image, out_transform = mask(src, clip_geometry, crop=True)
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Define the output path with "Name" as a prefix
            output_path = os.path.join(output_folder, f"{name}_clipped.tif")
            
            # Save the clipped raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        
        print(f"Saved clipped raster: {output_path}")

clipping_gpkg_path = r"Y:\ATD\GIS\Bennett\Bennett_watersheds.gpkg"
input_raster_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\dem_2021_bennett_clip_filtered.tif"
output_folder = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\Watershed_Clipped"


clip_raster_by_name(clipping_gpkg_path, input_raster_path, output_folder)
