import geopandas as gpd
import os

def clip_gpkg_by_name(clipping_gpkg_path, input_gpkg_path, output_folder):
    # Load the clipping GeoPackage and input GeoPackage
    clipping_gdf = gpd.read_file(clipping_gpkg_path)
    input_gdf = gpd.read_file(input_gpkg_path)
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each unique "Name" in the clipping GeoPackage
    for name in clipping_gdf["Name"].unique():
        # Filter the clipping GeoDataFrame by the current "Name"
        clip_gdf = clipping_gdf[clipping_gdf["Name"] == name]
        
        # Clip the input GeoDataFrame using the filtered clipping GeoDataFrame
        clipped_gdf = gpd.clip(input_gdf, clip_gdf)
        
        # Define the output path with "Name" as a prefix
        output_path = os.path.join(output_folder, f"{name}_clipped.gpkg")
        
        # Save the clipped GeoDataFrame to the output GeoPackage
        clipped_gdf.to_file(output_path, driver="GPKG")
        
        print(f"Saved clipped GeoPackage: {output_path}")

# Example usage
clipping_gpkg_path = r"Y:\ATD\GIS\Bennett\Bennett_watersheds.gpkg"
input_gpkg_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\streams.gpkg"
output_folder = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines"

clip_gpkg_by_name(clipping_gpkg_path, input_gpkg_path, output_folder)
