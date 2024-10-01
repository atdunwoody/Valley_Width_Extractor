import geopandas as gpd
import os

def clip_gpkg_by_name(clipping_gpkg_path, input_gpkg_path, output_folder):
    """
    Clips the input GeoPackage by each unique "Name" in the clipping GeoPackage.
    All geometries for each "Name" are collected and converted to single-part geometries before clipping.

    Parameters:
    - clipping_gpkg_path (str): Path to the clipping GeoPackage.
    - input_gpkg_path (str): Path to the input GeoPackage to be clipped.
    - output_folder (str): Directory where the clipped GeoPackages will be saved.
    """
    
    # Load the clipping GeoPackage and input GeoPackage
    clipping_gdf = gpd.read_file(clipping_gpkg_path)
    input_gdf = gpd.read_file(input_gpkg_path)
    
    # Ensure both GeoDataFrames use the same Coordinate Reference System (CRS)
    if clipping_gdf.crs != input_gdf.crs:
        print("CRS mismatch between clipping and input GeoPackages. Reprojecting input to match clipping CRS.")
        input_gdf = input_gdf.to_crs(clipping_gdf.crs)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through each unique "Name" in the clipping GeoPackage
    for name in clipping_gdf["Name"].unique():
        # Filter the clipping GeoDataFrame by the current "Name"
        clip_gdf = clipping_gdf[clipping_gdf["Name"] == name]
        
        # Collect all geometries for this "Name" and convert to single-part geometries
        # This handles any multipart geometries by exploding them into single parts
        # clip_gdf_single = clip_gdf.explode(index_parts=False).reset_index(drop=True)
        
        # # Optional: Dissolve all single-part geometries into a single geometry if desired
        # # This merges all parts into one geometry, which can be useful for certain clipping operations
        # # clip_gdf_single = clip_gdf_single.dissolve()
        
        # # Clip the input GeoDataFrame using the single-part clipping GeoDataFrame
        # input_gdf_single = input_gdf.explode(index_parts=False).reset_index(drop=True)
        # input_gdf_single = input_gdf_single.dissolve()  
        clipped_gdf = gpd.clip(input_gdf, clip_gdf)
        
        # Check if the clipping resulted in any geometries
        if clipped_gdf.empty:
            print(f"No features clipped for Name: {name}. Skipping saving.")
            continue
        
        # Define the output path with "Name" as a prefix
        # Replace or sanitize 'name' if it contains characters invalid for file names
        sanitized_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else "_" for c in name])
        output_path = os.path.join(output_folder, f"{sanitized_name}_clipped.gpkg")
        
        # Save the clipped GeoDataFrame to the output GeoPackage
        clipped_gdf.to_file(output_path, driver="GPKG")
        
        print(f"Saved clipped GeoPackage for '{name}': {output_path}")

if __name__ == "__main__":
    clipping_gpkg_path = r"Y:\ATD\GIS\Bennett\Bennett_watersheds.gpkg"
    input_gpkg_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Perpendiculars\Bennet_perps_5m.gpkg"
    output_folder = r"Y:\ATD\GIS\Bennett\Valley Widths\Perpendiculars\Clipped_by_Watershed"
    os.makedirs(output_folder, exist_ok=True)

    clip_gpkg_by_name(clipping_gpkg_path, input_gpkg_path, output_folder)
