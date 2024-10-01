import geopandas as gpd
import os
import fiona
from tqdm import tqdm  # Import tqdm for progress bar
import pandas as pd

def create_filtered_polygons_layers(points_gpkg_path, polygons_gpkg_path, output_gpkg_path):
    """
    Creates an output GeoPackage with a new polygon layer for each input point geometry.
    Each layer contains a single combined polygon where Watershed_ID <= point.ID.

    :param points_gpkg_path: Path to the input points GeoPackage.
    :param polygons_gpkg_path: Path to the input polygons GeoPackage.
    :param output_gpkg_path: Path to the output GeoPackage.
    """
    
    # Load the points GeoPackage
    print(f"Loading points from {points_gpkg_path}...")
    points_layers = fiona.listlayers(points_gpkg_path)
    if not points_layers:
        raise ValueError("No layers found in the points GeoPackage.")
    points_gdf = gpd.read_file(points_gpkg_path, layer=points_layers[0])
    
    # Check for 'ID' field
    if 'ID' not in points_gdf.columns:
        raise ValueError("The points GeoPackage does not contain an 'ID' field.")
    
    # Load the polygons GeoPackage
    print(f"Loading polygons from {polygons_gpkg_path}...")
    polygons_layers = fiona.listlayers(polygons_gpkg_path)
    if not polygons_layers:
        raise ValueError("No layers found in the polygons GeoPackage.")
    polygons_gdf = gpd.read_file(polygons_gpkg_path, layer=polygons_layers[0])
    
    # Check for 'Watershed_ID' field
    if 'Watershed_ID' not in polygons_gdf.columns:
        raise ValueError("The polygons GeoPackage does not contain a 'Watershed_ID' field.")
    
    # Ensure both GeoDataFrames use the same CRS
    if points_gdf.crs != polygons_gdf.crs:
        print("CRS mismatch between points and polygons. Reprojecting points to match polygons CRS.")
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)
    
    # Prepare the output GeoPackage
    if os.path.exists(output_gpkg_path):
        print(f"Output GeoPackage {output_gpkg_path} already exists and will be overwritten.")
        os.remove(output_gpkg_path)  # Remove existing file to start fresh
    
    # Iterate over each point with a progress bar
    for idx, point in tqdm(points_gdf.iterrows(), total=points_gdf.shape[0], desc="Processing Points"):
        point_id = point['ID']
        layer_name = f"polygon_ID_{point_id}"
        
        # Sanitize layer name to comply with GeoPackage layer naming rules
        layer_name = layer_name[:63]  # GeoPackage layer names are limited to 63 characters
        
        # Debugging information
        # print(f"Processing Point ID: {point_id} -> Creating layer '{layer_name}'")
        
        # Filter polygons where Watershed_ID <= point.ID
        filtered_polygons = polygons_gdf[polygons_gdf['Watershed_ID'] <= point_id]
        
        if filtered_polygons.empty:
            # print(f"  No polygons found for Point ID {point_id} with Watershed_ID <= {point_id}. Skipping layer creation.")
            continue  # Skip creating empty layers
        
        # Combine the filtered polygons into a single geometry
        combined_geometry = filtered_polygons.unary_union
        
        # Create a new GeoDataFrame with the combined geometry
        combined_gdf = gpd.GeoDataFrame(
            {
                'ID': [point_id],  # You can add more attributes if needed
            },
            geometry=[combined_geometry],
            crs=polygons_gdf.crs
        )
        
        # Ensure the GeoDataFrame uses a standard RangeIndex to avoid FutureWarning
        combined_gdf.reset_index(drop=True, inplace=True)
        
        # Write the combined geometry to the output GeoPackage as a new layer
        combined_gdf.to_file(
            output_gpkg_path,
            layer=layer_name,
            driver="GPKG"
        )
        # print(f"  Layer '{layer_name}' created with a combined polygon.")
    
    print(f"All layers have been processed and saved to {output_gpkg_path}.")

if __name__ == "__main__":
    import argparse

    # Uncomment the argparse section if you want to run the script via command line with arguments.
    """
    parser = argparse.ArgumentParser(description="Create filtered polygon layers based on point IDs.")
    parser.add_argument("points_gpkg", help="Path to the input points GeoPackage.")
    parser.add_argument("polygons_gpkg", help="Path to the input polygons GeoPackage.")
    parser.add_argument("output_gpkg", help="Path to the output GeoPackage.")
    args = parser.parse_args()
    
    create_filtered_polygons_layers(args.points_gpkg, args.polygons_gpkg, args.output_gpkg)
    """
    
    # For now, using hardcoded paths as per original script
    points = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\Watershed\pour_points.shp"
    polygons = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\Watershed\watershed_polygons.gpkg"
    output = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\Watershed\watershed_polygons_split.gpkg"
    create_filtered_polygons_layers(points, polygons, output)
