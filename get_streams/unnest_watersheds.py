import geopandas as gpd
import os
import fiona
from tqdm import tqdm  # Import tqdm for progress bar
import pandas as pd
import warnings

# Suppress specific FutureWarnings from GeoPandas (use with caution)
warnings.filterwarnings("ignore", category=FutureWarning, module="geopandas")

def unnest_watersheds(pour_points_gpkg_path, wbt_watersheds_gpkg_path, output_gpkg_path, watershed_join_field='FID'):
    """
    Takes in the 
    Each layer contains a single combined polygon where Watershed_ID <= point.ID.

    :param pour_points_gpkg_path: Path to the Jensen pour points GeoPackage used to create watersheds GeoPackage.
    :param wbt_watersheds_gpkg_path: Path to the watersheds GeoPackage created by get_watersheds.py.
    :param output_gpkg_path: Path to the output GeoPackage.
    """
    
    # Load the points GeoPackage
    print(f"Loading points from {pour_points_gpkg_path}...")
    points_layers = fiona.listlayers(pour_points_gpkg_path)
    if not points_layers:
        raise ValueError("No layers found in the points GeoPackage.")
    points_gdf = gpd.read_file(pour_points_gpkg_path, layer=points_layers[0])
    
    # Load the polygons GeoPackage
    print(f"Loading polygons from {wbt_watersheds_gpkg_path}...")
    polygons_layers = fiona.listlayers(wbt_watersheds_gpkg_path)
    if not polygons_layers:
        raise ValueError("No layers found in the polygons GeoPackage.")
    polygons_gdf = gpd.read_file(wbt_watersheds_gpkg_path, layer=polygons_layers[0])
    
    # Check for join field in polygons and all uppercase version
    if watershed_join_field not in polygons_gdf.columns:
        print(f"Columns in wbt_watersheds_gpkg_path:\n {polygons_gdf.columns}")
        raise ValueError(f"The polygons GeoPackage does not contain the specified join_field. \n Specify the correct field name in the 'watershed_join_field' parameter.")
    
    # Ensure both GeoDataFrames use the same CRS
    if points_gdf.crs != polygons_gdf.crs:
        print("CRS mismatch between points and polygons. Reprojecting points to match polygons CRS.")
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)
    
    # Prepare the output GeoPackage
    if os.path.exists(output_gpkg_path):
        print(f"Output GeoPackage {output_gpkg_path} already exists and will be overwritten.")
        os.remove(output_gpkg_path)  # Remove existing file to start fresh
    
    # Perform spatial join to assign watershed_join_field to each point
    print("Performing spatial join to assign 'ID' to each point based on watershed polygons...")
    points_gdf = gpd.sjoin(
        points_gdf, 
        polygons_gdf[[watershed_join_field, 'geometry']], 
        how='left', 
        predicate='within'
    )

    
    # Optionally, handle points that did not match any polygon
    missing_ids = points_gdf[watershed_join_field].isna().sum()
    if missing_ids > 0:
        print(f"Warning: {missing_ids} points did not match any watershed polygon and will have watershed_join_field set to NaN.")
    
    # Iterate over each point with a progress bar
    for idx, point in tqdm(points_gdf.iterrows(), total=points_gdf.shape[0], desc="Processing Points"):
        # Get point_id by overlaying it on the polygons with the watershed_join_field
        point_id = point[watershed_join_field]

        if pd.isna(point_id):
            print(f"  Point at index {idx} does not have a valid watershed_join_field. Skipping layer creation.")
            continue  # Skip points without a valid ID

        layer_name = f"polygon_ID_{int(point_id)}"
        
        # Sanitize layer name to comply with GeoPackage layer naming rules
        layer_name = layer_name[:63]  # GeoPackage layer names are limited to 63 characters
        
        # Debugging information
        # print(f"Processing Point ID: {point_id} -> Creating layer '{layer_name}'")
        
        # Filter polygons where Watershed_ID <= point.ID
        try:
            filtered_polygons = polygons_gdf[polygons_gdf[watershed_join_field] >= point_id].copy()
        except KeyError:
            raise ValueError(f"Field {watershed_join_field} not found in polygons GeoDataFrame.")
        if filtered_polygons.empty:
            print(f"  No polygons found for Point ID {point_id} with {watershed_join_field} <= {point_id}. Skipping layer creation.")
            continue  # Skip creating empty layers
        
        # Fix invalid geometries using .loc to avoid SettingWithCopyWarning
        filtered_polygons.loc[:, 'geometry'] = filtered_polygons.geometry.buffer(0)
        # Combine the filtered polygons into a single geometry
        combined_geometry = filtered_polygons.unary_union
        
        # Create a new GeoDataFrame with the combined geometry
        combined_gdf = gpd.GeoDataFrame(
            {
                'Watershed ID': [point_id],  
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
    
    unnest_watersheds(args.points_gpkg, args.polygons_gpkg, args.output_gpkg)
    """
    
    # For now, using hardcoded paths as per original script
    points = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\Watershed\pour_points.shp"
    polygons = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\Watershed\watershed_polygons.gpkg"
    output = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\Watershed\watershed_polygons_split.gpkg"
    unnest_watersheds(points, polygons, output)
