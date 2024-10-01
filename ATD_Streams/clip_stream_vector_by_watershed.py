import geopandas as gpd
import os
import fiona

def clip_gpkg_by_layers(clipping_gpkg_path, input_gpkg_path, output_folder):
    """
    Clips each layer in the input GeoPackage using the geometries from the clipping GeoPackage.
    Each clipped layer is saved as a separate GeoPackage in the specified output folder.

    Parameters:
    - clipping_gpkg_path (str): Path to the clipping GeoPackage.
    - input_gpkg_path (str): Path to the input GeoPackage to be clipped.
    - output_folder (str): Directory where the clipped GeoPackages will be saved.
    """
    
    # Load the clipping GeoPackage (assuming a single layer)
    input_gdf = gpd.read_file(input_gpkg_path)
    
    # List all layers in the input GeoPackage
    try:
        input_layers = fiona.listlayers(clipping_gpkg_path)
    except Exception as e:
        print(f"Error reading input GeoPackage layers: {e}")
        return
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through each layer in the input GeoPackage
    print(f"Layers in the input GeoPackage: {input_layers}")
    for layer in input_layers:
        print(f"Processing layer: {layer}")
        
        try:
            # Read the current layer from the input GeoPackage
            clipping_gdf = gpd.read_file(clipping_gpkg_path, layer=layer)
        except Exception as e:
            print(f"  Error reading layer '{layer}': {e}. Skipping this layer.")
            continue
        
        # Check and align CRS
        if clipping_gdf.crs != input_gdf.crs:
            print(f"  CRS mismatch for layer '{layer}'. Reprojecting input layer to match clipping CRS.")
            input_gdf = input_gdf.to_crs(clipping_gdf.crs)
        
        # Perform the clipping operation
        try:
            clipped_gdf = gpd.clip(input_gdf, clipping_gdf)
        except Exception as e:
            print(f"  Error clipping layer '{layer}': {e}. Skipping this layer.")
            continue
        
        # Check if the clipping resulted in any geometries
        if clipped_gdf.empty:
            print(f"  No features clipped for layer '{layer}'. Skipping saving.")
            continue
        
        # Sanitize the layer name to create a valid filename
        sanitized_layer = "".join([c if c.isalnum() or c in (' ', '_', '-') else "_" for c in layer])
        output_path = os.path.join(output_folder, f"{sanitized_layer}_centerline.gpkg")
        
        # Save the clipped GeoDataFrame to a new GeoPackage
        try:
            clipped_gdf.to_file(output_path, driver="GPKG", layer=layer)
            print(f"  Saved clipped GeoPackage for layer '{layer}': {output_path}")
        except Exception as e:
            print(f"  Error saving clipped layer '{layer}': {e}.")
            continue

if __name__ == "__main__":
    clipping_gpkg_path = r"Y:\ATD\GIS\ETF\Watershed_Boundaries.gpkg"
    input_gpkg_path = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\WBT_Outputs_Low\streams_100k.gpkg"
    output_folder = r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines"
    
    clip_gpkg_by_layers(clipping_gpkg_path, input_gpkg_path, output_folder)
