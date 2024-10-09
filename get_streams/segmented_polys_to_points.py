import geopandas as gpd
import os

def compute_centroids(input_gpkg, input_layer, output_gpkg, output_layer):
    """
    Computes the centroids of multipolygons in a GeoPackage and saves them to a new GeoPackage.

    Parameters:
    - input_gpkg (str): Path to the input GeoPackage file.
    - input_layer (str): Name of the layer in the input GeoPackage.
    - output_gpkg (str): Path to the output GeoPackage file.
    - output_layer (str): Name of the layer in the output GeoPackage.
    """

    # Check if input file exists
    if not os.path.exists(input_gpkg):
        raise FileNotFoundError(f"Input GeoPackage '{input_gpkg}' does not exist.")

    # Read the input GeoPackage layer
    print(f"Reading layer '{input_layer}' from '{input_gpkg}'...")
    gdf = gpd.read_file(input_gpkg, layer=input_layer)

    # Ensure geometries are multipolygons
    multipolygons = gdf[gdf.geometry.type.isin(['MultiPolygon', 'Polygon'])].copy()
    if multipolygons.empty:
        raise ValueError("No MultiPolygon or Polygon geometries found in the input layer.")

    # Optionally, handle only multipolygons
    # multipolygons = gdf[gdf.geometry.type == 'MultiPolygon'].copy()

    # Compute centroids
    print("Computing centroids...")
    multipolygons['centroid'] = multipolygons.centroid

    # Create a new GeoDataFrame for centroids
    centroids_gdf = gpd.GeoDataFrame(
        multipolygons.drop(columns='geometry'),  # Drop original geometry if not needed
        geometry=multipolygons['centroid'],
        crs=multipolygons.crs
    )

    # Drop the temporary 'centroid' column
    centroids_gdf = centroids_gdf.drop(columns='centroid')

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_gpkg)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the centroids to the new GeoPackage
    print(f"Saving centroids to layer '{output_layer}' in '{output_gpkg}'...")
    centroids_gdf.to_file(output_gpkg, layer=output_layer, driver="GPKG")

    print("Centroids successfully saved.")

if __name__ == "__main__":

    # Define input and output parameters
    input_geopackage = "path/to/input.gpkg"         # Replace with your input GeoPackage path
    input_layer_name = "multipolygons"              # Replace with your input layer name
    output_geopackage = "path/to/output_centroids.gpkg"  # Replace with desired output GeoPackage path
    output_layer_name = "centroids"                  # Desired output layer name

    # Compute and save centroids
    compute_centroids(input_geopackage, input_layer_name, output_geopackage, output_layer_name)
