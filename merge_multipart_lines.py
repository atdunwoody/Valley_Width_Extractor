import geopandas as gpd
from shapely.ops import linemerge, unary_union

def merge_multipart_lines(gpkg_input_path, gpkg_output_path):
    # Load the GeoPackage layer
    gdf = gpd.read_file(gpkg_input_path)

    # Ensure that the geometry type is LineString or MultiLineString
    if gdf.geom_type.isin(['LineString', 'MultiLineString']).all():
        # Merge all lines into a single geometry
        merged_geometry = linemerge(unary_union(gdf['geometry']))
        
        # Create a new GeoDataFrame with the merged geometry
        merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=gdf.crs)
        
        # Save the output to a new GeoPackage
        merged_gdf.to_file(gpkg_output_path, driver="GPKG")
        print(f"Merged lines saved to {gpkg_output_path}.")
    else:
        print("The layer does not contain LineString or MultiLineString geometries.")

# Example usage
gpkg_input_path = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\ME_Centerlines_EPSG26913.gpkg"
gpkg_output_path = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\ME_Centerlines_EPSG26913_single.gpkg"

merge_multipart_lines(gpkg_input_path, gpkg_output_path)
