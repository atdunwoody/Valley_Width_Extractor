import geopandas as gpd
from shapely.geometry import Polygon
import os 
import warnings

# Suppress FutureWarning related to pandas.Int64Index deprecation
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="pandas.Int64Index is deprecated"
)


# Define file paths
lines_gpkg_list= [
    r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\LM2_smooth_perpendiculars_1m.gpkg",
r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\LPM_smooth_perpendiculars_1m.gpkg",
r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\MM_smooth_perpendiculars_1m.gpkg",
r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\MPM_smooth_perpendiculars_1m.gpkg",
r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\UM1_smooth_perpendiculars_1m.gpkg",
r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\UM2_smooth_perpendiculars_1m.gpkg",
]
polygon_gpkg_list = [
r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\LM2_valleys_wavelets.gpkg",
r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\LPM_valleys_wavelets.gpkg",
r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\MM_valleys_wavelets.gpkg",
r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\MPM_valleys_wavelets.gpkg",
r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\UM1_valleys_wavelets.gpkg",
r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\UM2_valleys_wavelets.gpkg",
]
output_gpkg_dir = r"Y:\ATD\GIS\ETF\Valley Bottoms"  # Path for the output GeoPackage

for lines_gpkg_path, polygon_gpkg_path in zip(lines_gpkg_list, polygon_gpkg_list):
    # Load the lines GeoPackage and the polygon GeoPackage
    lines_gdf = gpd.read_file(lines_gpkg_path)
    polygon_gdf = gpd.read_file(polygon_gpkg_path)

    # Check and align CRS
    print("Checking CRS alignment...")
    if lines_gdf.crs != polygon_gdf.crs:
        print(f"CRS mismatch detected. Reprojecting polygon from {polygon_gdf.crs} to {lines_gdf.crs}.")
        polygon_gdf = polygon_gdf.to_crs(lines_gdf.crs)
    else:
        print("CRS are aligned.")

    # If multiple polygons, you might want to merge them into a single geometry
    print("Merging polygons (if multiple)...")
    merged_polygon = polygon_gdf.unary_union  # This creates a single (multi)polygon

    # Perform the clipping operation
    clipped_lines_gdf = gpd.clip(lines_gdf, merged_polygon)

    # Remove any empty geometries resulting from the clip
    print("Removing empty geometries...")
    clipped_lines_gdf = clipped_lines_gdf[~clipped_lines_gdf.is_empty]

    # Save the clipped lines to a new GeoPackage
    output_name = f"{polygon_gpkg_path.split('.gpkg')[0]}_widths.gpkg"
    output_gpkg_path = os.path.join(output_gpkg_dir, output_name)
    print(f"Saving clipped lines to {output_gpkg_path}...")
    clipped_lines_gdf.to_file(output_gpkg_path, driver='GPKG')