import rasterio
from rasterio.features import shapes
import numpy as np
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
import geopandas as gpd
import os
import warnings

def extract_outline(input_raster_path, output_vector_path):
    """
    Extracts the exterior outline between valid and NoData areas in a raster and saves it as a vector file.

    Parameters:
    - input_raster_path: Path to the input raster file.
    - output_vector_path: Path to the output vector file (.shp, .geojson, or .gpkg).
    """
    # Open the raster file
    with rasterio.open(input_raster_path) as src:
        # Read the first band
        band1 = src.read(1)
        # Get the NoData value
        nodata = src.nodata

        if nodata is None:
            raise ValueError("The raster does not have a NoData value defined.")

        # Create a mask: 1 for valid data, 0 for NoData
        mask = np.where(band1 != nodata, 1, 0).astype(np.uint8)

        # Generate shapes (polygons) from the mask
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(mask, mask=mask, transform=src.transform))
        )

        # Convert shapes to geometries
        geoms = [shape(feature['geometry']) for feature in results if feature['properties']['raster_val'] == 1]

        if not geoms:
            print(f"No valid data found in the raster: {input_raster_path}")
            return

        # Merge all geometries into a single geometry
        merged = unary_union(geoms)

        # Extract the exterior boundaries as Polygons
        outlines = []
        if merged.type == 'Polygon':
            outlines.append(Polygon(merged.exterior))
        elif merged.type == 'MultiPolygon':
            for poly in merged.geoms:
                outlines.append(Polygon(poly.exterior))
        else:
            raise ValueError(f"Unexpected geometry type after merging: {merged.type}")

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': outlines}, crs=src.crs)

        # Save to the desired format based on file extension
        output_ext = os.path.splitext(output_vector_path)[1].lower()
        if output_ext == '.shp':
            gdf.to_file(output_vector_path, driver='ESRI Shapefile')
        elif output_ext in ['.geojson', '.json']:
            gdf.to_file(output_vector_path, driver='GeoJSON')
        elif output_ext == '.gpkg':
            gdf.to_file(output_vector_path, driver='GPKG')
        else:
            raise ValueError("Unsupported output format. Use .shp, .gpkg, or .geojson/.json.")

        print(f"Outline successfully saved to {output_vector_path}")

def main():
    """
    Processes all raster files in the input directory and saves their outlines as vector files.
    """
    watersheds = ["MM", "MW", "UM", "UW",  "UE"]
    for watershed in watersheds:
        input_raster_dir= os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters CRS", watershed)  # Replace with your input raster path
        output_vector_dir = os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Polygons", watershed)

        # Ensure the output directory exists
        os.makedirs(output_vector_dir, exist_ok=True)

        # List all raster files in the input directory
        input_raster_list = [
            os.path.join(input_raster_dir, f)
            for f in os.listdir(input_raster_dir)
            if f.lower().endswith('.tif')
        ]

        # Process each raster file
        for input_raster in input_raster_list:
            print(f"Processing {input_raster}")
            # Generate output file path with .gpkg extension
            base_name = os.path.splitext(os.path.basename(input_raster))[0]
            output_vector = os.path.join(output_vector_dir, f"{base_name}.gpkg")
            try:
                extract_outline(input_raster, output_vector)
            except Exception as e:
                print(f"Error processing {input_raster}: {e}")

if __name__ == "__main__":
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pandas.Int64Index.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*ShapelyDeprecationWarning.*")
    main()
