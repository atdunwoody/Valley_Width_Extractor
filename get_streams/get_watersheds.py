# dem_processor.py

import os
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from unnest_watersheds import unnest_watersheds
from shapely.ops import unary_union
from whitebox_workflows import WbEnvironment, download_sample_data
import whitebox

def find_intersections(centerline_file, perpendiculars_file, output_file):
    """
    Finds intersection points between centerline and perpendiculars vector files.

    Parameters:
    - centerline_file (str): Path to the centerline vector file (.shp or .gpkg).
    - perpendiculars_file (str): Path to the perpendiculars vector file (.shp or .gpkg).
    - output_file (str): Path for the output points file (.shp or .gpkg).

    Returns:
    - GeoDataFrame containing the intersection points.
    """
    # Validate input file formats
    valid_extensions = ['.shp', '.gpkg']
    center_ext = os.path.splitext(centerline_file)[1].lower()
    perp_ext = os.path.splitext(perpendiculars_file)[1].lower()
    output_ext = os.path.splitext(output_file)[1].lower()

    if center_ext not in valid_extensions:
        raise ValueError(f"Centerline file must be one of {valid_extensions}, got {center_ext}")
    if perp_ext not in valid_extensions:
        raise ValueError(f"Perpendiculars file must be one of {valid_extensions}, got {perp_ext}")
    if output_ext not in valid_extensions:
        raise ValueError(f"Output file must be one of {valid_extensions}, got {output_ext}")

    # Read the centerline and perpendiculars
    center_gdf = gpd.read_file(centerline_file)
    perp_gdf = gpd.read_file(perpendiculars_file)

    # Ensure both GeoDataFrames have the same CRS
    if center_gdf.crs != perp_gdf.crs:
        print("CRS mismatch detected. Reprojecting perpendiculars to match centerline CRS.")
        perp_gdf = perp_gdf.to_crs(center_gdf.crs)

    # Perform spatial join using intersection
    # This can be resource-intensive for large datasets
    intersections = []

    # To optimize, create a spatial index on perpendiculars
    perp_sindex = perp_gdf.sindex

    for idx, center_geom in center_gdf.geometry.iteritems():
        # Potential matches using spatial index
        possible_matches_index = list(perp_sindex.intersection(center_geom.bounds))
        possible_matches = perp_gdf.iloc[possible_matches_index]

        for _, perp_geom in possible_matches.geometry.iteritems():
            if center_geom.intersects(perp_geom):
                intersection = center_geom.intersection(perp_geom)
                if "Point" == intersection.geom_type:
                    intersections.append(intersection)
                elif "MultiPoint" == intersection.geom_type:
                    intersections.extend([pt for pt in intersection])
                # Handle other geometry types if necessary

    if not intersections:
        print("No intersections found.")
        return None

    # Create a GeoDataFrame from the intersection points
    intersection_gdf = gpd.GeoDataFrame(geometry=intersections, crs=center_gdf.crs)

    # Optionally, remove duplicate points
    intersection_gdf = intersection_gdf.drop_duplicates()

    # Save to the desired output format
    if output_ext == '.shp':
        intersection_gdf.to_file(output_file, driver='ESRI Shapefile')
    elif output_ext == '.gpkg':
        intersection_gdf.to_file(output_file, driver='GPKG')
    else:
        raise ValueError(f"Unsupported output file format: {output_ext}")

    print(f"Intersection points saved to {output_file}")
    return intersection_gdf

def polygonize_raster(raster_path, vector_path, attribute_name='DN'):
    """
    Polygonizes a raster file and saves it as a vector file.

    Parameters:
    - raster_path (str): Path to the input raster file.
    - vector_path (str): Path to the output vector file (.shp or .gpkg).
    - attribute_name (str): Name of the attribute to store raster values.

    Returns:
    - GeoDataFrame of the polygonized raster.
    """
    with rasterio.open(raster_path) as src:
        image = src.read(1)  # Read the first band
        mask = image != src.nodata  # Create a mask for valid data

        print("Starting polygonization of the raster...")
        results = (
            {'properties': {attribute_name: v}, 'geometry': shape(s)}
            for s, v in shapes(image, mask=mask, transform=src.transform)
        )
        geoms = list(results)
        print(f"Extracted {len(geoms)} polygons from the raster.")

    # Create a GeoDataFrame from the shapes
    gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

    # Save to the desired vector format
    vector_ext = os.path.splitext(vector_path)[1].lower()
    if vector_ext == '.shp':
        gdf.to_file(vector_path, driver='ESRI Shapefile')
    elif vector_ext == '.gpkg':
        gdf.to_file(vector_path, driver='GPKG')
    else:
        raise ValueError(f"Unsupported vector file format: {vector_ext}")

    print(f"Polygonized raster saved to {vector_path}")
    return gdf

def get_wbt_watersheds(d8_pntr, output_dir, pour_points=None, watershed_join_field=None,
                      stream_raster=None, stream_vector=None, perpendiculars=None):
    """
    Processes a list of DEM files to extract streams and convert them to GeoPackage format.

    Parameters:
    - d8_pntr (str): Path to the D8 pointer raster file.
    - output_dir (str): Path to the output directory.
    - pour_points (str, Optional): Path to the pour points shapefile.
    - stream_vector (str, Optional): Path to the stream vector file.
    - perpendiculars (str, Optional): Path to the perpendiculars vector file.

    Returns:
    - None
    """

    # Initialize WhiteboxTools
    wbt = whitebox.WhiteboxTools()

    # Set the environment
    wbe = WbEnvironment()

    # Create working directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if perpendiculars is not None or stream_vector is not None:
        pour_points = os.path.join(output_dir, "pour_points.shp")
        find_intersections(stream_vector, perpendiculars, pour_points)
    elif stream_raster is not None and pour_points is not None:
        pour_points_snapped = os.path.join(output_dir, "pour_points_snapped.shp")
        if not os.path.exists(pour_points_snapped):
            print("Snapping pour points to streams...")
            wbt.jenson_snap_pour_points(
                pour_pts=pour_points,
                streams=stream_raster,
                output=pour_points_snapped,
                snap_dist=50,
            )
        pour_points = pour_points_snapped
    else:
        raise ValueError("Must provide stream_raster for pour points.")

    watershed_raster = os.path.join(output_dir, "watersheds.tif")
    watershed_vector = os.path.join(output_dir, "watersheds.gpkg")
    unnested_watersheds = os.path.join(output_dir, "unnested_watersheds.gpkg")

    if not os.path.exists(watershed_raster):
        print("Generating watershed raster...")
        wbt.watershed(
            d8_pntr,
            pour_points,
            watershed_raster,
        )

    if not os.path.exists(watershed_vector):
        print("Polygonizing watershed raster...")
        polygonize_raster(watershed_raster, watershed_vector, attribute_name=watershed_join_field or 'DN')

    print("Unnesting watersheds...")
    # Update the watershed_vector path if necessary
    # Adjust the following line based on your actual workflow
    # Example assumes the polygonization was successful and the shapefile was created
    unnest_watersheds(pour_points, watershed_vector, unnested_watersheds, watershed_join_field=watershed_join_field)

def main():
    # Define input paths (update these paths as needed)
    d8_pointer = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\WBT_Outputs_Low\flow_direction.tif"
    output_dir = r"Y:\ATD\GIS\ETF\Watershed Stats\Unnested Watersheds\LM2"
    pour_points = r"Y:\ATD\GIS\ETF\Watershed Stats\Channel Stats\LM2_channel_stats_points.shp"
    stream_raster = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\WBT_Outputs_Low\streams_100k.tif"
    watershed_join_field = 'VALUE'  # Update if necessary

    get_wbt_watersheds(
        d8_pntr=d8_pointer,
        output_dir=output_dir,
        pour_points=pour_points,
        stream_raster=stream_raster,
        watershed_join_field=watershed_join_field
    )

if __name__ == "__main__":
    main()
