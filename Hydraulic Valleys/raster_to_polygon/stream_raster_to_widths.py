# raster_processor.py

import os
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Polygon, LineString, Point, MultiLineString, MultiPolygon
from shapely.ops import unary_union
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_valid_geometry(raster_path):
    """
    Create a GeoDataFrame of polygons representing valid (non-NoData) raster areas.
    
    Returns:
        GeoDataFrame with polygons of valid data areas.
    """
    with rasterio.open(raster_path) as src:
        band1 = src.read(1)
        nodata = src.nodata
        if nodata is None:
            # If NoData is not defined, assume all data is valid
            valid_mask = np.ones(band1.shape, dtype=bool)
        else:
            valid_mask = band1 != nodata

        # Convert the valid mask to polygons
        shapes_gen = shapes(band1, mask=valid_mask, transform=src.transform)
        polygons = []
        for geom, value in shapes_gen:
            if value:  # Only keep valid areas
                poly = shape(geom)  # Convert dict to Shapely geometry
                if poly.is_valid:
                    polygons.append(poly)
                else:
                    poly = poly.buffer(0)
                    if poly.is_valid:
                        polygons.append(poly)

        if not polygons:
            logging.warning(f"No valid polygons found in raster: {raster_path}")
            return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=src.crs)

        valid_polygons = unary_union(polygons)
        if isinstance(valid_polygons, Polygon):
            valid_polygons = [valid_polygons]
        elif isinstance(valid_polygons, MultiPolygon):
            valid_polygons = list(valid_polygons)
        else:
            logging.error(f"Unexpected geometry type: {type(valid_polygons)}")
            return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=src.crs)

        valid_gdf = gpd.GeoDataFrame(geometry=valid_polygons, crs=src.crs)
        return valid_gdf

def clip_lines_to_valid_areas(lines_gdf, valid_gdf):
    """Clip lines to the valid data polygons."""
    clipped_lines = gpd.overlay(lines_gdf, valid_gdf, how='intersection')
    return clipped_lines

def save_gpkg(gdf, output_path, layer_name='clipped_lines'):
    """Save GeoDataFrame to a GeoPackage."""
    gdf.to_file(output_path, layer=layer_name, driver="GPKG")
    logging.info(f"Saved clipped lines to {output_path}")

def determine_right_end(line):
    """
    Determine the 'right' end of a line.
    Assuming 'right' is defined as the end with the higher X coordinate.
    If X coordinates are equal, use the higher Y coordinate.
    """
    if not isinstance(line, LineString):
        raise ValueError("Geometry is not a LineString")
    coords = list(line.coords)
    start, end = Point(coords[0]), Point(coords[-1])
    if end.x > start.x:
        return end
    elif end.x < start.x:
        return start
    else:
        # X coordinates are equal, compare Y
        return end if end.y > start.y else start

def create_connected_polygon(lines_gdf):
    """
    Create a non-self-intersecting polygon by connecting the right ends of lines.

    Returns:
        A GeoDataFrame containing the polygon.
    """
    # Sort lines if necessary (assuming the list is ordered)
    lines_sorted = lines_gdf.sort_index()

    # Extract right ends
    right_ends = [determine_right_end(line) for line in lines_sorted.geometry]

    # Ensure the polygon is closed by adding the first point at the end
    polygon_coords = [(pt.x, pt.y) for pt in right_ends]
    polygon_coords.append(polygon_coords[0])

    # Create the polygon
    polygon = Polygon(polygon_coords)

    if not polygon.is_valid:
        # Attempt to fix the polygon if it's invalid
        polygon = polygon.buffer(0)
        if not polygon.is_valid:
            raise ValueError("Failed to create a valid polygon.")

    polygon_gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs=lines_gdf.crs)
    return polygon_gdf

def process_rasters(raster_list, lines_gpkg_path, output_dir, polygon_output_gpkg):
    """
    Process a list of rasters to extract widths and save outputs to GPKG files.

    Args:
        raster_list (list): List of raster file paths.
        lines_gpkg_path (str): Path to the perpendicular lines GeoPackage.
        output_dir (str): Directory to save the output GPKG files.
        polygon_output_gpkg (str): Path to save the connected polygon GPKG.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load perpendicular lines
    logging.info(f"Loading perpendicular lines from {lines_gpkg_path}")
    lines_gdf = gpd.read_file(lines_gpkg_path)
    if lines_gdf.empty:
        logging.error("Perpendicular lines GeoPackage is empty.")
        return

    # Ensure lines are LineStrings
    lines_gdf = lines_gdf[lines_gdf.geometry.type == 'LineString']
    if lines_gdf.empty:
        logging.error("No LineString geometries found in perpendicular lines GeoPackage.")
        return

    # Convert all LineStrings to MultiLineStrings to ensure consistency
    lines_gdf['geometry'] = lines_gdf['geometry'].apply(
        lambda geom: MultiLineString([geom]) if isinstance(geom, LineString) else geom
    )

    # Process each raster
    for raster_path in raster_list:
        raster_name = os.path.splitext(os.path.basename(raster_path))[0]
        logging.info(f"Processing raster: {raster_path}")

        # Get valid geometry
        valid_gdf = get_valid_geometry(raster_path)
        if valid_gdf.empty:
            logging.warning(f"No valid areas to clip for raster: {raster_name}")
            continue

        logging.debug(f"Number of valid polygons for {raster_name}: {len(valid_gdf)}")

        # Clip lines to valid areas
        clipped_lines = clip_lines_to_valid_areas(lines_gdf, valid_gdf)
        if clipped_lines.empty:
            logging.warning(f"No lines clipped for raster: {raster_name}")
            continue

        # Ensure all geometries are MultiLineStrings
        clipped_lines['geometry'] = clipped_lines['geometry'].apply(
            lambda geom: MultiLineString([geom]) if isinstance(geom, LineString) else geom
        )

        # Calculate widths (lengths of the clipped lines)
        clipped_lines['width'] = clipped_lines.length
        logging.info(f"Calculated widths for clipped lines in raster: {raster_name}")

        # Define output GeoPackage path
        output_gpkg = os.path.join(output_dir, f"{raster_name}.gpkg")

        # Save clipped lines with widths
        save_gpkg(clipped_lines, output_gpkg)

    # Create connected polygon
    logging.info("Creating connected polygon from perpendicular lines.")
    try:
        polygon_gdf = create_connected_polygon(lines_gdf)
    except ValueError as e:
        logging.error(f"Error creating polygon: {e}")
        return

    # Save the polygon to GeoPackage
    polygon_gdf.to_file(polygon_output_gpkg, driver="GPKG")
    logging.info(f"Saved connected polygon to {polygon_output_gpkg}")
