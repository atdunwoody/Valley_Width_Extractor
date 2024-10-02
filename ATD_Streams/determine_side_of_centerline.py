import os
import logging
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import split
import rasterio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def sample_raster_along_line(line, raster, n_points=None, nodata_value=None):
    """
    Vectorized sampling of raster values along a LineString or MultiLineString.

    Parameters:
        line (LineString or MultiLineString): The line along which to sample.
        raster (rasterio.io.DatasetReader): The opened raster dataset.
        n_points (int, optional): Number of points to sample along the line.
        nodata_value (float, optional): The NoData value in the raster.

    Returns:
        distances_valid (list): Distances along the line where sampling was successful.
        values (list): Raster values at the sampled points.
        points_valid (list): Shapely Point objects of valid sampled points.
    """
    logging.info(f"Entering sample_raster_along_line with parameters: n_points={n_points}, nodata_value={nodata_value}")
    if n_points is None:
        n_points = int(line.length)
    distances = np.linspace(0, line.length, n_points)
    points = [line.interpolate(distance) for distance in distances]

    xs = np.array([point.x for point in points])
    ys = np.array([point.y for point in points])

    # Transform coordinates to raster row and column indices using affine transform
    transform = raster.transform
    try:
        inv_transform = ~transform
    except Exception as e:
        logging.error(f"Error inverting transform: {e}")
        return [], [], []

    cols, rows = inv_transform * (xs, ys)

    # Convert to integer indices
    cols = np.array(cols).astype(int)
    rows = np.array(rows).astype(int)

    # Get raster dimensions
    raster_height, raster_width = raster.read(1, masked=False).shape

    # Create mask for valid rows and cols
    valid_mask = (rows >= 0) & (rows < raster_height) & (cols >= 0) & (cols < raster_width)

    # Log warnings for invalid points
    invalid_indices = np.where(~valid_mask)[0]
    for i in invalid_indices:
        logging.warning(f"Point at distance {distances[i]:.2f}m falls outside raster bounds (row: {rows[i]}, col: {cols[i]}) and will be skipped.")

    # Apply valid mask
    rows_valid = rows[valid_mask]
    cols_valid = cols[valid_mask]
    distances_valid = distances[valid_mask]
    points_valid = [points[i] for i in range(len(points)) if valid_mask[i]]

    # Read raster values
    try:
        data = raster.read(1)
    except Exception as e:
        logging.error(f"Error reading raster data: {e}")
        return [], [], []

    values = data[rows_valid, cols_valid]

    # Mask out nodata values
    if nodata_value is not None:
        nodata_mask = values != nodata_value
        if not np.any(nodata_mask):
            logging.warning("All sampled raster values are NoData.")
            return [], [], []
        values = values[nodata_mask]
        distances_valid = distances_valid[nodata_mask]
        points_valid = [points_valid[i] for i in range(len(points_valid)) if nodata_mask[i]]

    logging.info(f"Sampled {len(values)} valid raster points out of {n_points if n_points else 'variable'} requested.")
    return distances_valid.tolist(), values.tolist(), points_valid

def determine_side_of_centerline(perpendiculars_path, centerlines_path, dem_path, output_dir):

    # Buffer distance in meters
    buffer_distance = 5.0

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read centerlines and perpendiculars
    logging.info("Reading centerlines...")
    centerlines_gdf = gpd.read_file(centerlines_path)
    logging.info(f"Centerlines loaded: {len(centerlines_gdf)} features.")

    logging.info("Reading perpendiculars...")
    perpendiculars_gdf = gpd.read_file(perpendiculars_path)
    logging.info(f"Perpendiculars loaded: {len(perpendiculars_gdf)} features.")

    # Add 'perp_idx' field to perpendiculars_gdf
    if 'perp_idx' not in perpendiculars_gdf.columns:
        perpendiculars_gdf = perpendiculars_gdf.reset_index().rename(columns={'index': 'perp_idx'})
    else:
        logging.info("'perp_idx' field already exists in perpendiculars_gdf.")

    # Save the updated perpendiculars_gdf with 'perp_idx' if needed
    perpendiculars_with_idx_path = os.path.join(output_dir, "perpendiculars_with_idx.gpkg")
    perpendiculars_gdf.to_file(perpendiculars_with_idx_path, driver="GPKG")
    logging.info(f"Perpendiculars with 'perp_idx' saved to {perpendiculars_with_idx_path}.")

    # Open DEM raster
    logging.info("Opening DEM raster...")
    try:
        dem_raster = rasterio.open(dem_path)
    except Exception as e:
        logging.error(f"Error opening DEM raster: {e}")
        return

    # Initialize lists to collect points
    side1_records = []
    side2_records = []

    # Iterate over each perpendicular line
    logging.info("Processing perpendicular lines...")
    for idx, perp_row in perpendiculars_gdf.iterrows():
        perp_geom = perp_row.geometry
        perp_idx = perp_row['perp_idx']

        if not isinstance(perp_geom, (LineString, MultiLineString)):
            logging.warning(f"Perpendicular at perp_idx {perp_idx} is not a LineString or MultiLineString. Skipping.")
            continue

        # Sample points along the perpendicular
        distances, elevations, sampled_points = sample_raster_along_line(perp_geom, dem_raster, nodata_value=dem_raster.nodata)

        if not sampled_points:
            logging.info(f"No valid sampled points for perpendicular at perp_idx {perp_idx}. Skipping.")
            continue

        # Find the corresponding centerline (assuming single centerline)
        # If multiple centerlines, additional logic is needed to match the correct one
        centerline = centerlines_gdf.unary_union  # Merge all centerlines into a single geometry

        # Split the perpendicular by the centerline
        try:
            split_result = split(perp_geom, centerline)
        except Exception as e:
            logging.warning(f"Error splitting perpendicular at perp_idx {perp_idx} by centerline: {e}. Skipping.")
            continue

        # Filter for LineString geometries
        split_geoms = [geom for geom in split_result.geoms if isinstance(geom, (LineString, MultiLineString))]

        if len(split_geoms) < 2:
            logging.warning(f"Perpendicular at perp_idx {perp_idx} was not split into two LineStrings. Skipping.")
            continue

        # Assume the first two LineStrings are the desired segments
        seg1, seg2 = split_geoms[:2]

        # Buffer each segment
        buffer1 = seg1.buffer(buffer_distance)
        buffer2 = seg2.buffer(buffer_distance)

        # Assign each point to a side based on which buffer contains it
        for point, elevation in zip(sampled_points, elevations):
            record = {'geometry': point, 'elevation': elevation, 'perp_idx': perp_idx}
            if buffer1.contains(point):
                side1_records.append(record)
            elif buffer2.contains(point):
                side2_records.append(record)
            else:
                logging.debug(f"Point {point} does not fall within any buffer. Skipping.")

    # Close the raster
    dem_raster.close()

    # Create GeoDataFrames for each side
    logging.info("Creating GeoDataFrame for Side 1...")
    if side1_records:
        side1_gdf = gpd.GeoDataFrame(side1_records, crs=perpendiculars_gdf.crs)
        side1_output_path = os.path.join(output_dir, "side1_points.gpkg")
        side1_gdf.to_file(side1_output_path, driver="GPKG")
        logging.info(f"Side 1 points saved to {side1_output_path} with {len(side1_gdf)} points.")
    else:
        logging.info("No points assigned to Side 1.")
        side1_output_path = None

    logging.info("Creating GeoDataFrame for Side 2...")
    if side2_records:
        side2_gdf = gpd.GeoDataFrame(side2_records, crs=perpendiculars_gdf.crs)
        side2_output_path = os.path.join(output_dir, "side2_points.gpkg")
        side2_gdf.to_file(side2_output_path, driver="GPKG")
        logging.info(f"Side 2 points saved to {side2_output_path} with {len(side2_gdf)} points.")
    else:
        logging.info("No points assigned to Side 2.")
        side2_output_path = None

    return side1_output_path, side2_output_path, perpendiculars_with_idx_path

if __name__ == "__main__":
    perpendiculars_path = r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\LPM_smooth_perpendiculars_20m.gpkg"
    centerlines_path = r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\LPM_centerline.gpkg"
    dem_path = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\WBT_Outputs_Low\filled_dem.tif"
    output_dir = r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\Wavelets"
    determine_side_of_centerline(perpendiculars_path, centerlines_path, dem_path, output_dir)
