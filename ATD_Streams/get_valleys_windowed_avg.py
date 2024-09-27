import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
import os
import logging
from datetime import datetime
import sys
import pandas as pd

# ----------------------------- Sampling Function ----------------------------- #

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
    logging.debug(f"Entering sample_raster_along_line with parameters: n_points={n_points}, nodata_value={nodata_value}")
    if n_points is None:
        n_points = int(line.length)
    distances = np.linspace(0, line.length, n_points)
    points = [line.interpolate(distance) for distance in distances]

    xs = np.array([point.x for point in points])
    ys = np.array([point.y for point in points])

    # Transform coordinates to raster row and column indices using affine transform
    transform = raster.transform
    cols, rows = ~transform * (xs, ys)  # Invert the transform to get col, row

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
    data = raster.read(1)
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

    logging.debug(f"Sampled {len(values)} valid raster points out of {n_points if n_points else 'variable'} requested.")
    return distances_valid.tolist(), values.tolist(), points_valid

# ------------------------- Hydraulic Computation Functions --------------------- #

def compute_cross_sectional_area_trapezoidal(x, y, depth):
    """
    Compute cross-sectional area using the trapezoidal rule.

    Parameters:
        x (array-like): Horizontal distances.
        y (array-like): Elevations.
        depth (float): Depth at which to compute the area.

    Returns:
        area (float): Cross-sectional area.
    """
    logging.debug(f"Computing cross-sectional area with depth={depth}")
    y_adjusted = np.clip(depth - y, 0, None)
    area = np.trapz(y_adjusted, x)
    return area

def compute_wetted_perimeter(x, y, depth):
    """
    Compute wetted perimeter for a given depth.

    Parameters:
        x (array-like): Horizontal distances.
        y (array-like): Elevations.
        depth (float): Depth at which to compute the wetted perimeter.

    Returns:
        perimeter (float): Wetted perimeter.
    """
    logging.debug(f"Computing wetted perimeter with depth={depth}")
    x = np.array(x)
    y = np.array(y)

    # Determine where points are below the specified depth
    below = y < depth

    # Compute differences between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)

    # Compute segment lengths
    lengths = np.sqrt(dx**2 + dy**2)

    # Segments where both points are below depth
    both_below = below[:-1] & below[1:]
    perimeter = lengths[both_below].sum()

    # Segments crossing the depth
    crossing = (below[:-1] & ~below[1:]) | (~below[:-1] & below[1:])
    if np.any(crossing):
        # Avoid division by zero
        dy_cross = dy[crossing]
        dy_cross[dy_cross == 0] = 1e-6  # Small number to prevent division by zero
        t = (depth - y[:-1][crossing]) / dy_cross
        t = np.clip(t, 0, 1)
        submerged_lengths = np.sqrt((dx[crossing] * t)**2 + (dy[crossing] * t)**2)
        perimeter += submerged_lengths.sum()

    logging.debug(f"Wetted perimeter calculated: {perimeter}")
    return perimeter

# ---------------------------- Centerline Side Function ------------------------- #

def determine_side_of_centerline(points, centerline):
    """
    Determine the side of each point relative to the centerline.

    Parameters:
        points (list): List of Shapely Point objects.
        centerline (LineString or MultiLineString): The centerline geometry.

    Returns:
        sides (np.array): Array indicating side (-1: left, 1: right, 0: on centerline).
    """
    logging.debug(f"Determining side of centerline for {len(points)} points.")
    if isinstance(centerline, LineString):
        nearest_line = centerline
        lines = [nearest_line] * len(points)
    elif isinstance(centerline, MultiLineString):
        # Find the closest LineString within the MultiLineString for each point
        lines = [min(centerline.geoms, key=lambda line: line.distance(point)) for point in points]
    else:
        raise ValueError("Unsupported geometry type for centerline.")

    sides = []
    for point, line in zip(points, lines):
        nearest_point_on_centerline = line.interpolate(line.project(point))
        vector_to_point = np.array([point.x - nearest_point_on_centerline.x, point.y - nearest_point_on_centerline.y])
        reference_direction = np.array([line.coords[-1][0] - line.coords[0][0], line.coords[-1][1] - line.coords[0][1]])

        # Compute the sign of the cross product to determine the side
        cross_product = np.cross(reference_direction, vector_to_point)
        side = np.sign(cross_product)
        sides.append(side)

    sides_array = np.array(sides)
    logging.debug(f"Sides determined: {sides_array}")
    return sides_array

# ------------------------- Leveling Point Functions ----------------------------- #

def moving_average(data, window_size=10):
    """
    Compute the moving average of the data with the specified window size.

    Parameters:
        data (np.array): Input data array.
        window_size (int): Number of points in the moving average window.

    Returns:
        np.array: Moving average of the data.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def find_inflection_points(data, window_size=10):
    """
    Find all inflection points in the data based on the moving average.

    Parameters:
        data (np.array): Input data array.
        window_size (int): Window size for the moving average.

    Returns:
        list: Indices of inflection points.
    """
    logging.debug("Computing moving average.")
    ma = moving_average(data, window_size=window_size)

    logging.debug("Computing second derivative.")
    first_derivative = np.gradient(ma)
    second_derivative = np.gradient(first_derivative)

    logging.debug("Identifying inflection points.")
    sign_changes = np.where(np.diff(np.sign(second_derivative)) != 0)[0]

    inflection_indices = sign_changes + window_size // 2  # Adjust index due to moving average
    inflection_indices = inflection_indices.tolist()

    logging.info(f"Inflection points found at indices: {inflection_indices}")
    return inflection_indices

def find_leveling_point_moving_average(depth, ratio, window_size=10, desired_inflection=1):
    """
    Determine the leveling-out depth based on the specified inflection point of the moving average.

    Parameters:
        depth (np.array): Depth values.
        ratio (np.array): Hydraulic radius ratio values.
        window_size (int): Window size for the moving average.
        desired_inflection (int): Which inflection point to select (1 for first, 2 for second, etc.).

    Returns:
        leveling_out_elevation (float or None): Depth where leveling-out occurs.
    """
    inflection_indices = find_inflection_points(ratio, window_size=window_size)

    if len(inflection_indices) >= desired_inflection:
        inflection_index = inflection_indices[desired_inflection - 1]
        if inflection_index < len(depth):
            leveling_out_elevation = depth[inflection_index]
            logging.info(f"Determined leveling-out elevation: {leveling_out_elevation:.2f}m at depth index {inflection_index}")
            return leveling_out_elevation
        else:
            logging.warning(f"Desired inflection point index {inflection_index} is out of bounds. Using maximum depth.")
    else:
        logging.warning(f"Only {len(inflection_indices)} inflection point(s) found. Requested inflection point {desired_inflection}.")

    # Fallback to maximum depth if desired inflection point not found
    if len(depth) > 0:
        leveling_out_elevation = depth[-1]
        logging.info(f"Using maximum depth as leveling-out elevation: {leveling_out_elevation:.2f}m")
        return leveling_out_elevation
    else:
        logging.warning("Depth array is empty. No leveling-out elevation can be determined.")
        return None

# --------------------------- Plotting Function ---------------------------------- #

def plot_cross_section_area_to_wetted_perimeter_ratio(
    x, y, idx='', depth_increment=0.1, fig_output_path='', 
    print_output=True, leveling_out_elevation=None, window_size=10, desired_inflection=1
):
    """
    Plot hydraulic radius vs. depth and overlay the moving average with its selected inflection point.
    Additionally, plot the first and second derivatives of the moving average.

    Parameters:
        x (list): Distances along the line.
        y (list): Raster values at sampled points.
        idx (str): Identifier for the current line (used in filenames).
        depth_increment (float): Increment step for depth.
        fig_output_path (str): Directory to save the figure.
        print_output (bool): Whether to generate and save the plot.
        leveling_out_elevation (float): Depth at which leveling-out occurs.
        window_size (int): Window size for the moving average.
        desired_inflection (int): Which inflection point to select (1 for first, 2 for second, etc.).

    Returns:
        depth (np.array): Depth values.
        ratio (np.array): Hydraulic radius ratio values.
        leveling_out_elevation (float): Depth where leveling-out occurs.
    """
    logging.debug(f"Plotting hydraulic radius vs. depth with parameters: idx={idx}, depth_increment={depth_increment}, leveling_out_elevation={leveling_out_elevation}, window_size={window_size}, desired_inflection={desired_inflection}")
    
    if leveling_out_elevation is not None:
        depth = np.arange(min(y), leveling_out_elevation, depth_increment)
        logging.debug(f"Depth array limited to leveling_out_elevation={leveling_out_elevation}")
    else:
        depth = np.arange(min(y), max(y), depth_increment)

    # Compute cross-sectional areas and wetted perimeters
    y_adjusted = np.maximum(depth[:, np.newaxis] - y, 0)
    areas = np.trapz(y_adjusted, x, axis=1)
    perimeters = np.array([compute_wetted_perimeter(x, y, d) for d in depth])

    # Address RuntimeWarning by introducing epsilon
    epsilon = 1e-6
    safe_perimeters = np.where(perimeters < epsilon, epsilon, perimeters)
    ratio = np.where(perimeters > 0, areas / (safe_perimeters), 0)

    # Compute moving average and find inflection point
    leveling_out_elevation = find_leveling_point_moving_average(depth, ratio, window_size=window_size, desired_inflection=desired_inflection)

    if print_output:
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig.subplots_adjust(hspace=0.4)

        # ----------------- Subplot 1: Hydraulic Radius vs Depth ----------------- #
        axs[0].plot(depth, ratio, marker='o', linestyle='-', label='Hydraulic Radius', color='black')

        # Compute moving average
        ma_ratio = moving_average(ratio, window_size=window_size)
        
        # Adjust ma_depth slicing to match ma_ratio length
        if window_size % 2 == 0:
            # Even window size
            start_idx = (window_size // 2) - 1
            end_idx = -(window_size // 2)
            ma_depth = depth[start_idx:end_idx]
        else:
            # Odd window size
            start_idx = window_size // 2
            end_idx = -(window_size // 2) + 1
            ma_depth = depth[start_idx:end_idx]

        axs[0].plot(ma_depth, ma_ratio, linestyle='-', label=f'{window_size}-point Moving Average', color='blue')

        # Plot inflection point
        if leveling_out_elevation is not None:
            axs[0].axvline(x=leveling_out_elevation, color='red', linestyle='--', label=f'Leveling-Out Depth (Inflection {desired_inflection})')

            # Highlight the inflection point on the plot
            # Find the corresponding ratio value
            inflection_ratio = ratio[np.argmin(np.abs(depth - leveling_out_elevation))]
            axs[0].plot(leveling_out_elevation, inflection_ratio, 'ro')  # Red dot

        # Handle cases where only one side is present
        if len(ma_depth) == len(ma_ratio):
            axs[0].legend()
        else:
            axs[0].legend(loc='upper right')
            axs[0].text(0.5, 0.95, 'Note: Partial data available', transform=axs[0].transAxes, 
                        fontsize=12, verticalalignment='top', horizontalalignment='center', 
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor='orange', facecolor='yellow', alpha=0.5))

        axs[0].set_ylabel('Hydraulic Radius (Area / Wetted Perimeter)')
        axs[0].set_title(f'Hydraulic Radius vs. Depth (Index: {idx})')
        axs[0].grid(True)

        # ----------------- Subplot 2: First Derivative ----------------- #
        # Compute first derivative of the moving average
        first_derivative = np.gradient(ma_ratio)
        # Adjust depth for derivative plot
        derivative_depth = ma_depth

        axs[1].plot(derivative_depth, first_derivative, linestyle='-', color='green', label='First Derivative')
        axs[1].set_ylabel('First Derivative')
        axs[1].set_title(f'First Derivative of Moving Average vs. Depth (Index: {idx})')
        axs[1].legend()
        axs[1].grid(True)

        # ----------------- Subplot 3: Second Derivative ----------------- #
        # Compute second derivative of the moving average
        second_derivative = np.gradient(first_derivative)
        # Adjust depth for second derivative plot
        second_derivative_depth = derivative_depth

        axs[2].plot(second_derivative_depth, second_derivative, linestyle='-', color='purple', label='Second Derivative')
        axs[2].set_xlabel('Depth (m)')
        axs[2].set_ylabel('Second Derivative')
        axs[2].set_title(f'Second Derivative of Moving Average vs. Depth (Index: {idx})')
        axs[2].legend()
        axs[2].grid(True)

        # Save the figure
        if fig_output_path:
            fig_save_path1 = os.path.join(fig_output_path, f'{idx}_hydraulic_radius_vs_depth.png')
            fig.savefig(fig_save_path1)
            plt.close(fig)
            logging.info(f"Saved plot for segment {idx} at {fig_save_path1}")

    return depth, ratio, leveling_out_elevation

# ------------------------------ Main Function ----------------------------------- #

def main(gpkg_path, raster_path, output_folder, output_gpkg_path=None, centerline_gpkg=None, 
         depth_increment=0.1, print_output=True, window_size=10, desired_inflection=1):
    """
    Main function to process GeoPackage lines, compute hydraulic radii, determine leveling-out elevations,
    and generate plots and output GeoPackage.

    Parameters:
        gpkg_path (str): Path to the input GeoPackage containing perpendicular lines.
        raster_path (str): Path to the raster file (e.g., DEM).
        output_folder (str): Directory to save outputs.
        output_gpkg_path (str, optional): Path to save the output GeoPackage polygon.
        centerline_gpkg (str, optional): Path to the GeoPackage containing the centerline.
        depth_increment (float): Increment step for depth in meters.
        print_output (bool): Whether to generate and save plots.
        window_size (int): Window size for the moving average.
        desired_inflection (int): Which inflection point to select (1 for first, 2 for second, etc.).
    """
    # Setup logging
    log_file = os.path.join(output_folder, f'processing_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(level=logging.INFO,  # Set to INFO; change to DEBUG for more details
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

    # Log the path of the executed script
    script_path = os.path.abspath(sys.argv[0])
    logging.info(f"Script executed from path: {script_path}")

    logging.info("Script started.")
    logging.info(f"User-defined parameters:")
    logging.info(f"  gpkg_path: {gpkg_path}")
    logging.info(f"  raster_path: {raster_path}")
    logging.info(f"  output_folder: {output_folder}")
    logging.info(f"  output_gpkg_path: {output_gpkg_path}")
    logging.info(f"  centerline_gpkg: {centerline_gpkg}")
    logging.info(f"  depth_increment: {depth_increment}")
    logging.info(f"  print_output: {print_output}")
    logging.info(f"  window_size: {window_size}")
    logging.info(f"  desired_inflection: {desired_inflection}")

    # Read input GeoPackages
    gdf = gpd.read_file(gpkg_path)
    centerline_gdf = gpd.read_file(centerline_gpkg)
    centerline = centerline_gdf.geometry.iloc[0]

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created output folder at {output_folder}")
    else:
        logging.info(f"Output folder exists at {output_folder}")

    # Remove existing output GeoPackage if it exists
    if output_gpkg_path and os.path.exists(output_gpkg_path):
        os.remove(output_gpkg_path)
        logging.info(f"Removed existing output GeoPackage at {output_gpkg_path}")

    left_points = []
    right_points = []

    with rasterio.open(raster_path) as raster:
        nodata_value = raster.nodata
        total_lines = len(gdf)
        logging.info(f"Total lines to process: {total_lines}")

        # Initialize prior_leveling_out_elevation
        prior_leveling_out_elevation = None

        # Iterate through each line using a for loop to maintain state
        for idx, row in gdf.iterrows():
            line = row.geometry
            if not isinstance(line, (LineString, MultiLineString)):
                geometry_type = line.geom_type
                logging.warning(f"Skipping non-LineString geometry at index {idx} with geometry type: {geometry_type}")
                continue

            logging.info(f'Processing line {idx + 1} of {total_lines}')
            valid_distances, valid_raster_values, valid_points = sample_raster_along_line(
                line, raster, nodata_value=nodata_value
            )

            if len(valid_distances) == 0 or len(valid_points) == 0:
                logging.warning(f"Skipping line {idx} due to all NoData values.")
                continue

            # Determine sides relative to the centerline
            sides = determine_side_of_centerline(valid_points, centerline)

            left_side_indices = np.where(sides < 0)[0]
            right_side_indices = np.where(sides > 0)[0]

            # Check if at least one side is present
            if len(left_side_indices) > 0 or len(right_side_indices) > 0:
                # Initialize variables
                chosen_side_values = None
                chosen_side_points = []
                available_sides = []

                # Process left side if available
                if len(left_side_indices) > 0:
                    left_side_values = np.array(valid_raster_values)[left_side_indices]
                    left_side_points = [valid_points[i] for i in left_side_indices]
                    # Choose left side for maximum elevation
                    max_elevation_left = np.max(left_side_values)
                    logging.info(f"Maximum elevation on left side for line index {idx}: {max_elevation_left:.2f}m")
                    chosen_side_values = left_side_values
                    chosen_side_points = left_side_points
                    available_sides.append('left')

                # Process right side if available
                if len(right_side_indices) > 0:
                    right_side_values = np.array(valid_raster_values)[right_side_indices]
                    right_side_points = [valid_points[i] for i in right_side_indices]
                    # Choose right side for maximum elevation
                    max_elevation_right = np.max(right_side_values)
                    logging.info(f"Maximum elevation on right side for line index {idx}: {max_elevation_right:.2f}m")
                    if chosen_side_values is not None:
                        # Compare which side has higher elevation
                        if max_elevation_right > np.max(chosen_side_values):
                            chosen_side_values = right_side_values
                            chosen_side_points = right_side_points
                            available_sides = ['right']
                    else:
                        chosen_side_values = right_side_values
                        chosen_side_points = right_side_points
                        available_sides.append('right')

                if chosen_side_values is not None:
                    # Find the maximum elevation on the chosen side
                    max_elevation_side = np.max(chosen_side_values)
                    max_depth = max_elevation_side

                    # Find corresponding y values (elevations)
                    y_array = np.array(valid_raster_values)

                    # Call plotting and leveling point determination function with max_depth
                    depth, ratio, leveling_out_elevation = plot_cross_section_area_to_wetted_perimeter_ratio(
                        x=np.array(valid_distances),
                        y=y_array,
                        idx=idx,
                        fig_output_path=output_folder,
                        print_output=print_output,
                        leveling_out_elevation=None,  # Will be determined within the function
                        window_size=window_size,
                        desired_inflection=desired_inflection
                    )

                    # Save hydraulic radius vs depth points to a CSV file
                    if depth is not None and ratio is not None:
                        df = pd.DataFrame({'Depth_m': depth, 'Hydraulic_Radius': ratio})
                        csv_path = os.path.join(output_folder, f'hydraulic_radius_depth_line_{idx}.csv')
                        df.to_csv(csv_path, index=False)
                        logging.info(f"Saved hydraulic radius vs depth data for line {idx} at {csv_path}")
                    else:
                        logging.warning(f"Depth and ratio data not available for line index {idx}. Skipping CSV save.")

                    # Use the leveling_out_elevation from the plot function
                    if leveling_out_elevation is None:
                        if prior_leveling_out_elevation is not None:
                            leveling_out_elevation = prior_leveling_out_elevation
                            logging.info(f"Using prior leveling-out elevation {leveling_out_elevation:.2f}m for line index {idx}")
                        else:
                            logging.warning(f"No leveling-out elevation available for line index {idx}. Skipping this segment.")
                            continue  # Skip if no leveling elevation is available

                    # Update the prior_leveling_out_elevation for the next iteration
                    prior_leveling_out_elevation = leveling_out_elevation

                    # Find the point closest to the leveling_out_elevation on each side
                    closest_left_point = None
                    closest_right_point = None

                    if len(left_side_indices) > 0:
                        closest_left_idx = left_side_indices[np.argmin(np.abs(left_side_values - leveling_out_elevation))]
                        closest_left_point = valid_points[closest_left_idx]
                    
                    if len(right_side_indices) > 0:
                        closest_right_idx = right_side_indices[np.argmin(np.abs(right_side_values - leveling_out_elevation))]
                        closest_right_point = valid_points[closest_right_idx]

                    # Append points if both sides are present
                    if closest_left_point and closest_right_point:
                        left_points.append(closest_left_point)
                        right_points.append(closest_right_point)
                        logging.info(f"Leveling out point on left side for line index {idx}: {closest_left_point} at elevation {leveling_out_elevation:.2f}m")
                        logging.info(f"Leveling out point on right side for line index {idx}: {closest_right_point} at elevation {leveling_out_elevation:.2f}m")
                    else:
                        # If one side is missing, log the available side
                        if closest_left_point:
                            left_points.append(closest_left_point)
                            logging.info(f"Leveling out point on left side for line index {idx}: {closest_left_point} at elevation {leveling_out_elevation:.2f}m")
                        if closest_right_point:
                            right_points.append(closest_right_point)
                            logging.info(f"Leveling out point on right side for line index {idx}: {closest_right_point} at elevation {leveling_out_elevation:.2f}m")
            else:
                logging.warning(f"Could not find any side elevations for line index {idx}. Skipping this segment.")
                # Generate a plot indicating missing data
                if print_output:
                    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
                    axs.text(0.5, 0.5, 'Insufficient data to plot Hydraulic Radius and Derivatives.', 
                             horizontalalignment='center', verticalalignment='center', fontsize=14, color='red')
                    axs.set_title(f'Insufficient Data for Plotting (Index: {idx})')
                    axs.axis('off')
                    
                    # Save the figure
                    fig_save_path1 = os.path.join(output_folder, f'{idx}_insufficient_data.png')
                    fig.savefig(fig_save_path1)
                    plt.close(fig)
                    logging.info(f"Saved insufficient data plot for segment {idx} at {fig_save_path1}")

    # Create polygon from left and right points
    if left_points and right_points:
        try:
            # Ensure the lists have the same length
            min_length = min(len(left_points), len(right_points))
            left_points_trimmed = left_points[:min_length]
            right_points_trimmed = right_points[:min_length]

            polygon_coords = left_points_trimmed + right_points_trimmed[::-1] + [left_points_trimmed[0]]
            polygon = Polygon([(point.x, point.y) for point in polygon_coords])

            polygon_gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs=gdf.crs)
            polygon_gdf.to_file(output_gpkg_path, driver="GPKG")
            logging.info(f"Polygon GeoPackage created at: {output_gpkg_path}")
        except Exception as e:
            logging.error(f"Failed to create polygon GeoPackage: {e}")
    else:
        logging.warning("No left and right points were collected. Polygon GeoPackage was not created.")

    logging.info("Script finished.")

# ----------------------------- Execution Entry Point ---------------------------- #

if __name__ == "__main__":
    # Define input paths (Hardcoded as per user request)
    perpendiculars_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Valley_CL_perpendiculars_4000m.gpkg"
    raster_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Terrain\WBT_Outputs\filled_dem.tif"
    centerline_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Valley_CL.gpkg"

    # Define depth increment
    depth_increment = 0.01  # Depth increment in meters

    # Define output paths
    output_folder = os.path.join(
        r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\ATD_Streams", 
        f"Results_{depth_increment}_custom_window_and_inflection"
    )
    os.makedirs(output_folder, exist_ok=True)
    
    output_gpkg_name = f"Valley_Footprint_di.gpkg"
    output_gpkg_path = os.path.join(output_folder, output_gpkg_name)

    # Define moving average window size and desired inflection point
    window_size = 20  # Number of points in moving average
    desired_inflection = 1  # 1 for first inflection point, 2 for second, etc.

    # Execute main function
    main(
        gpkg_path=perpendiculars_path,
        raster_path=raster_path,
        output_folder=output_folder,
        output_gpkg_path=output_gpkg_path, 
        centerline_gpkg=centerline_path,
        depth_increment=depth_increment,
        print_output=True,
        window_size=window_size,
        desired_inflection=desired_inflection
    )
