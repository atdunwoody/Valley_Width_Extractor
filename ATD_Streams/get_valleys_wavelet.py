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
from scipy.signal import savgol_filter
import json
from tqdm import tqdm
import pywt  # Added for wavelet decomposition

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
    logging.info(f"Entering sample_raster_along_line with parameters: n_points={n_points}, nodata_value={nodata_value}")
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

    logging.info(f"Sampled {len(values)} valid raster points out of {n_points if n_points else 'variable'} requested.")
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


    return perimeter

# ---------------------------- Centerline Side Function ------------------------- #

def determine_side_of_centerline(points, centerline):
    """
    Determine the side of each point relative to the centerline using the cross product method.

    Parameters:
        points (list): List of Shapely Point objects.
        centerline (LineString or MultiLineString): The centerline geometry.

    Returns:
        sides (np.array): Array indicating side (-1: left, 1: right, 0: on centerline).
    """
    from shapely.ops import linemerge

    if isinstance(centerline, MultiLineString):
        centerline = linemerge(centerline)
    elif not isinstance(centerline, LineString):
        raise TypeError("centerline must be a LineString or MultiLineString")

    # Extract point coordinates
    px = np.array([point.x for point in points])
    py = np.array([point.y for point in points])

    # Get centerline coordinates
    centerline_coords = np.array(centerline.coords)

    # Compute segment lengths and cumulative lengths
    dx = np.diff(centerline_coords[:, 0])
    dy = np.diff(centerline_coords[:, 1])
    segment_lengths = np.hypot(dx, dy)
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))

    # Project points onto the centerline
    proj_distances = np.array([centerline.project(Point(x, y)) for x, y in zip(px, py)])

    # Find the segment index for each projection distance
    segment_indices = np.searchsorted(cumulative_lengths, proj_distances, side='right') - 1
    segment_indices = np.clip(segment_indices, 0, len(dx) - 1)

    # Get the segment coordinates
    x1 = centerline_coords[segment_indices, 0]
    y1 = centerline_coords[segment_indices, 1]
    x2 = centerline_coords[segment_indices + 1, 0]
    y2 = centerline_coords[segment_indices + 1, 1]

    # Compute determinant (cross product)
    det = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)

    # Determine side
    sides = np.zeros(len(points), dtype=int)
    sides[det > 1e-8] = 1
    sides[det < -1e-8] = -1

    return sides

# ------------------------- Damping Onset Function via Wavelet Decomposition ----------------------------- #

def find_damping_onset_by_wavelet(depth, second_derivative, wavelet_threshold=0.1, minimum_depth=1):
    """
    Identify the depth where the oscillations in the second derivative begin to dampen using Morlet wavelet decomposition,
    while ignoring depths within a specified minimum depth threshold.

    Parameters:
        depth (np.array): Array of depth values.
        second_derivative (np.array): Array of second derivative values.
        wavelet_threshold (float): Threshold for wavelet energy to identify damping onset.
        minimum_depth (float): Minimum depth threshold to ignore damping onset before this depth.

    Returns:
        damping_depth (float or None): Depth where damping starts, after minimum depth threshold.
    """
    # Ensure inputs are numpy arrays
    depth = np.array(depth)
    second_derivative = np.array(second_derivative)

    # Perform Continuous Wavelet Transform using Morlet wavelet
    widths = np.arange(1, 31)
    cwtmatr, freqs = pywt.cwt(second_derivative, widths, 'morl')  # Changed 'mexh' to 'morl'

    # Compute the energy (squared coefficients) across scales
    energy = np.sum(np.abs(cwtmatr) ** 2, axis=0)  # Use absolute values for complex coefficients

    # Normalize the energy
    energy = energy / np.max(energy)

    # Find where energy starts to decrease significantly
    threshold = wavelet_threshold * np.max(energy)

    # Only consider depths greater than or equal to minimum depth threshold
    min_depth_value = depth[0] + minimum_depth
    below_threshold_indices = np.where((energy < threshold) & (depth >= min_depth_value))[0]

    if len(below_threshold_indices) == 0:
        logging.warning("Damping onset not detected: energy does not drop below threshold after minimum depth threshold.")
        return None

    # The first index where energy drops below the threshold after minimum depth threshold
    damping_index = below_threshold_indices[0]
    damping_onset_depth = depth[damping_index]

    logging.info(f"Damping onset detected at depth {damping_onset_depth:.2f}m based on Morlet wavelet decomposition.")

    return damping_onset_depth

# --------------------------- Plotting Function ---------------------------------- #

def plot_cross_section_area_to_wetted_perimeter_ratio(
    x, y, idx='', depth_increment=0.1, fig_output_path='', 
    print_output=True, leveling_out_elevation=None, window_size=11, 
    poly_order=2, wavelet_threshold=0.1, minimum_depth = 1):
    """
    Plot hydraulic radius vs. depth and overlay the smoothed data with its derivatives.
    Additionally, plot the first and second derivatives of the smoothed data and the wavelet energy.

    Parameters:
        x (list): Distances along the line.
        y (list): Raster values at sampled points.
        idx (str): Identifier for the current line (used in filenames).
        depth_increment (float): Increment step for depth.
        fig_output_path (str): Directory to save the figure.
        print_output (bool): Whether to generate and save the plot.
        leveling_out_elevation (float): Depth at which leveling-out occurs.
        window_size (int): Window size for the smoothing filter.
        poly_order (int): Polynomial order for the smoothing filter.

    Returns:
        depth (np.array): Depth values.
        ratio (np.array): Hydraulic radius ratio values.
        leveling_out_elevation (float): Depth where leveling-out occurs.
        damping_onset_depth (float or None): Depth where damping starts.
    """
   
    # Define depth array based on the full range since leveling_out_elevation will be determined by damping_onset_depth
    depth = np.arange(min(y), max(y), depth_increment)

    logging.info(f"Depth array created with {len(depth)} points ranging from {depth.min()} to {depth.max()} meters.")

    # Compute cross-sectional areas and wetted perimeters
    y_adjusted = np.maximum(depth[:, np.newaxis] - y, 0)
    areas = np.trapz(y_adjusted, x, axis=1)
    perimeters = np.array([compute_wetted_perimeter(x, y, d) for d in depth])

    # Compute Hydraulic Radius Ratio
    epsilon = 1e-6
    safe_perimeters = np.where(perimeters < epsilon, epsilon, perimeters)
    ratio = np.where(perimeters > 0, areas / (safe_perimeters **2), 0)


    # Smooth the ratio using Savitzky-Golay filter
    try:
        smoothed_ratio = savgol_filter(ratio, window_length=window_size, polyorder=poly_order, mode='constant')
    except ValueError as e:
        logging.error(f"Error in Savitzky-Golay filter: {e}")
        smoothed_ratio = ratio  # Fallback to unsmoothed data

    # Compute first and second derivatives
    first_derivative = np.gradient(smoothed_ratio)
    second_derivative = np.gradient(first_derivative)

    # Find damping onset depth based on wavelet decomposition
    damping_onset_depth = find_damping_onset_by_wavelet(depth, second_derivative, 
                                                        wavelet_threshold=wavelet_threshold,
                                                        minimum_depth=minimum_depth)

    # Set leveling_out_elevation based on damping_onset_depth
    leveling_out_elevation = damping_onset_depth

    if print_output:
        fig, axs = plt.subplots(4, 1, figsize=(12, 24), sharex=True)
        fig.subplots_adjust(hspace=0.4)

        # ----------------- Subplot 1: Hydraulic Radius vs Depth ----------------- #
        axs[0].plot(depth, ratio, marker='o', linestyle='-', label='Hydraulic Radius', color='black')
        axs[0].plot(depth, smoothed_ratio, linestyle='-', label=f'Savitzky-Golay Smoothing (window_size={window_size}, poly_order={poly_order})', color='blue')

        if leveling_out_elevation is not None:
            axs[0].axvline(x=leveling_out_elevation, color='red', linestyle='--', label=f'Damping Onset Depth ({leveling_out_elevation:.2f}m)')

            # Highlight the damping onset point on the plot
            damping_idx = np.argmin(np.abs(depth - leveling_out_elevation))
            damping_ratio = ratio[damping_idx]
            axs[0].plot(leveling_out_elevation, damping_ratio, 'ro')  # Red dot

        # Handle cases where smoothing might not align perfectly
        if len(depth) == len(smoothed_ratio):
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
        axs[1].plot(depth, first_derivative, linestyle='-', color='green', label='First Derivative')
        axs[1].set_ylabel('First Derivative')
        axs[1].set_title(f'First Derivative of Smoothed Ratio vs. Depth (Index: {idx})')
        axs[1].legend()
        axs[1].grid(True)

        # ----------------- Subplot 3: Second Derivative ----------------- #
        axs[2].plot(depth, second_derivative, linestyle='-', color='purple', label='Second Derivative')
        
        if damping_onset_depth is not None:
            axs[2].axvline(x=damping_onset_depth, color='orange', linestyle='--', label='Damping Onset Depth')
            damping_idx = np.argmin(np.abs(depth - damping_onset_depth))
            axs[2].plot(damping_onset_depth, second_derivative[damping_idx], 'o', color='orange')
            axs[2].legend()

        axs[2].set_ylabel('Second Derivative')
        axs[2].set_title(f'Second Derivative of Smoothed Ratio vs. Depth (Index: {idx})')
        axs[2].grid(True)

        # ----------------- Subplot 4: Wavelet Energy ----------------- #
        # Perform wavelet decomposition to compute energy
        widths = np.arange(1, 31)
        cwtmatr, freqs = pywt.cwt(second_derivative, widths, 'mexh')
        energy = np.sum(cwtmatr ** 2, axis=0)
        normalized_energy = energy / np.max(energy)
        axs[3].plot(depth, normalized_energy, linestyle='-', color='magenta', label='Wavelet Energy')
        
        if damping_onset_depth is not None:
            axs[3].axvline(x=damping_onset_depth, color='orange', linestyle='--', label='Damping Onset Depth')
            axs[3].legend()

        axs[3].set_xlabel('Depth (m)')
        axs[3].set_ylabel('Normalized Energy')
        axs[3].set_title(f'Wavelet Energy vs. Depth (Index: {idx})')
        axs[3].grid(True)

        # Save the figure
        if fig_output_path:
            fig_save_path1 = os.path.join(fig_output_path, "Plots", f'{idx}_hydraulic_radius_vs_depth.png')
            os.makedirs(os.path.dirname(fig_save_path1), exist_ok=True)
            fig.savefig(fig_save_path1)
            plt.close(fig)

    return depth, ratio, leveling_out_elevation, damping_onset_depth

# ------------------------------ Main Function ----------------------------------- #

def get_valleys(perpendiculars_path, dem_path, output_folder, output_gpkg_path=None, centerline_gpkg=None, 
         depth_increment=0.1, print_output=True, window_size=11, poly_order=2, wavelet_threshold=0.1, minimum_depth=1):
    """
    Main function to process GeoPackage lines, compute hydraulic radii, determine leveling-out elevations,
    identify damping onset via wavelet decomposition, and generate plots and output GeoPackage.

    Parameters:
        perpendiculars_path (str): Path to the input GeoPackage containing perpendicular lines.
        dem_path (str): Path to the filled dem.
        output_folder (str): Directory to save outputs.
        output_gpkg_path (str, optional): Path to save the output GeoPackage polygon.
        centerline_gpkg (str, optional): Path to the GeoPackage containing the centerline.
        depth_increment (float): Increment step for depth in meters.
        print_output (bool): Whether to generate and save plots.
        window_size (int): Window size for the Savitzky-Golay smoothing filter (must be odd and > poly_order).
        poly_order (int): Polynomial order for the Savitzky-Golay smoothing filter.
    """
    # Setup logging
    log_file = os.path.join(output_folder, f'processing_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    copy_inputs_to_outputs_folder(perpendiculars_path, dem_path, centerline_gpkg, output_folder)
    # Remove all handlers associated with the root logger object (if any)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler for logging INFO and above
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

    # Create console handler for logging WARNING and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log the path of the executed script
    script_path = os.path.abspath(sys.argv[0])
    logging.info(f"Script executed from path: {script_path}")

    logging.info("Script started.")
    logging.info(f"User-defined parameters:")
    logging.info(f"  perpendiculars_path: {perpendiculars_path}")
    logging.info(f"  dem_path: {dem_path}")
    logging.info(f"  output_folder: {output_folder}")
    logging.info(f"  output_gpkg_path: {output_gpkg_path}")   
    logging.info(f"  centerline_gpkg: {centerline_gpkg}")
    logging.info(f"  depth_increment: {depth_increment}")
    logging.info(f"  print_output: {print_output}")
    logging.info(f"  window_size: {window_size}")
    logging.info(f"  poly_order: {poly_order}")
    logging.info(f"  wavelet_threshold: {wavelet_threshold}")
    logging.info(f"  minimum_depth: {minimum_depth}")

    # Read input GeoPackages
    try:
        gdf = gpd.read_file(perpendiculars_path)
        logging.info(f"Read {len(gdf)} perpendicular lines from GeoPackage.")
    except Exception as e:
        logging.error(f"Failed to read GeoPackage at {perpendiculars_path}: {e}")
        return

    try:
        centerline_gdf = gpd.read_file(centerline_gpkg)
        centerline = centerline_gdf.geometry.iloc[0]
        logging.info(f"Centerline geometry loaded from {centerline_gpkg}.")
    except Exception as e:
        logging.error(f"Failed to read centerline GeoPackage at {centerline_gpkg}: {e}")
        return

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created output folder at {output_folder}")
    else:
        logging.info(f"Output folder exists at {output_folder}")

    # Remove existing output GeoPackage if it exists
    if output_gpkg_path and os.path.exists(output_gpkg_path):
        try:
            os.remove(output_gpkg_path)
            logging.info(f"Removed existing output GeoPackage at {output_gpkg_path}")
        except Exception as e:
            logging.error(f"Failed to remove existing GeoPackage at {output_gpkg_path}: {e}")

    left_points = []
    right_points = []
    flow_depths = {}  # Dictionary to store flow depths for each segment

    try:
        with rasterio.open(dem_path) as raster:
            nodata_value = raster.nodata
            total_lines = len(gdf)
            logging.info(f"Total lines to process: {total_lines}")

            # Initialize prior_leveling_out_elevation
            prior_leveling_out_elevation = None

            # Iterate through each line using a for loop to maintain state
            for idx, row in tqdm(gdf.iterrows(), total=total_lines, desc="Processing segments"):
                line = row.geometry
                if not isinstance(line, (LineString, MultiLineString)):
                    geometry_type = line.geom_type
                    logging.warning(f"Skipping non-LineString geometry at index {idx} with geometry type: {geometry_type}")
                    continue

                logging.info(f'Processing line {idx + 1} of {total_lines} (Index: {idx})')

                # Sample raster along the perpendicular line
                valid_distances, valid_raster_values, valid_points = sample_raster_along_line(
                    line, raster, nodata_value=nodata_value
                )

                logging.info(f"Line {idx}: Sampled {len(valid_distances)} distances and {len(valid_points)} valid points.")

                if len(valid_distances) == 0 or len(valid_points) == 0:
                    logging.warning(f"Skipping line {idx} due to all NoData values or no valid points.")
                    continue

                # Determine sides relative to the centerline
                sides = determine_side_of_centerline(valid_points, centerline)

                left_side_indices = np.where(sides < 0)[0]
                right_side_indices = np.where(sides > 0)[0]

                logging.info(f"Line {idx}: Found {len(left_side_indices)} left side points and {len(right_side_indices)} right side points.")

                # Log elevations on each side
                if len(left_side_indices) > 0:
                    left_elevations = np.array(valid_raster_values)[left_side_indices]
                    logging.info(f"Line {idx}: Left side elevations range from {left_elevations.min()}m to {left_elevations.max()}m.")
                if len(right_side_indices) > 0:
                    right_elevations = np.array(valid_raster_values)[right_side_indices]
                    logging.info(f"Line {idx}: Right side elevations range from {right_elevations.min()}m to {right_elevations.max()}m.")

                # Check if at least one side is present
                if len(left_side_indices) > 0 or len(right_side_indices) > 0:
                    # Define y_array as all valid raster values (both sides)
                    y_array = np.array(valid_raster_values)

                    # Call plotting and leveling point determination function with smoothed data
                    depth, ratio, leveling_out_elevation, damping_onset_depth = plot_cross_section_area_to_wetted_perimeter_ratio(
                        x=np.array(valid_distances),
                        y=y_array,
                        idx=idx,
                        fig_output_path=output_folder,
                        print_output=print_output,
                        leveling_out_elevation=None,  # Will be determined within the function based on damping
                        window_size=window_size,
                        poly_order=poly_order,
                        wavelet_threshold=wavelet_threshold,
                        minimum_depth=minimum_depth
                    )

                    # Use the leveling_out_elevation from the plot function (which is damping_onset_depth)
                    if leveling_out_elevation is None:
                        if prior_leveling_out_elevation is not None:
                            leveling_out_elevation = prior_leveling_out_elevation
                            logging.info(f"Line {idx}: Using prior leveling-out elevation {leveling_out_elevation:.2f}m")
                        else:
                            logging.warning(f"Line {idx}: No leveling-out elevation available and no prior elevation to use. Skipping this segment.")
                            continue  # Skip if no leveling elevation is available

                    # Update the prior_leveling_out_elevation for the next iteration
                    prior_leveling_out_elevation = leveling_out_elevation

                    # Store the flow depth for this segment
                    flow_depths[str(idx)] = leveling_out_elevation

                    # Find the point closest to the leveling_out_elevation on each side
                    closest_left_point = None
                    closest_right_point = None

                    if len(left_side_indices) > 0:
                        left_side_elevations = np.array(valid_raster_values)[left_side_indices]
                        # Find the index on the left side closest to leveling_out_elevation
                        closest_left_relative_idx = np.argmin(np.abs(left_side_elevations - leveling_out_elevation))
                        closest_left_idx = left_side_indices[closest_left_relative_idx]
                        closest_left_point = valid_points[closest_left_idx]
                        logging.info(f"Line {idx}: Closest left point elevation {left_side_elevations[closest_left_relative_idx]:.2f}m at index {closest_left_idx}")

                    if len(right_side_indices) > 0:
                        right_side_elevations = np.array(valid_raster_values)[right_side_indices]
                        # Find the index on the right side closest to leveling_out_elevation
                        closest_right_relative_idx = np.argmin(np.abs(right_side_elevations - leveling_out_elevation))
                        closest_right_idx = right_side_indices[closest_right_relative_idx]
                        closest_right_point = valid_points[closest_right_idx]
                        logging.info(f"Line {idx}: Closest right point elevation {right_side_elevations[closest_right_relative_idx]:.2f}m at index {closest_right_idx}")

                    # Append points if both sides are present
                    if closest_left_point and closest_right_point:
                        left_points.append(closest_left_point)
                        right_points.append(closest_right_point)
                        logging.info(f"Line {idx}: Leveling out point on left side: {closest_left_point} at elevation {leveling_out_elevation:.2f}m")
                        logging.info(f"Line {idx}: Leveling out point on right side: {closest_right_point} at elevation {leveling_out_elevation:.2f}m")
                    else:
                        # If one side is missing, log the available side
                        if closest_left_point:
                            left_points.append(closest_left_point)
                            logging.info(f"Line {idx}: Leveling out point on left side: {closest_left_point} at elevation {leveling_out_elevation:.2f}m")
                        if closest_right_point:
                            right_points.append(closest_right_point)
                            logging.info(f"Line {idx}: Leveling out point on right side: {closest_right_point} at elevation {leveling_out_elevation:.2f}m")
                else:
                    logging.warning(f"Line {idx}: Could not find any side elevations. Skipping this segment.")
                    # Generate a plot indicating missing data
                    if print_output:
                        fig, axs = plt.subplots(1, 1, figsize=(12, 6))
                        axs.text(0.5, 0.5, 'Insufficient data to plot Hydraulic Radius and Derivatives.', 
                                 horizontalalignment='center', verticalalignment='center', fontsize=14, color='red')
                        axs.set_title(f'Insufficient Data for Plotting (Index: {idx})')
                        axs.axis('off')
                        
                        # Save the figure
                        fig_save_path1 = os.path.join(output_folder, f'{idx}_insufficient_data.png')
                        try:
                            fig.savefig(fig_save_path1)
                            plt.close(fig)
                            logging.info(f"Line {idx}: Saved insufficient data plot at {fig_save_path1}")
                        except Exception as e:
                            logging.error(f"Line {idx}: Failed to save insufficient data plot at {fig_save_path1}: {e}")

    except Exception as e:
        logging.error(f"Failed to open or process raster at {dem_path}: {e}")
        return

    # Save flow depths to a json file
    json_path = os.path.join(output_folder, 'flow_depths.json')
    try:
        with open(json_path, 'w') as f:
            json.dump(flow_depths, f)
        logging.info(f"Saved flow depths to {json_path}")
    except Exception as e:
        logging.error(f"Failed to save flow depths to json file: {e}")

    # Create polygon from left and right points
    if left_points and right_points:
        try:
            # Ensure the lists have the same length
            min_length = min(len(left_points), len(right_points))
            left_points_trimmed = left_points[:min_length]
            right_points_trimmed = right_points[:min_length]

            logging.info(f"Creating polygon with {min_length} points on each side.")

            # Combine left and right points to form a closed polygon
            polygon_coords = left_points_trimmed + right_points_trimmed[::-1] + [left_points_trimmed[0]]
            polygon = Polygon([(point.x, point.y) for point in polygon_coords])

            polygon_gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs=gdf.crs)
            polygon_gdf.to_file(output_gpkg_path, driver="GPKG")
            logging.info(f"Polygon GeoPackage created at: {output_gpkg_path}")
        except Exception as e:
            logging.error(f"Failed to create polygon GeoPackage: {e}")
    else:
        logging.warning("No left and/or right points were collected. Polygon GeoPackage was not created.")

    logging.info("Script finished.")
    return json_path
# ----------------------------- Execution Entry Point ---------------------------- #

def copy_inputs_to_outputs_folder(perpendiculars_path, dem_path, centerline_gpkg, output_folder):
    """
    Copy input files to the output folder for reference.

    Parameters:
        perpendiculars_path (str): Path to the input GeoPackage containing perpendicular lines.
        dem_path (str): Path to the filled dem.
        centerline_gpkg (str): Path to the GeoPackage containing the centerline.
        output_folder (str): Directory to save outputs.
    """
    try:
        # Copy input files to the output folder
        output_perpendiculars_path = os.path.join(output_folder, "Inputs", os.path.basename(perpendiculars_path))
        output_dem_path = os.path.join(output_folder, "Inputs", os.path.basename(dem_path))
        output_centerline_gpkg = os.path.join(output_folder, "Inputs", os.path.basename(centerline_gpkg))
        os.makedirs(os.path.dirname(output_perpendiculars_path), exist_ok=True)
        # Copy files
        for src, dest in zip([perpendiculars_path, dem_path, centerline_gpkg], 
                             [output_perpendiculars_path, output_dem_path, output_centerline_gpkg]):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.system(f'copy "{src}" "{dest}"')
    except Exception as e:
        logging.error(f"Failed to copy input files to output folder: {e}")

    return output_perpendiculars_path, output_dem_path, output_centerline_gpkg

if __name__ == "__main__":
    from datetime import datetime
    ######################################################################################################################
    ########################################### User-Defined Parameters ##################################################
    ######################################################################################################################
    #perpendiculars_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Valley_CL_single_part.gpkg"
    #dem_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Terrain\WBT_Outputs\filled_dem.tif"
    # centerline_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Valley_CL_perpendiculars_1000m.gpkg"
    # output_dir = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Test"
    perpendiculars_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\ATD_algorithm\Wavelets\ME\WBT_Perpendiculars\ME_WBT_perps_100m_dissolve.gpkg"
    dem_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\dem_2021.tif"
    ###############IMPORTANT################
    # The centerline path must be a multiline string with collected geometries (e.g. only a single line)
    centerline_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\ATD_algorithm\Wavelets\ME\WBT_Perpendiculars\ME_WBT_CL_dissolved.gpkg"
    output_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\ATD_algorithm\Wavelets\ME\WBT_Perpendiculars"

    # Define depth increment
    depth_increment = 0.01  # Depth increment in meters
    # Define smoothing parameters
    window_size = 11  # Must be odd and > poly_order
    poly_order = 9    # Polynomial order for Savitzky-Golay filter
    wavlet_threshold = 0.1
    minimum_depth = 1.5  # Minimum depth threshold to ignore damping onset before this depth
    print_output = False
    ######################################################################################################################
    ############################################## Main Function Call ####################################################
    ################################################################################################################

    # Define output paths
    output_folder = os.path.join(
        output_dir, 
        f"{datetime.now().strftime('%Y-%m-%d_%Hh%Mm')}_10m_{wavlet_threshold}_thresh_{window_size}_win_{poly_order}_poly"
    )
    os.makedirs(output_folder, exist_ok=True)
    
    output_gpkg_name = f"Valley_Footprint_{wavlet_threshold}_thresh.gpkg"
    output_gpkg_path = os.path.join(output_folder, output_gpkg_name)

    start_time = datetime.now()
    # Execute main function
    json_path = get_valleys(
        perpendiculars_path=perpendiculars_path,
        dem_path=dem_path,
        output_folder=output_folder,
        output_gpkg_path=output_gpkg_path, 
        centerline_gpkg=centerline_path,
        depth_increment=depth_increment,
        print_output=print_output,
        window_size=window_size,
        poly_order=poly_order,
        wavelet_threshold=wavlet_threshold,
        minimum_depth=minimum_depth
    )
    print(f"Execution time: {datetime.now() - start_time}")
    from call_plot_cross_sections import run_cross_section_plotting
    run_cross_section_plotting(perpendiculars_path=perpendiculars_path, dem_raster=dem_path, json_file=json_path)

 
