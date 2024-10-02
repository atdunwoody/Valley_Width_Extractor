import os
import sys
import logging
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import split
import rasterio
from datetime import datetime
from tqdm import tqdm
import pywt
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


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
        n_points = max(int(line.length), 1)  # Ensure at least one point
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

    logging.info(f"Sampled {len(values)} valid raster points out of {n_points} requested.")
    return distances_valid.tolist(), values.tolist(), points_valid


def compute_cross_sectional_area_trapezoidal(x, y, depth_increment=0.1):
    """
    Compute cross-sectional areas and hydraulic radius ratio using the trapezoidal rule.

    Parameters:
        x (list or np.array): Distances along the line.
        y (list or np.array): Elevations.
        depth_increment (float): Increment step for depth in meters.

    Returns:
        depth (np.array): Depth values.
        ratio (np.array): Hydraulic radius ratio values.
    """
    if len(y) == 0:
        logging.warning("Empty elevation array provided to compute_cross_sectional_area_trapezoidal.")
        return np.array([]), np.array([])

    min_y = min(y)
    max_y = max(y)

    if min_y == max_y:
        logging.warning("All elevation values are identical. Creating a default depth range.")
        # Define a small range around the constant elevation
        depth = np.linspace(min_y - 0.5, max_y + 0.5, num=5)
    else:
        depth = np.arange(min_y, max_y, depth_increment)

    if len(depth) == 0:
        logging.warning("Depth array is empty after initialization.")
        return depth, np.array([])

    logging.info(f"Depth array created with {len(depth)} points ranging from {depth.min()} to {depth.max()} meters.")

    # Compute cross-sectional areas and wetted perimeters
    y = np.array(y)
    x = np.array(x)

    y_adjusted = np.maximum(depth[:, np.newaxis] - y, 0)
    areas = np.trapz(y_adjusted, x, axis=1)
    perimeters = np.array([compute_wetted_perimeter(x, y, d) for d in depth])

    # Compute Hydraulic Radius Ratio
    epsilon = 1e-6
    safe_perimeters = np.where(perimeters < epsilon, epsilon, perimeters)
    ratio = np.where(perimeters > 0, areas / (safe_perimeters ** 2), 0)

    return depth, ratio


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
        submerged_lengths = np.sqrt((dx[crossing] * t) ** 2 + (dy[crossing] * t) ** 2)
        perimeter += submerged_lengths.sum()

    return perimeter


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
    if len(depth) == 0 or len(second_derivative) == 0:
        logging.warning("Empty depth or second_derivative array provided to find_damping_onset_by_wavelet.")
        return None

    # Ensure inputs are numpy arrays
    depth = np.array(depth)
    second_derivative = np.array(second_derivative)

    # Perform Continuous Wavelet Transform using Morlet wavelet
    widths = np.arange(1, 31)
    try:
        cwtmatr, freqs = pywt.cwt(second_derivative, widths, 'morl')
    except Exception as e:
        logging.error(f"Error performing CWT: {e}")
        return None

    # Compute the energy (squared coefficients) across scales
    energy = np.sum(np.abs(cwtmatr) ** 2, axis=0)  # Use absolute values for complex coefficients

    if np.max(energy) == 0:
        logging.warning("Energy array contains all zeros. Cannot normalize.")
        return None

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


def plot_cross_section_area_to_wetted_perimeter_ratio(
    x, y, idx='', depth_increment=0.1, fig_output_path='', 
    print_output=True, leveling_out_elevation=None, window_size=11, 
    poly_order=2, wavelet_threshold=0.1, minimum_depth=1):
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
        wavelet_threshold (float): Threshold for wavelet energy to identify damping onset.
        minimum_depth (float): Minimum depth threshold to ignore damping onset before this depth.

    Returns:
        depth (np.array): Depth values.
        ratio (np.array): Hydraulic radius ratio values.
        leveling_out_elevation (float): Depth where leveling-out occurs.
        damping_onset_depth (float or None): Depth where damping starts.
    """
    if not print_output:
        return

    # Define depth array based on the full range since leveling_out_elevation will be determined by damping_onset_depth
    depth, ratio = compute_cross_sectional_area_trapezoidal(x, y, depth_increment)
    
    if len(depth) == 0 or len(ratio) == 0:
        logging.warning(f"Insufficient data to plot for idx {idx}.")
        return

    logging.info(f"Plotting cross-section for idx {idx}.")

    # Smooth the ratio using Savitzky-Golay filter
    try:
        smoothed_ratio = savgol_filter(ratio, window_length=window_size, polyorder=poly_order, mode='interp')
    except ValueError as e:
        logging.error(f"Error in Savitzky-Golay filter for plotting idx {idx}: {e}")
        smoothed_ratio = ratio  # Fallback to unsmoothed data

    # Compute first and second derivatives
    first_derivative = np.gradient(smoothed_ratio)
    second_derivative = np.gradient(first_derivative)

    # Perform wavelet decomposition to compute energy
    widths = np.arange(1, 31)
    try:
        cwtmatr, freqs = pywt.cwt(second_derivative, widths, 'mexh')
        energy = np.sum(cwtmatr ** 2, axis=0)
        normalized_energy = energy / np.max(energy)
    except Exception as e:
        logging.error(f"Error performing wavelet decomposition for plotting idx {idx}: {e}")
        normalized_energy = np.zeros_like(second_derivative)

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
    
    if leveling_out_elevation is not None:
        axs[2].axvline(x=leveling_out_elevation, color='orange', linestyle='--', label='Damping Onset Depth')
        damping_idx = np.argmin(np.abs(depth - leveling_out_elevation))
        axs[2].plot(leveling_out_elevation, second_derivative[damping_idx], 'o', color='orange')
        axs[2].legend()

    axs[2].set_ylabel('Second Derivative')
    axs[2].set_title(f'Second Derivative of Smoothed Ratio vs. Depth (Index: {idx})')
    axs[2].grid(True)

    # ----------------- Subplot 4: Wavelet Energy ----------------- #
    axs[3].plot(depth, normalized_energy, linestyle='-', color='magenta', label='Wavelet Energy')
    
    if leveling_out_elevation is not None:
        axs[3].axvline(x=leveling_out_elevation, color='orange', linestyle='--', label='Damping Onset Depth')
        axs[3].legend()

    axs[3].set_xlabel('Depth (m)')
    axs[3].set_ylabel('Normalized Energy')
    axs[3].set_title(f'Wavelet Energy vs. Depth (Index: {idx})')
    axs[3].grid(True)

    # Save the figure
    if fig_output_path:
        plot_dir = os.path.join(fig_output_path, "Plots")
        os.makedirs(plot_dir, exist_ok=True)
        fig_save_path = os.path.join(plot_dir, f'{idx}_hydraulic_radius_vs_depth.png')
        fig.savefig(fig_save_path)
        plt.close(fig)
        logging.info(f"Plot saved to {fig_save_path}.")


def process_damping_depth(x, y, idx='', depth_increment=0.1, wavelet_threshold=0.1, minimum_depth=1):
    """
    Compute hydraulic metrics and identify damping onset depth.

    Parameters:
        x (list or np.array): Distances along the line.
        y (list or np.array): Elevations.
        idx (str): Identifier for the current line (used in logging).
        depth_increment (float): Increment step for depth.
        wavelet_threshold (float): Threshold for wavelet energy to identify damping onset.
        minimum_depth (float): Minimum depth threshold to ignore damping onset before this depth.

    Returns:
        damping_onset_depth (float or None): Depth where damping starts.
    """
    if len(x) == 0 or len(y) == 0:
        logging.warning(f"No data provided to process_damping_depth for perp_idx {idx}.")
        return None

    logging.info(f"Processing damping depth for perp_idx {idx}")

    # Compute cross-sectional area and hydraulic radius ratio
    depth, ratio = compute_cross_sectional_area_trapezoidal(x, y, depth_increment=depth_increment)

    if len(depth) == 0 or len(ratio) == 0:
        logging.warning(f"Empty depth or ratio array after compute_cross_sectional_area_trapezoidal for perp_idx {idx}.")
        return None

    # Smooth the ratio using Savitzky-Golay filter
    try:
        window_length = 11 if len(ratio) >= 11 else (len(ratio)//2)*2 + 1  # Ensure window_length is odd

        #ensure that the window length is less than the length of the data
        if window_length > len(ratio):
            window_length = len(ratio) 
            #make sure that the window length is odd
            if window_length % 2 == 0:
                window_length -= 1
        if window_length < 3 and len(ratio) > 2:
            smoothed_ratio = ratio
        else:
            smoothed_ratio = savgol_filter(ratio, window_length=window_length, polyorder=2)
    except ValueError as e:
        logging.error(f"Error in Savitzky-Golay filter for perp_idx {idx}: {e}")
        smoothed_ratio = ratio  # Fallback to unsmoothed data

    # Compute first and second derivatives
    if len(smoothed_ratio) < 2:
        logging.warning(f"Insufficient data for derivatives for perp_idx {idx}.")
        return None
    first_derivative = np.gradient(smoothed_ratio)
    second_derivative = np.gradient(first_derivative)

    # Find damping onset depth
    damping_onset_depth = find_damping_onset_by_wavelet(
        depth, second_derivative,
        wavelet_threshold=wavelet_threshold,
        minimum_depth=minimum_depth
    )

    return damping_onset_depth


def determine_side_of_centerline(perpendiculars_path, centerlines_path, dem_path, output_dir, print_output=True,
                                 minimum_depth=1, wavelet_threshold=0.1):
    """
    Determine the side of each perpendicular point relative to the centerline and compute damping onset elevations.

    Parameters:
        perpendiculars_path (str): Path to the input GeoPackage containing perpendicular lines.
        centerlines_path (str): Path to the GeoPackage containing centerline(s).
        dem_path (str): Path to the DEM raster.
        output_dir (str): Directory to save output files.
        print_output (bool): Whether to generate and save plots.
        minimum_depth (float): Minimum depth threshold to ignore damping onset before this depth.
        wavelet_threshold (float): Threshold for wavelet energy to identify damping onset.

    Returns:
        damping_polygon_path (str or None): Path to the damping onset polygon GeoPackage.
        damping_onset_json_path (str or None): Path to damping onset depths JSON file.
    """
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

    # Add 'perp_idx' field to perpendiculars_gdf if not present
    if 'perp_idx' not in perpendiculars_gdf.columns:
        perpendiculars_gdf = perpendiculars_gdf.reset_index().rename(columns={'index': 'perp_idx'})
        logging.info("'perp_idx' field created in perpendiculars_gdf.")
    else:
        logging.info("'perp_idx' field already exists in perpendiculars_gdf.")

    # Save the updated perpendiculars_gdf with 'perp_idx'
    perpendiculars_with_idx_path = os.path.join(output_dir, "perpendiculars_with_idx.gpkg")
    perpendiculars_gdf.to_file(perpendiculars_with_idx_path, driver="GPKG")
    logging.info(f"Perpendiculars with 'perp_idx' saved to {perpendiculars_with_idx_path}.")

    # Open DEM raster
    logging.info("Opening DEM raster...")
    try:
        dem_raster = rasterio.open(dem_path)
    except Exception as e:
        logging.error(f"Error opening DEM raster: {e}")
        return None, None

    # Initialize lists to collect points
    side1_records = []
    side2_records = []
    damping_onset_records = []

    # Read centerline as unary_union
    if centerlines_gdf.empty:
        logging.error("No centerline geometries found. Exiting processing.")
        dem_raster.close()
        return None, None

    centerline = centerlines_gdf.unary_union  # Merge all centerlines into a single geometry

    # Iterate over each perpendicular line
    logging.info("Processing perpendicular lines...")
    for idx, perp_row in tqdm(perpendiculars_gdf.iterrows(), total=len(perpendiculars_gdf), desc="Perpendiculars"):
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
        side1_distances = []
        side1_elevations = []
        side1_points = []

        side2_distances = []
        side2_elevations = []
        side2_points = []

        for distance, elevation, point in zip(distances, elevations, sampled_points):
            record = {'geometry': point, 'elevation': elevation, 'perp_idx': perp_idx, 'distance_along': distance}
            if buffer1.contains(point):
                side1_records.append(record)
                side1_distances.append(distance)
                side1_elevations.append(elevation)
                side1_points.append(point)
            elif buffer2.contains(point):
                side2_records.append(record)
                side2_distances.append(distance)
                side2_elevations.append(elevation)
                side2_points.append(point)
            else:
                logging.debug(f"Point {point} does not fall within any buffer. Skipping.")

        # Compute damping onset elevation for Side 1
        damping_onset_depth_side1 = None
        if side1_distances and side1_elevations:
            damping_onset_depth_side1 = process_damping_depth(
                x=side1_distances,
                y=side1_elevations,
                idx=f"{perp_idx}_side1",
                depth_increment=0.1,
                wavelet_threshold=wavelet_threshold,
                minimum_depth=minimum_depth
            )
        else:
            logging.info(f"No points on Side1 for perp_idx {perp_idx}.")

        # Compute damping onset elevation for Side 2
        damping_onset_depth_side2 = None
        if side2_distances and side2_elevations:
            damping_onset_depth_side2 = process_damping_depth(
                x=side2_distances,
                y=side2_elevations,
                idx=f"{perp_idx}_side2",
                depth_increment=0.1,
                wavelet_threshold=wavelet_threshold,
                minimum_depth=minimum_depth
            )
        else:
            logging.info(f"No points on Side2 for perp_idx {perp_idx}.")

        # Make damping onset depth the minimum of the two sides if both are detected
        if damping_onset_depth_side1 is not None and damping_onset_depth_side2 is not None:
            damping_onset_depth = min(damping_onset_depth_side1, damping_onset_depth_side2)
        elif damping_onset_depth_side1 is not None:
            damping_onset_depth = damping_onset_depth_side1
        elif damping_onset_depth_side2 is not None:
            damping_onset_depth = damping_onset_depth_side2
        else:
            damping_onset_depth = None

        if damping_onset_depth is not None:
            # For each side, find the closest point to the damping_onset_depth
            for side, distances_side, elevations_side, points_side in [
                ('Side1', side1_distances, side1_elevations, side1_points),
                ('Side2', side2_distances, side2_elevations, side2_points)
            ]:
                if distances_side and elevations_side and points_side:
                    closest_index = np.argmin(np.abs(np.array(elevations_side) - damping_onset_depth))
                    closest_point = points_side[closest_index]
                    damping_record = {
                        'geometry': closest_point,
                        'damping_onset_depth': damping_onset_depth,  # Ideal elevation
                        'point_elevation': elevations_side[closest_index],  # Actual elevation
                        'perp_idx': perp_idx,
                        'side': side,
                        'distance_along': distances_side[closest_index]
                    }
                    damping_onset_records.append(damping_record)
                    logging.info(f"Damping onset for perp_idx {perp_idx} {side} at elevation {elevations_side[closest_index]:.2f}m.")

                    # Plotting
                    plot_cross_section_area_to_wetted_perimeter_ratio(
                        x=distances_side,
                        y=elevations_side,
                        idx=f"{perp_idx}_{side.lower()}",
                        depth_increment=0.1,
                        fig_output_path=output_dir,
                        print_output=print_output,
                        leveling_out_elevation=damping_onset_depth,
                        window_size=11,
                        poly_order=2,
                        wavelet_threshold=wavelet_threshold,
                        minimum_depth=minimum_depth
                    )
        else:
            logging.info(f"No damping onset detected for perp_idx {perp_idx}.")

    # Close the raster
    dem_raster.close()

    # Create GeoDataFrames for each side
    logging.info("Creating GeoDataFrame for Side 1...")
    side1_output_path = None
    if side1_records:
        side1_gdf = gpd.GeoDataFrame(side1_records, crs=perpendiculars_gdf.crs)
        side1_output_path = os.path.join(output_dir, "side1_points.gpkg")
        side1_gdf.to_file(side1_output_path, driver="GPKG")
        logging.info(f"Side 1 points saved to {side1_output_path} with {len(side1_gdf)} points.")
    else:
        logging.info("No points assigned to Side 1.")

    logging.info("Creating GeoDataFrame for Side 2...")
    side2_output_path = None
    if side2_records:
        side2_gdf = gpd.GeoDataFrame(side2_records, crs=perpendiculars_gdf.crs)
        side2_output_path = os.path.join(output_dir, "side2_points.gpkg")
        side2_gdf.to_file(side2_output_path, driver="GPKG")
        logging.info(f"Side 2 points saved to {side2_output_path} with {len(side2_gdf)} points.")
    else:
        logging.info("No points assigned to Side 2.")

    # Create GeoDataFrame for Damping Onset Points
    logging.info("Creating GeoDataFrame for Damping Onset Points...")
    damping_onset_output_path = None
    if damping_onset_records:
        damping_gdf = gpd.GeoDataFrame(damping_onset_records, crs=perpendiculars_gdf.crs)
        damping_onset_output_path = os.path.join(output_dir, "damping_onset_points.gpkg")
        damping_gdf.to_file(damping_onset_output_path, driver="GPKG")
        logging.info(f"Damping onset points saved to {damping_onset_output_path} with {len(damping_gdf)} points.")
    else:
        logging.info("No damping onset points detected.")

    # Save damping onset depths to JSON
    logging.info("Saving damping onset depths to JSON...")
    damping_onset_json_path = None
    if damping_onset_records:
        damping_onset_json = {}
        for record in damping_onset_records:
            perp_id = record['perp_idx']
            damping_depth = record['damping_onset_depth']

            # Convert perp_id to string to match the example structure
            perp_id_str = str(perp_id)

            # If a perp_idx already exists, ensure consistency
            if perp_id_str in damping_onset_json:
                # Optional: Check if the existing depth is the same
                if damping_onset_json[perp_id_str] != damping_depth:
                    logging.warning(f"Inconsistent damping_onset_depth for perp_idx {perp_id_str}. Overwriting with new value.")

            # Assign damping_onset_depth to perp_idx
            damping_onset_json[perp_id_str] = damping_depth

        damping_onset_json_path = os.path.join(output_dir, "damping_onset_depths.json")
        try:
            with open(damping_onset_json_path, 'w') as json_file:
                json.dump(damping_onset_json, json_file, indent=4)
            logging.info(f"Damping onset depths saved to {damping_onset_json_path}.")
        except Exception as e:
            logging.error(f"Failed to save damping onset depths to JSON: {e}")
            damping_onset_json_path = None
    else:
        logging.warning("No damping onset records to save to JSON.")

    # Create the damping onset polygon from damping_onset_records
    damping_polygon_path = None
    if damping_onset_records:
        damping_gdf = gpd.GeoDataFrame(damping_onset_records, crs=perpendiculars_gdf.crs)

        # Separate Side1 and Side2 points
        side1_gdf = damping_gdf[damping_gdf['side'] == 'Side1'].sort_values('perp_idx')
        side2_gdf = damping_gdf[damping_gdf['side'] == 'Side2'].sort_values('perp_idx')

        # Ensure there are points on both sides
        if not side1_gdf.empty and not side2_gdf.empty:
            # Extract coordinates
            side1_coords = list(side1_gdf.geometry.apply(lambda point: (point.x, point.y)))
            side2_coords = list(side2_gdf.geometry.apply(lambda point: (point.x, point.y)))

            # Reverse Side2 coordinates to ensure proper polygon closure
            side2_coords_reversed = side2_coords[::-1]

            # Combine coordinates to form the polygon exterior
            polygon_coords = side1_coords + side2_coords_reversed + [side1_coords[0]]

            # Create the polygon
            damping_polygon = gpd.GeoDataFrame({
                'geometry': [Polygon(polygon_coords)],
                # Convert perp_indices to a comma-separated string if needed
                'perp_indices': [', '.join(map(str, damping_gdf['perp_idx'].unique()))]
            }, crs=perpendiculars_gdf.crs)

            # Save the polygon to a GeoPackage
            damping_polygon_path = os.path.join(output_dir, "damping_onset_polygon.gpkg")
            damping_polygon.to_file(damping_polygon_path, driver="GPKG")
            logging.info(f"Damping onset polygon saved to {damping_polygon_path}.")
        else:
            logging.warning("Insufficient points on one or both sides to create a polygon.")
    else:
        logging.warning("No damping onset records available to create a polygon.")

    return damping_polygon_path, damping_onset_json_path


def get_valleys(perpendiculars_path, centerlines_path, dem_path, output_dir, 
               wavelet_threshold=0.1, minimum_depth=1, print_output=True):
    """
    Main function to process GeoPackage lines, compute hydraulic radii, determine damping onset elevations,
    and generate output GeoPackages.

    Parameters:
        perpendiculars_path (str): Path to the input GeoPackage containing perpendicular lines.
        centerlines_path (str): Path to the GeoPackage containing centerline(s).
        dem_path (str): Path to the DEM raster.
        output_dir (str): Directory to save outputs.
        wavelet_threshold (float): Threshold for wavelet energy to identify damping onset.
        minimum_depth (float): Minimum depth threshold to ignore damping onset before this depth.
        print_output (bool): Whether to generate and save plots.

    Returns:
        damping_polygon_path (str or None): Path to the damping onset polygon GeoPackage.
        damping_onset_json_path (str or None): Path to damping onset depths JSON file.
    """
    # Setup logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_file = os.path.join(output_dir, f'processing_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    # Create file handler for logging INFO and above
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

    # Create console handler for logging WARNING and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

    # Add handlers to logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set to lowest level to allow all messages to be handled by handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log the path of the executed script
    script_path = os.path.abspath(sys.argv[0])
    logging.info(f"Script executed from path: {script_path}")

    logging.info("Script started.")
    logging.info(f"User-defined parameters:")
    logging.info(f"  perpendiculars_path: {perpendiculars_path}")
    logging.info(f"  centerlines_path: {centerlines_path}")
    logging.info(f"  dem_path: {dem_path}")
    logging.info(f"  output_dir: {output_dir}")
    logging.info(f"  wavelet_threshold: {wavelet_threshold}")
    logging.info(f"  minimum_depth: {minimum_depth}")
    logging.info(f"  print_output: {print_output}")

    # Process perpendiculars to determine sides and damping onset
    damping_polygon_path, damping_onset_json_path = determine_side_of_centerline(
        perpendiculars_path, centerlines_path, dem_path, output_dir, 
        print_output=print_output, minimum_depth=minimum_depth, 
        wavelet_threshold=wavelet_threshold
    )

    logging.info("Script completed.")

    return damping_polygon_path, damping_onset_json_path


def main():
    """
    Entry point for the script when run directly.
    Modify this function as needed to pass different parameters.
    """
    # Example usage with hard-coded paths
    # Replace these paths with desired input paths or make them configurable
    perpendiculars_path = r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\LM2_perpendiculars_100m.gpkg"
    centerlines_path = r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\LM2_centerline.gpkg"
    dem_path = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\WBT_Outputs_Low\filled_dem.tif"
    output_dir = r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\Wavelets\LM2"

    # Append current datetime to output directory
    output_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%Hh%Mm"))
    os.makedirs(output_dir, exist_ok=True)

    print_output = False  # Set to False to disable plotting

    get_valleys(
        perpendiculars_path=perpendiculars_path,
        centerlines_path=centerlines_path,
        dem_path=dem_path,
        output_dir=output_dir,
        wavelet_threshold=0.1,
        minimum_depth=1,
        print_output=print_output  
    )


if __name__ == "__main__":
    main()
