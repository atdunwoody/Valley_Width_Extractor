import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
import os
from scipy.signal import savgol_filter
import json  # Added for JSON operations
from tqdm import tqdm  # Added for progress bar
from sklearn.preprocessing import StandardScaler  # Added for scaling

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
    cols = cols.astype(int)
    rows = rows.astype(int)

    # Get raster dimensions
    raster_height, raster_width = raster.read(1, masked=False).shape

    # Create mask for valid rows and cols
    valid_mask = (rows >= 0) & (rows < raster_height) & (cols >= 0) & (cols < raster_width)

    # Log warnings for invalid points
    invalid_indices = np.where(~valid_mask)[0]
    for i in invalid_indices:
        print(f"Point at distance {distances[i]:.2f}m falls outside raster bounds (row: {rows[i]}, col: {cols[i]}) and will be skipped.")

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
            print("All sampled raster values are NoData.")
            return [], [], []
        values = values[nodata_mask]
        distances_valid = distances_valid[nodata_mask]
        points_valid = [points_valid[i] for i in range(len(points_valid)) if nodata_mask[i]]

    return distances_valid.tolist(), values.tolist(), points_valid


def determine_side_of_centerline(points, centerline):
    """
    Determine the side of each point relative to the centerline.

    Parameters:
        points (list): List of Shapely Point objects.
        centerline (LineString or MultiLineString): The centerline geometry.

    Returns:
        sides (np.array): Array indicating side (-1: left, 1: right, 0: on centerline).
    """
    if isinstance(centerline, MultiLineString):
        # Combine all LineStrings into a single LineString
        centerline = LineString([pt for line in centerline.geoms for pt in line.coords])
    elif not isinstance(centerline, LineString):
        raise TypeError("centerline must be a LineString or MultiLineString")

    center_length = centerline.length
    sides = np.zeros(len(points), dtype=int)

    for i, point in enumerate(points):
        proj_distance = centerline.project(point)
        proj_point = centerline.interpolate(proj_distance)

        # Avoid projecting beyond the end using a small epsilon
        epsilon = 1e-6
        next_proj_distance = proj_distance + epsilon if proj_distance + epsilon < center_length else proj_distance - epsilon
        tangent_point = centerline.interpolate(next_proj_distance)

        dx, dy = tangent_point.x - proj_point.x, tangent_point.y - proj_point.y
        length = np.hypot(dx, dy)
        if length == 0:
            sides[i] = 0
            continue

        # Normalize the tangent vector to get the reference direction
        nx, ny = -dy / length, dx / length  # Normal vector
        vx, vy = point.x - proj_point.x, point.y - proj_point.y  # Vector from projection to point

        # Compute the dot product
        dot = vx * nx + vy * ny
        if np.isclose(dot, 0, atol=1e-8):
            sides[i] = 0
        elif dot > 0:
            sides[i] = 1
        else:
            sides[i] = -1

    return sides

def compute_cross_sectional_area_trapezoidal(x, y, depth):
    """
    Vectorized computation of cross-sectional area using the trapezoidal rule.

    Parameters:
        x (np.ndarray): Horizontal distances.
        y (np.ndarray): Elevations.
        depth (float): Depth at which to compute the area.

    Returns:
        area (float): Cross-sectional area.
    """
    y_adjusted = np.clip(depth - y, 0, None)
    area = np.trapz(y_adjusted, x)
    return area


def compute_wetted_perimeter(x, y, depth):
    """
    Vectorized computation of wetted perimeter for a given depth.

    Parameters:
        x (np.ndarray): Horizontal distances.
        y (np.ndarray): Elevations.
        depth (float): Depth at which to compute the wetted perimeter.

    Returns:
        perimeter (float): Wetted perimeter.
    """
    y = np.array(y)
    below = y < depth

    dx = np.diff(x)
    dy = np.diff(y)

    # Segments where both points are below depth
    both_below = below[:-1] & below[1:]
    perimeter = np.sum(np.sqrt(dx[both_below]**2 + dy[both_below]**2))

    # Segments crossing the depth
    crossing = (below[:-1] & ~below[1:]) | (~below[:-1] & below[1:])
    if np.any(crossing):
        dy_cross = dy[crossing]
        dy_cross = np.where(dy_cross == 0, 1e-6, dy_cross)  # Prevent division by zero
        t = (depth - y[:-1][crossing]) / dy_cross
        t = np.clip(t, 0, 1)
        submerged_lengths = np.sqrt((dx[crossing] * t)**2 + (dy[crossing] * t)**2)
        perimeter += np.sum(submerged_lengths)

    return perimeter


def smooth_ratio(ratio, window_length=11, polyorder=3):
    """
    Smooth the ratio using Savitzky-Golay filter.

    Parameters:
        ratio (np.ndarray): The ratio array to smooth.
        window_length (int): The length of the filter window (must be odd).
        polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
        smoothed_ratio (np.ndarray): The smoothed ratio array.
    """
    # Ensure window_length is appropriate given the size of the ratio array
    if len(ratio) < window_length:
        window_length = len(ratio)
        if window_length % 2 == 0:
            window_length -= 1

    # Ensure polyorder is less than window_length
    if polyorder >= window_length:
        polyorder = window_length - 1

    return savgol_filter(ratio, window_length, polyorder)


def find_leveling_point_depth(x, y, depth_increment=0.05, polynomial_order=4, 
                             smooth=True, window_length=11, polyorder=9, 
                             percentile=20, max_depth=None):
    """
    Vectorized determination of leveling out elevation based on cross-sectional area and wetted perimeter ratios.
    
    Parameters:
        x (np.ndarray): Distances along the line.
        y (np.ndarray): Raster values at sampled points.
        depth_increment (float): Increment step for depth.
        polynomial_order (int): Order of the polynomial fit.
        smooth (bool): Whether to smooth the ratio data.
        window_length (int): Window length for smoothing.
        polyorder (int): Polynomial order for smoothing.
        percentile (float): Percentile for channel depth threshold.
        max_depth (float, optional): Maximum depth to consider.
    
    Returns:
        leveling_out_elevation (float or None): The determined leveling out elevation.
    """
    min_y = np.min(y)
    max_y = np.max(y)
    if max_depth is not None:
        upper_depth = min_y + max_depth
        max_depth = upper_depth if upper_depth < max_y else max_y
    else:
        max_depth = max_y

    depth = np.arange(min_y, max_depth, depth_increment)
    areas = np.maximum(depth[:, np.newaxis] - y, 0)
    cross_sections = np.trapz(areas, x, axis=1)
    perimeters = np.array([compute_wetted_perimeter(x, y, d) for d in depth])

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(perimeters > 0, cross_sections / perimeters, 0)

    if smooth:
        ratio = smooth_ratio(ratio, window_length, polyorder)

    # Scale the depth data
    scaler = StandardScaler()
    depth_scaled = scaler.fit_transform(depth.reshape(-1, 1)).flatten()

    try:
        poly_coeffs = np.polyfit(depth_scaled, ratio, polynomial_order)
        poly_fit = np.polyval(poly_coeffs, depth_scaled)
    except np.RankWarning as e:
        print(f"Polyfit RankWarning: {e}")
        return None

    # First and second derivatives
    first_derivative = np.gradient(poly_fit, depth)
    second_derivative = np.gradient(first_derivative, depth)

    # Identify leveling out point where second derivative approaches zero
    inflection_indices = np.where(np.abs(second_derivative) < 1)[0]

    true_depth = depth - np.min(depth)
    channel_depth_threshold = np.percentile(true_depth, percentile)

    filtered_inflection_indices = inflection_indices[true_depth[inflection_indices] > channel_depth_threshold]
    print(f"Max depth considered: {np.max(true_depth)}")

    if len(filtered_inflection_indices) > 1:
        leveling_out_elevation = depth[filtered_inflection_indices[1]]
    elif len(filtered_inflection_indices) == 1:
        leveling_out_elevation = depth[filtered_inflection_indices[0]]
    else:
        leveling_out_elevation = None

    return leveling_out_elevation


def plot_cross_section_area_to_wetted_perimeter_ratio(x, y, idx='', depth_increment=0.1, fig_output_path='', 
                                                      polynomial_order=4, smooth=False, window_length=5, polyorder=4, 
                                                      print_output=True, second_derivative_num=None, percentile=20, max_depth=None):
    """
    Vectorized plotting of cross-sectional area to wetted perimeter ratio.

    Parameters:
        x (np.ndarray): Distances along the line.
        y (np.ndarray): Raster values at sampled points.
        idx (str): Identifier for the current line (used in filenames).
        depth_increment (float): Increment step for depth.
        fig_output_path (str): Directory to save the figure.
        polynomial_order (int): Order of the polynomial fit.
        smooth (bool): Whether to smooth the ratio data.
        window_length (int): Window length for smoothing.
        polyorder (int): Polynomial order for smoothing.
        print_output (bool): Whether to generate and save the plot.
        second_derivative_num (int, optional): Specific parameter for leveling point determination.
        percentile (float): Percentile for channel depth threshold.
        max_depth (float, optional): Maximum depth to consider.

    Returns:
        leveling_out_elevation (float or None): The determined leveling out elevation.
    """
    leveling_out_elevation = None

    if percentile is not None:
        leveling_out_elevation = find_leveling_point_depth(x, y, depth_increment, polynomial_order, 
                                                           smooth, window_length, polyorder, percentile, max_depth)
    else:
        raise ValueError("Percentile must be specified for leveling out point determination.")

    depth = np.arange(np.min(y), np.max(y), depth_increment)
    if max_depth is not None:
        depth = depth[depth <= (np.min(y) + max_depth)]

    areas = np.maximum(depth[:, np.newaxis] - y, 0)
    cross_sections = np.trapz(areas, x, axis=1)
    perimeters = np.array([compute_wetted_perimeter(x, y, d) for d in depth])

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(perimeters > 0, cross_sections / (perimeters ** 2), 0)

    if smooth:
        ratio = smooth_ratio(ratio, window_length, polyorder)

    poly_coeffs = np.polyfit(depth, ratio, polynomial_order)
    poly_fit = np.polyval(poly_coeffs, depth)

    # First and second derivatives
    first_derivative = np.gradient(poly_fit, depth)
    second_derivative = np.gradient(first_derivative, depth)

    if print_output:
        plt.figure(figsize=(10, 6))
        plt.plot(depth, ratio, marker='o', linestyle='-', label='Data')
        plt.plot(depth, poly_fit, 'g--', label=f'{polynomial_order}th Degree Polynomial Fit')
        plt.plot(depth, second_derivative, 'b--', label='Second Derivative')
        if leveling_out_elevation is not None:
            plt.axvline(x=leveling_out_elevation, color='red', linestyle='--', label='Leveling-Out Point')
        plt.xlabel('Depth (m)')
        plt.ylabel('Cross-Sectional Area / Wetted Perimeter ** 2')
        plt.title(f'Cross-Sectional Area / Wetted Perimeter vs. Depth (Index: {idx})')
        plt.legend()
        plt.grid(True)

        fig_save_path = os.path.join(fig_output_path, "Plots", f'{idx}_cross_section_area_to_wetted_perimeter_ratio.png')
        os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
        plt.savefig(fig_save_path)
        plt.close()

    return leveling_out_elevation


def main(gpkg_path, raster_path, output_folder, output_gpkg_path=None, centerline_gpkg=None, 
         second_derivative_num=None, percentile=20, print_output=True, max_depth=None):
    """
    Vectorized main function to process GeoPackage lines, compute hydraulic radii, determine leveling-out elevations,
    and generate output GeoPackage.

    Parameters:
        gpkg_path (str): Path to the input GeoPackage containing perpendicular lines.
        raster_path (str): Path to the raster file (e.g., DEM).
        output_folder (str): Directory to save outputs.
        output_gpkg_path (str, optional): Path to save the output GeoPackage polygon.
        centerline_gpkg (str, optional): Path to the GeoPackage containing the centerline.
        second_derivative_num (int, optional): Specific parameter for leveling point determination.
        percentile (float): Percentile for channel depth threshold.
        print_output (bool): Whether to generate and save plots.
        max_depth (float, optional): Maximum depth to limit elevation selection.

    Returns:
        json_path (str): Path to the saved JSON file containing outputs.
    """
    gdf = gpd.read_file(gpkg_path)
    centerline_gdf = gpd.read_file(centerline_gpkg)
    centerline = centerline_gdf.geometry.iloc[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder at: {output_folder}")

    if output_gpkg_path and os.path.exists(output_gpkg_path):
        os.remove(output_gpkg_path)
        print(f"Removed existing GeoPackage at: {output_gpkg_path}")

    left_points = []
    right_points = []
    flow_depths = {}  # Dictionary to store leveling_out_elevation for each line

    with rasterio.open(raster_path) as raster:
        nodata_value = raster.nodata
        total_lines = len(gdf)
        print(f"Total lines to process: {total_lines}")

        # Initialize tqdm progress bar
        for idx, row in tqdm(gdf.iterrows(), total=total_lines, desc="Processing lines"):
            line = row.geometry
            if isinstance(line, (LineString, MultiLineString)):
                print(f'Processing line {idx + 1} of {total_lines}')
                valid_distances, valid_raster_values, valid_points = sample_raster_along_line(
                    line, raster, nodata_value=nodata_value
                )

                if not valid_distances or not valid_points:
                    print(f"Skipping line {idx} due to all NoData values.")
                    continue

                leveling_out_elevation = plot_cross_section_area_to_wetted_perimeter_ratio(
                    x=np.array(valid_distances),
                    y=np.array(valid_raster_values),
                    idx=idx,
                    fig_output_path=output_folder,
                    print_output=print_output,
                    second_derivative_num=second_derivative_num,
                    percentile=percentile,
                    max_depth=max_depth  # Pass the max_depth parameter
                )

                if leveling_out_elevation is not None:
                    flow_depths[idx] = leveling_out_elevation  # Save to flow_depths dictionary

                    sides = determine_side_of_centerline(valid_points, centerline)

                    left_side_indices = np.where(sides < 0)[0]
                    right_side_indices = np.where(sides > 0)[0]

                    if left_side_indices.size > 0 and right_side_indices.size > 0:
                        left_side_values = np.array(valid_raster_values)[left_side_indices]
                        right_side_values = np.array(valid_raster_values)[right_side_indices]

                        closest_left_idx = left_side_indices[np.argmin(np.abs(left_side_values - leveling_out_elevation))]
                        closest_right_idx = right_side_indices[np.argmin(np.abs(right_side_values - leveling_out_elevation))]

                        closest_left_point = valid_points[closest_left_idx]
                        closest_right_point = valid_points[closest_right_idx]

                        left_points.append(closest_left_point)
                        right_points.append(closest_right_point)

                        print(f"Leveling out point on left side for line index {idx}: {closest_left_point} at elevation {leveling_out_elevation}")
                        print(f"Leveling out point on right side for line index {idx}: {closest_right_point} at elevation {leveling_out_elevation}")
            else:
                geometry_type = line.geom_type
                print(f'Skipping non-LineString geometry at index {idx} with geometry type: {geometry_type}')

    # Save flow_depths to a JSON file
    json_path = os.path.join(output_folder, 'flow_depths.json')
    try:
        with open(json_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            flow_depths_serializable = {str(k): float(v) for k, v in flow_depths.items()}
            json.dump(flow_depths_serializable, f, indent=4)
        print(f"Flow depths saved to JSON file at: {json_path}")
    except Exception as e:
        print(f"Failed to save flow depths to JSON file: {e}")

    if left_points and right_points:
        # Ensure both lists have the same length for polygon creation
        min_length = min(len(left_points), len(right_points))
        left_trimmed = left_points[:min_length]
        right_trimmed = right_points[:min_length]

        polygon_coords = left_trimmed + right_trimmed[::-1] + [left_trimmed[0]]
        polygon = Polygon([(point.x, point.y) for point in polygon_coords])

        polygon_gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs=gdf.crs)
        polygon_gdf.to_file(output_gpkg_path, driver="GPKG")
        print(f"Polygon GeoPackage created at: {output_gpkg_path}")
    else:
        print("No left and/or right points were collected. Polygon GeoPackage was not created.")

    return json_path  # Return the path to the JSON file


"""
Uncomment to run a Monte Carlo on a single watershed with depth percentile method
"""

if __name__ == "__main__":
    perpendiculars_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Valley_CL_perpendiculars_100m.gpkg"
    raster_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Terrain\WBT_Outputs\filled_dem.tif"
    centerline_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Valley_CL.gpkg"

    # Define output paths
    from datetime import datetime
    output_folder = os.path.join(
        r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\ATD_Streams\Percentile", 
        f"{datetime.now().strftime('%Y%m%d_%Hh%Mm')}"
    )
    os.makedirs(output_folder, exist_ok=True)

    output_gpkg_name = f"Valley_Footprint_6th_Per_0.01depth.gpkg"
    output_gpkg_path = os.path.join(output_folder, output_gpkg_name)

    # Define maximum depth (e.g., 10 meters)
    max_depth = 2.5  # Set your desired maximum depth here

    json_path = main(
        gpkg_path=perpendiculars_path,
        raster_path=raster_path,
        output_folder=output_folder,
        output_gpkg_path=output_gpkg_path, 
        centerline_gpkg=centerline_path, 
        percentile=6,
        print_output=True,
        max_depth=max_depth  # Pass the max_depth parameter
    )

    from call_plot_cross_sections import run_cross_section_plotting
    run_cross_section_plotting(perpendiculars_path=perpendiculars_path, dem_raster=raster_path, json_file=json_path)

"""
Uncomment to run multiple watersheds
All watersheds must be present in perpendiculars_dir, raster_dir, and centerlines_dir and start with {watershed_prefix}
e.g. MM_perpendiculars.gpkg, MM_raster.tif, MM_centerline.gpkg
"""
# if __name__ == "__main__":
#     perpendiculars_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Perpendiculars_120m"
#     raster_dir = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\Watershed_Clipped"
#     centerlines_dir = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\Centerlines"
#     output_folder = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\120m_perp_Percentile_Testing"
    
#     perpendiculars_paths = [os.path.join(perpendiculars_dir, f) for f in os.listdir(perpendiculars_dir) if f.endswith('.gpkg')]
#     raster_paths = [os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith('.tif')]
#     centerline_paths = [os.path.join(centerlines_dir, f) for f in os.listdir(centerlines_dir) if f.endswith('.gpkg')]
    
#     for perpendiculars_path, raster_path, centerline_gpkg in zip(perpendiculars_paths, raster_paths, centerline_paths):
#         watershed_name = os.path.basename(perpendiculars_path).split('_')[0]
        
#         percentile_list = np.arange(2, 30, 2)
#         print(f"Processing watershed: {watershed_name}")
#         for percentile in percentile_list:
#             output_gpkg_name = f"{watershed_name}_Valley_Footprint_{percentile}_Percentile.gpkg"
#             output_gpkg_path = os.path.join(output_folder, output_gpkg_name)
#             if os.path.exists(output_gpkg_path):
#                 print(f"Output file already exists at: {output_gpkg_path}")
#                 continue
        
#             main(
#                 gpkg_path=perpendiculars_path,
#                 raster_path=raster_path,
#                 output_folder=output_folder,
#                 output_gpkg_path=output_gpkg_path, 
#                 centerline_gpkg=centerline_gpkg, 
#                 percentile=percentile,
#                 print_output=True
#             )
