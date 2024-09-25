import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import geopandas as gpd
from rasterio.merge import merge
import os
import logging
import pandas as pd  # Added import for pandas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("depth_leveling.log"),
        logging.StreamHandler()
    ]
)

def extract_and_clip_selected_raster_by_segment(shapefile, flow_raster_dict, selected_flow_dict, output_dir):
    """
    Clips the selected raster based on the flow values where depth levels out for each segment.

    Parameters:
        shapefile (str or gpd.GeoDataFrame): Path to the shapefile or a GeoDataFrame.
        flow_raster_dict (dict): Dictionary mapping flow values to raster file paths.
        selected_flow_dict (dict): Dictionary mapping segment indices to selected flow values.
        output_dir (str): Directory to save the clipped rasters.

    Returns:
        list: List of paths to the clipped raster files.
    """
    if isinstance(shapefile, str):
        segments_gdf = gpd.read_file(shapefile)
    elif isinstance(shapefile, gpd.GeoDataFrame):
        segments_gdf = shapefile
    else:
        raise ValueError("shapefile must be a path to a shapefile or a GeoDataFrame.")
    
    stitched_rasters = []

    for idx, segment in segments_gdf.geometry.iteritems():
        # Get the selected flow value for this segment
        selected_flow_value = selected_flow_dict.get(idx)
        
        if selected_flow_value is None:
            logging.warning(f"No selected flow value for segment {idx}. Skipping.")
            continue
        
        # Get the corresponding raster path for the selected flow value
        raster_path = flow_raster_dict.get(selected_flow_value)
        
        if raster_path:
            try:
                with rasterio.open(raster_path) as src:
                    # Mask raster with polygon
                    out_image, out_transform = rasterio.mask.mask(src, [segment], crop=True)
                    out_meta = src.meta.copy()
                    
                    # Update metadata to reflect the new dimensions and transform
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })
                    
                    # Save the clipped raster for the current segment and flow value
                    clipped_raster_path = os.path.join(
                        output_dir, 
                        f"segment_{idx}_flow_{selected_flow_value}.tif"
                    )
                    with rasterio.open(clipped_raster_path, "w", **out_meta) as dest:
                        dest.write(out_image)
                    
                    # Append to the list of clipped rasters for merging later
                    stitched_rasters.append(clipped_raster_path)
            except Exception as e:
                logging.error(f"Error processing segment {idx} with flow {selected_flow_value}: {e}")
        else:
            logging.warning(f"Flow value {selected_flow_value} not found in flow_raster_dict for segment {idx}")

    return stitched_rasters

def stitch_rasters(raster_paths, output_stitched_path):
    """
    Merges multiple raster files into a single mosaic.

    Parameters:
        raster_paths (list): List of paths to raster files to be merged.
        output_stitched_path (str): Path to save the merged raster.
    """
    src_files_to_mosaic = []
    try:
        # Open all clipped raster files for merging using context managers
        for path in raster_paths:
            src = rasterio.open(path)
            src_files_to_mosaic.append(src)
        
        if not src_files_to_mosaic:
            logging.warning("No raster files to merge.")
            return
        
        # Merge the rasters
        mosaic, out_trans = merge(src_files_to_mosaic)
        logging.info("Rasters merged successfully.")
        
        # Copy metadata from one of the source files
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        
        # Write the mosaic to a new file
        with rasterio.open(output_stitched_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        logging.info(f"Merged raster saved: {output_stitched_path}")
        
    except Exception as e:
        logging.error(f"Error stitching rasters: {e}")
    finally:
        # Ensure all sources are closed
        for src in src_files_to_mosaic:
            src.close()

def extract_max_depths_by_segment(shapefile, flow_raster_dict, percentile=90):
    """
    Extracts the specified percentile depth for each segment and flow value.

    Parameters:
        shapefile (str or gpd.GeoDataFrame): Path to the shapefile or a GeoDataFrame.
        flow_raster_dict (dict): Dictionary mapping flow values to raster file paths.
        percentile (float): Percentile to compute (e.g., 90 for 90th percentile).

    Returns:
        dict: Nested dictionary mapping segment indices to flow values and their corresponding percentile depths.
    """
    if isinstance(shapefile, str):
        segments_gdf = gpd.read_file(shapefile)
    elif isinstance(shapefile, gpd.GeoDataFrame):
        segments_gdf = shapefile
    else:
        raise ValueError("shapefile must be a path to a shapefile or a GeoDataFrame.")

    segment_depths = {}

    for idx, segment in segments_gdf.geometry.iteritems():
        segment_depths[idx] = {}
        for flow_value, raster_path in flow_raster_dict.items():
            try:
                with rasterio.open(raster_path) as src:
                    # Mask raster with polygon
                    out_image, _ = rasterio.mask.mask(src, [segment], crop=True)
                    # Assuming single band
                    data = out_image[0]
                    
                    # Handle masked arrays and nodata values
                    if src.nodata is not None:
                        valid_data = data[data != src.nodata]
                    else:
                        valid_data = data.flatten()
                    
                    valid_data = valid_data[~np.isnan(valid_data)]
                    
                    if valid_data.size > 0:
                        max_depth = np.percentile(valid_data, percentile)
                        segment_depths[idx][flow_value] = max_depth
                        logging.debug(f"Segment {idx}, Flow {flow_value}: {percentile}th percentile depth = {max_depth}")
                    else:
                        segment_depths[idx][flow_value] = np.nan
                        logging.warning(f"Segment {idx}, Flow {flow_value}: No valid data found.")
            except Exception as e:
                segment_depths[idx][flow_value] = np.nan
                logging.error(f"Error extracting depth for segment {idx}, flow {flow_value}: {e}")

    return segment_depths

def smooth_data(data, window_length=5, polyorder=2, smooth=False):
    """
    Applies Savitzky-Golay filter for data smoothing.

    Parameters:
        data (array-like): Input data to smooth.
        window_length (int): Length of the filter window (must be odd).
        polyorder (int): Order of the polynomial used to fit the samples.
        smooth (bool): Flag to apply smoothing.

    Returns:
        np.ndarray: Smoothed data.
    """
    try:
        if not smooth:
            return data
        if window_length >= len(data):
            window_length = len(data) - 1 if len(data) % 2 == 0 else len(data)
            if window_length < polyorder + 2:
                window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1  # window_length must be odd
        smoothed = savgol_filter(data, window_length=window_length, polyorder=polyorder)
        return smoothed
    except Exception as e:
        logging.error(f"Error smoothing data: {e}")
        return data  # Return original data if smoothing fails

def fit_polynomial(x, y, degree=3):
    """
    Fits a polynomial to the data.

    Parameters:
        x (array-like): Independent variable.
        y (array-like): Dependent variable.
        degree (int): Degree of the polynomial.

    Returns:
        np.poly1d: Fitted polynomial.
    """
    try:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        return poly
    except Exception as e:
        logging.error(f"Error fitting polynomial: {e}")
        return np.poly1d([0])

def find_leveling_out_point(flow_values, smoothed_depths):
    """
    Uses first and second derivatives to find the leveling out point.

    Parameters:
        flow_values (list or np.ndarray): Sorted flow values.
        smoothed_depths (np.ndarray): Smoothed depth values.

    Returns:
        int: Index of the leveling out point.
    """
    try:
        first_derivative = np.gradient(smoothed_depths, flow_values)
        second_derivative = np.gradient(first_derivative, flow_values)

        zero_crossings = np.where(np.diff(np.sign(second_derivative)))[0]
        
        # Find the index where the second derivative changes sign
        if len(zero_crossings) >= 1:
            leveling_out_index = zero_crossings[0]
            logging.debug(f"Zero crossing found at index {leveling_out_index}")
        else:
            # If no inflection points are found, use the largest difference in the second derivative
            leveling_out_index = np.argmax(np.abs(np.diff(second_derivative)))
            logging.debug(f"No zero crossing found. Using max diff index {leveling_out_index}")
             
        return leveling_out_index
    except Exception as e:
        logging.error(f"Error finding leveling out point: {e}")
        return len(flow_values) - 1  # Default to the last index

def plot_flow_vs_depth_by_segment(segment_depths, percentile, output_dir):
    """
    Plots flow value vs depth for each segment and identifies the leveling out point.

    Parameters:
        segment_depths (dict): Nested dictionary mapping segment indices to flow values and depths.
        percentile (float): The percentile used for depth calculation.
        output_dir (str): Directory to save the plot images.

    Returns:
        dict: Dictionary mapping segment indices to flow values where depth levels out.
    """
    flow_value_dict = {}

    for segment_idx, depths in segment_depths.items():
        # Sort the flow values and corresponding depths
        sorted_flow_values = sorted(depths.keys())
        sorted_max_depths = [depths[flow] for flow in sorted_flow_values]
        
        # Convert to numpy arrays for processing
        sorted_flow_values_np = np.array(sorted_flow_values)
        sorted_max_depths_np = np.array(sorted_max_depths)
        
        # Handle segments with all NaN depths
        if np.all(np.isnan(sorted_max_depths_np)):
            logging.warning(f"Segment {segment_idx}: All depth values are NaN. Skipping.")
            continue

        # Replace NaNs with nearest valid value or interpolate
        if np.isnan(sorted_max_depths_np).any():
            # Simple forward fill, can be replaced with more sophisticated methods if needed
            nan_indices = np.isnan(sorted_max_depths_np)
            sorted_max_depths_np[nan_indices] = np.interp(
                np.flatnonzero(nan_indices),
                np.flatnonzero(~nan_indices),
                sorted_max_depths_np[~nan_indices]
            )
        
        # Smooth the max depths
        smoothed_depths = smooth_data(sorted_max_depths_np, smooth=True)

        # Find leveling out point
        leveling_out_index = find_leveling_out_point(sorted_flow_values_np, smoothed_depths)
        
        # Fit a polynomial to the smoothed data
        poly = fit_polynomial(sorted_flow_values_np, smoothed_depths)
        fitted_values = poly(sorted_flow_values_np)
        
        # Plot original and smoothed data
        plt.figure(figsize=(10, 6))
        plt.plot(
            sorted_flow_values_np, sorted_max_depths_np, 
            marker='o', label=f'Original Depths at {percentile}th percentile'
        )
        plt.plot(
            sorted_flow_values_np, smoothed_depths, 
            label='Smoothed Depths', color='orange'
        )
        plt.plot(
            sorted_flow_values_np, fitted_values, 
            label='Fitted Polynomial', color='green'
        )
        if leveling_out_index < len(sorted_flow_values_np):
            plt.axvline(
                sorted_flow_values_np[leveling_out_index], 
                color='red', linestyle='--', 
                label=f'Leveling out at Flow = {sorted_flow_values_np[leveling_out_index]}'
            )
            flow_value_dict[segment_idx] = sorted_flow_values_np[leveling_out_index]
        else:
            logging.warning(f"Segment {segment_idx}: Leveling out index out of range.")
            flow_value_dict[segment_idx] = sorted_flow_values_np[-1]
        
        plt.title(f'Flow Value vs Depth at {percentile}th percentile for Segment {segment_idx}')
        plt.xlabel('Flow Value')
        plt.ylabel(f'Depth at {percentile}th percentile')
        plt.grid(True)
        plt.legend()
        
        # Save the plot instead of showing
        plot_path = os.path.join(output_dir, f'segment_{segment_idx}_flow_depth_plot.png')
        plt.savefig(plot_path)
        plt.close()

    return flow_value_dict

def compute_window_average_flow_depth(flow_value_dict, window_size=10):
    """
    Computes the window average of flow depths using a rolling window.

    Parameters:
        flow_value_dict (dict): Dictionary mapping segment indices to flow values.
        window_size (int): Number of segments to include in the moving window.

    Returns:
        dict: Dictionary mapping segment indices to window averaged flow values.
    """
    # Sort the segments by their index to ensure correct order
    sorted_indices = sorted(flow_value_dict.keys())
    flow_series = pd.Series([flow_value_dict[idx] for idx in sorted_indices], index=sorted_indices)
    
    # Apply rolling window with the specified window size
    window_avg_series = flow_series.rolling(window=window_size, min_periods=1, center=True).mean()
    
    # Convert back to dictionary
    window_average_flow_value_dict = window_avg_series.to_dict()
    
    return window_average_flow_value_dict

def round_flow_values(flow_value_dict, available_flow_values):
    """
    Rounds each flow value to the nearest available flow value in flow_raster_dict.

    Parameters:
        flow_value_dict (dict): Dictionary mapping segment indices to flow values.
        available_flow_values (list or np.ndarray): List of available flow values from flow_raster_dict.

    Returns:
        dict: Dictionary mapping segment indices to rounded flow values.
    """
    # Sort the segments by their index to ensure correct order
    sorted_indices = sorted(flow_value_dict.keys())
    flow_series = pd.Series([flow_value_dict[idx] for idx in sorted_indices], index=sorted_indices)
    
    # Define a function to find the nearest flow value
    def find_nearest_flow(flow):
        if np.isnan(flow):
            return np.nan
        return min(available_flow_values, key=lambda x: abs(x - flow))
    
    # Apply the function to each flow value
    rounded_flow_series = flow_series.apply(find_nearest_flow)
    
    # Convert back to dictionary
    rounded_flow_value_dict = rounded_flow_series.to_dict()
    
    return rounded_flow_value_dict

def main():
    # Update input paths
    segmented_polygon_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Inputs\Segmented_Buffered_Valley_5m.gpkg"
    # Alternative path (uncomment if needed)
    # segmented_polygon_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Segmented Valleys\Bennett_Segmented_Valleys.gpkg"

    flow_raster_dict = {
        0.25: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_0o25cms.tif",
        0.5: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_0o5cms.tif",
        1: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_1cms.tif",
        2: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_2cms.tif",
        3: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_3cms.tif",
        4: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_4cms.tif",
        5: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_5cms.tif",
        6: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_6cms.tif",
        7: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_7cms.tif",
        8: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_8cms.tif",
        9: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_9cms.tif",
        10: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters\ME\ME_10cms.tif",
    }

    scratch_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Clipped_Rasters"
    output_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Results\Inflection Point Windowed"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    percentile_array = [95]  # You can add more percentiles if needed
    window_size = 10  # Define the window size for averaging

    for percentile in percentile_array:
        logging.info(f"Processing {percentile}th percentile...")
        stitched_raster_title = f"Stitched_Raster_{percentile}th_per_5m.tif"
        stitched_raster_output_path = os.path.join(output_dir, stitched_raster_title)

        try:
            # Extract max depths for each segment
            segmented_polygons_gdf = gpd.read_file(segmented_polygon_path)
            max_depths = extract_max_depths_by_segment(segmented_polygons_gdf, flow_raster_dict, percentile)

            # Plot and get the flow values where depth levels out for each segment
            flow_value_dict = plot_flow_vs_depth_by_segment(max_depths, percentile, output_dir)

            # Compute window average flow depth with a window of 10 segments
            window_average_flow_value_dict = compute_window_average_flow_depth(flow_value_dict, window_size=window_size)
            logging.info(f"Computed window average flow depth with window size {window_size} segments.")

            # Round the window averaged flow depths to the nearest available flow value
            available_flow_values = sorted(flow_raster_dict.keys())
            rounded_flow_value_dict = round_flow_values(window_average_flow_value_dict, available_flow_values)
            logging.info("Rounded window-averaged flow depths to the nearest available flow values.")

            # Optional: Save the window averaged flow values to a CSV for reference
            window_avg_df = pd.DataFrame({
                'Segment': list(window_average_flow_value_dict.keys()),
                'Window_Avg_Flow_Value': list(window_average_flow_value_dict.values()),
                'Rounded_Flow_Value': [rounded_flow_value_dict[idx] for idx in sorted(rounded_flow_value_dict.keys())]
            })
            window_avg_csv_path = os.path.join(output_dir, f'window_average_flow_depth_{percentile}th_percentile.csv')
            window_avg_df.to_csv(window_avg_csv_path, index=False)
            logging.info(f"Window averaged and rounded flow depths saved to {window_avg_csv_path}")

            # Clip only the selected raster for each segment based on rounded_flow_value_dict
            clipped_rasters = extract_and_clip_selected_raster_by_segment(
                segmented_polygons_gdf, flow_raster_dict, rounded_flow_value_dict, scratch_dir
            )

            if not clipped_rasters:
                logging.warning("No rasters were clipped. Skipping stitching.")
                continue

            # Stitch clipped rasters into a single output
            stitch_rasters(clipped_rasters, stitched_raster_output_path)

            # Optionally, remove clipped rasters to save space
            for path in clipped_rasters:
                os.remove(path)
                logging.debug(f"Removed temporary clipped raster: {path}")

            for segment, flow in rounded_flow_value_dict.items():
                logging.info(f"Segment {segment} window-averaged and rounded inflection at flow value {flow}")

        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
