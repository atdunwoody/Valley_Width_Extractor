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
import json  # Added import for JSON operations

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

def smooth_data(data, window_length=5, polyorder=2, smooth = False):
    """
    Applies Savitzky-Golay filter for data smoothing.

    Parameters:
        data (array-like): Input data to smooth.
        window_length (int): Length of the filter window (must be odd).
        polyorder (int): Order of the polynomial used to fit the samples.

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

import matplotlib.pyplot as plt
import numpy as np
import os
import logging

def plot_flow_vs_depth_by_segment(segment_depths, percentile, output_dir):
    """
    Plots flow value vs depth for each segment along with their first and second derivatives,
    and identifies the leveling out point where the depth's rate of increase decreases significantly.

    Parameters:
        segment_depths (dict): Nested dictionary mapping segment indices to flow values and depths.
        percentile (float): The percentile used for depth calculation.
        output_dir (str): Directory to save the plot images.

    Returns:
        dict: Dictionary mapping segment indices to flow values where depth levels out.
    """
    flow_value_dict = {}
    plots_dir = os.path.join(output_dir, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)

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
            # Simple interpolation to fill NaNs
            nan_indices = np.isnan(sorted_max_depths_np)
            valid_indices = ~nan_indices
            if valid_indices.sum() == 0:
                logging.warning(f"Segment {segment_idx}: No valid depth data available after NaN removal. Skipping.")
                continue
            sorted_max_depths_np[nan_indices] = np.interp(
                np.flatnonzero(nan_indices),
                np.flatnonzero(valid_indices),
                sorted_max_depths_np[valid_indices]
            )
        
        # Smooth the max depths
        smoothed_depths = smooth_data(sorted_max_depths_np, smooth=True)
        
        # Find leveling out point
        leveling_out_index = find_leveling_out_point(sorted_flow_values_np, smoothed_depths)
        
        # Fit a polynomial to the smoothed data
        poly = fit_polynomial(sorted_flow_values_np, smoothed_depths)
        fitted_values = poly(sorted_flow_values_np)
        
        # Calculate derivatives
        first_derivative = np.gradient(smoothed_depths, sorted_flow_values_np)
        second_derivative = np.gradient(first_derivative, sorted_flow_values_np)
        
        # Initialize the plot with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # ---- Top Subplot: Flow vs Depth ----
        ax1.plot(
            sorted_flow_values_np, sorted_max_depths_np, 
            marker='o', linestyle='-', label=f'Original Depths ({percentile}th percentile)'
        )
        ax1.plot(
            sorted_flow_values_np, smoothed_depths, 
            linestyle='-', label='Smoothed Depths', color='orange'
        )
        ax1.plot(
            sorted_flow_values_np, fitted_values, 
            linestyle='--', label='Fitted Polynomial', color='green'
        )
        if 0 <= leveling_out_index < len(sorted_flow_values_np):
            leveling_out_flow = sorted_flow_values_np[leveling_out_index]
            ax1.axvline(
                leveling_out_flow, 
                color='red', linestyle='--', 
                label=f'Leveling Out at Flow = {leveling_out_flow:.2f}'
            )
            flow_value_dict[segment_idx] = leveling_out_flow
        else:
            logging.warning(f"Segment {segment_idx}: Leveling out index out of range.")
            flow_value_dict[segment_idx] = sorted_flow_values_np[-1]
        
        ax1.set_title(f'Segment {segment_idx}: Flow vs Depth ({percentile}th Percentile)')
        ax1.set_ylabel('Depth')
        ax1.grid(True)
        ax1.legend()
        
        # ---- Bottom Subplot: Derivatives ----
        ax2.plot(sorted_flow_values_np, first_derivative, label='First Derivative', color='blue')
        ax2.plot(sorted_flow_values_np, second_derivative, label='Second Derivative', color='brown')
        if 0 <= leveling_out_index < len(sorted_flow_values_np):
            ax2.axvline(
                leveling_out_flow, 
                color='red', linestyle='--', 
                label=f'Leveling Out at Flow = {leveling_out_flow:.2f}'
            )
        ax2.set_title('Derivatives of Flow vs Depth')
        ax2.set_xlabel('Flow Value')
        ax2.set_ylabel('Derivative')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the combined plot
        plot_path = os.path.join(plots_dir, f'segment_{segment_idx}_flow_depth_derivatives_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        logging.info(f"Plot saved: {plot_path}")

    return flow_value_dict

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
    output_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Results\Inflection Point"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    percentile_array = [95]  # You can add more percentiles if needed

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

            # Save the flow_value_dict to a JSON file
            flow_values_output_path = os.path.join(output_dir, "flow_values.json")
            # Convert dictionary keys to strings for JSON compatibility
            flow_value_dict_str_keys = {str(k): v for k, v in flow_value_dict.items()}
            with open(flow_values_output_path, 'w') as f:
                json.dump(flow_value_dict_str_keys, f, indent=4)
            logging.info(f"Flow values saved to: {flow_values_output_path}")

            # Clip only the selected raster for each segment based on flow_value_dict
            # Note: Since the keys in flow_value_dict are strings now, ensure that the extract_and_clip_selected_raster_by_segment
            # function can handle string keys if necessary. If it expects integer keys, you may need to convert them back.
            # For now, assuming keys are integers.
            # If not, modify the keys accordingly.
            # Convert keys back to integers if necessary
            flow_value_dict_int_keys = {int(k): v for k, v in flow_value_dict_str_keys.items()}
            clipped_rasters = extract_and_clip_selected_raster_by_segment(
                segmented_polygons_gdf, flow_raster_dict, flow_value_dict_int_keys, scratch_dir
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

            for segment, flow in flow_value_dict.items():
                logging.info(f"Segment {segment} inflection at flow value {flow}")

        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
