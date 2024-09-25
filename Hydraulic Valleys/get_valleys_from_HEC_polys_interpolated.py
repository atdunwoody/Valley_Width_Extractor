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
import json
from shapely.geometry import LineString, Point, MultiLineString
from shapely.affinity import rotate, translate
import pandas as pd
import fiona
from osgeo import ogr
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("depth_leveling.log"),
        logging.StreamHandler()
    ]
)

def extract_lengths_from_flow_dict(shapefile, flow_dict, percentile=90, buffer_distance=10):
    """
    Extracts lengths from input lines in flow_dict for each segment, buffering each segment by a small amount.

    Parameters:
        shapefile (str or gpd.GeoDataFrame): Path to the shapefile or a GeoDataFrame.
        flow_dict (dict): Dictionary mapping flow values to GeoPackage paths containing input lines.
        percentile (float): Percentile to compute (e.g., 90 for 90th percentile).
        buffer_distance (float): Distance to buffer each segment geometry before intersection (in CRS units).

    Returns:
        dict: Nested dictionary mapping segment indices to flow values and their corresponding lengths.
    """
    if isinstance(shapefile, str):
        segments_gdf = gpd.read_file(shapefile)
    elif isinstance(shapefile, gpd.GeoDataFrame):
        segments_gdf = shapefile
    else:
        raise ValueError("shapefile must be a path to a shapefile or a GeoDataFrame.")

    segment_lengths = {}

    for idx, segment in segments_gdf.geometry.iteritems():
        segment_lengths[idx] = {}
        try:
            # Buffer the segment by the specified distance
            buffered_segment = segment.buffer(buffer_distance)
            logging.debug(f"Segment {idx}: Buffered by {buffer_distance} units.")
        except Exception as e:
            logging.error(f"Error buffering segment {idx}: {e}")
            # If buffering fails, use the original segment
            buffered_segment = segment

        for flow_value, gpkg_path in flow_dict.items():
            try:
                # Read the input lines from the GeoPackage
                input_gdf = gpd.read_file(gpkg_path)

                # Filter lines that intersect the buffered segment
                intersecting_lines = input_gdf[input_gdf.intersects(buffered_segment)]

                if not intersecting_lines.empty:
                    # Extract lengths from geometries
                    lengths = intersecting_lines.geometry.length.values
                    # Compute the desired percentile of lengths
                    desired_length = np.percentile(lengths, percentile)
                    segment_lengths[idx][flow_value] = desired_length
                    logging.debug(f"Segment {idx}, Flow {flow_value}: {percentile}th percentile length = {desired_length}")
                else:
                    segment_lengths[idx][flow_value] = np.nan
                    logging.warning(f"Segment {idx}, Flow {flow_value}: No intersecting input lines found.")
            except Exception as e:
                segment_lengths[idx][flow_value] = np.nan
                logging.error(f"Error extracting length for segment {idx}, flow {flow_value}: {e}")

    return segment_lengths

def smooth_data(data, window_length=5, polyorder=2, smooth=False):
    """
    Applies Savitzky-Golay filter for data smoothing.

    Parameters:
        data (array-like): Input data to smooth.
        window_length (int): Length of the filter window (must be odd).
        polyorder (int): Order of the polynomial used to fit the samples.
        smooth (bool): Whether to apply smoothing.

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

def find_exact_leveling_out_point(flow_values, smoothed_lengths):
    """
    Uses first and second derivatives to find the exact leveling out point.

    Parameters:
        flow_values (np.ndarray): Sorted flow values.
        smoothed_lengths (np.ndarray): Smoothed length values.

    Returns:
        float: Exact flow value where leveling out occurs.
    """
    try:
        first_derivative = np.gradient(smoothed_lengths, flow_values)
        second_derivative = np.gradient(first_derivative, flow_values)

        # Find zero crossing in second derivative for exact leveling out flow
        zero_crossings = np.where(np.diff(np.sign(second_derivative)))[0]

        if len(zero_crossings) >= 1:
            # Linear interpolation to find exact zero crossing
            idx = zero_crossings[0]
            flow1, flow2 = flow_values[idx], flow_values[idx + 1]
            sd1, sd2 = second_derivative[idx], second_derivative[idx + 1]
            # Avoid division by zero
            if sd2 - sd1 != 0:
                exact_flow = flow1 - sd1 * (flow2 - flow1) / (sd2 - sd1)
            else:
                exact_flow = flow1
            logging.debug(f"Zero crossing found between flows {flow1} and {flow2}, exact_flow = {exact_flow}")
        else:
            # If no inflection point, use the flow with the maximum second derivative magnitude
            exact_flow = flow_values[np.argmax(np.abs(second_derivative))]
            logging.debug(f"No zero crossing found. Using flow with max second derivative: {exact_flow}")

        return exact_flow
    except Exception as e:
        logging.error(f"Error finding leveling out point: {e}")
        return flow_values[-1]  # Default to the last flow value

def interpolate_length(exact_flow, flow_dict):
    """
    Interpolates length based on the exact flow value.

    Parameters:
        exact_flow (float): The exact flow value where leveling out occurs.
        flow_dict (dict): Dictionary mapping flow values to GeoPackage paths containing input lines.

    Returns:
        float: Interpolated length.
    """
    try:
        sorted_flows = sorted(flow_dict.keys())
        if exact_flow <= sorted_flows[0]:
            lower_flow = upper_flow = sorted_flows[0]
        elif exact_flow >= sorted_flows[-1]:
            lower_flow = upper_flow = sorted_flows[-1]
        else:
            lower_flow = max([f for f in sorted_flows if f <= exact_flow])
            upper_flow = min([f for f in sorted_flows if f >= exact_flow])

        if lower_flow == upper_flow:
            # No interpolation needed
            gpkg_path = flow_dict[lower_flow]
            input_gdf = gpd.read_file(gpkg_path)
            # Assuming uniform length across all input lines for simplicity
            length = input_gdf.geometry.length.median()
            logging.debug(f"Exact flow {exact_flow} matches flow {lower_flow}. Using length {length}.")
            return length
        else:
            # Interpolate between lower_flow and upper_flow
            gpkg_lower = flow_dict[lower_flow]
            gpkg_upper = flow_dict[upper_flow]

            input_gdf_lower = gpd.read_file(gpkg_lower)
            input_gdf_upper = gpd.read_file(gpkg_upper)

            # Assuming uniform length across all input lines for simplicity
            length_lower = input_gdf_lower.geometry.length.median()
            length_upper = input_gdf_upper.geometry.length.median()

            # Linear interpolation
            interpolated_length = length_lower + (length_upper - length_lower) * ((exact_flow - lower_flow) / (upper_flow - lower_flow))
            logging.debug(f"Interpolated length for exact flow {exact_flow}: {interpolated_length} (between {length_lower} and {length_upper})")
            return interpolated_length
    except Exception as e:
        logging.error(f"Error interpolating length for flow {exact_flow}: {e}")
        return np.nan

def update_segment_lengths(json_file_path, geopackage_path, output_geopackage_path=None):
    """
    Updates the 'length' attribute of segments in a GeoPackage based on a JSON file.

    Parameters:
    - json_file_path (str): Path to the JSON file containing segment lengths.
    - geopackage_path (str): Path to the input GeoPackage file.
    - layer_name (str): Name of the layer in the GeoPackage to modify.
    - output_geopackage_path (str, optional): Path to save the modified GeoPackage. 
      If None, the input GeoPackage will be overwritten.

    Returns:
    - None
    """

    # Step 1: Load the JSON data
    try:
        with open(json_file_path, 'r') as f:
            segment_lengths = json.load(f)
        logging.info(f"Successfully loaded JSON data from {json_file_path}.")
    except Exception as e:
        raise IOError(f"Error reading JSON file: {e}")

    # Step 2: Load the GeoPackage layer
    try:
        gdf = gpd.read_file(geopackage_path)
        logging.info(f"Successfully loaded GeoPackage '{geopackage_path}'.")
    except Exception as e:
        raise IOError(f"Error reading GeoPackage layer: {e}")

    # Step 3: Identify the Feature ID (FID) column
    # GeoPackages typically have an internal FID, but it's not always exposed as a column.
    # For this function, we assume there's a column that uniquely identifies each segment.
    # Commonly, this could be 'fid', 'id', or similar. Adjust as necessary.
    possible_id_columns = ['fid', 'id', 'FID', 'ID']
    fid_column = None
    for col in possible_id_columns:
        if col in gdf.columns:
            fid_column = col
            break

    if fid_column is None:
        # If no standard ID column is found, use the GeoDataFrame's index as FID
        fid_column = gdf.index.name if gdf.index.name else 'index'
        gdf.reset_index(inplace=True)  # Ensure the index is a column
        logging.info(f"No standard ID column found. Using '{fid_column}' as FID.")

    # Step 4: Update the 'length' field based on the JSON data
    if 'length' not in gdf.columns:
        # If 'length' column doesn't exist, create it
        gdf['length'] = None
        logging.info("No 'length' column found. Creating 'length' column.")

    updated_count = 0
    for idx, row in gdf.iterrows():
        fid = str(row[fid_column])  # Convert to string to match JSON keys
        if fid in segment_lengths:
            new_length = segment_lengths[fid].get('interpolated_length')
            if new_length is not None:
                gdf.at[idx, 'length'] = new_length
                updated_count += 1
        else:
            logging.warning(f"FID {fid} not found in JSON data.")

    logging.info(f"Updated 'length' for {updated_count} segments based on JSON data.")

    # Step 5: Save the updated GeoPackage
    try:
        if output_geopackage_path is None:
            output_geopackage_path = geopackage_path
            overwrite = True
        else:
            overwrite = False

        if overwrite:
            # To overwrite the existing layer in the GeoPackage, we'll create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.gpkg') as tmpfile:
                temp_path = tmpfile.name

            # Save the updated GeoDataFrame to the temporary GeoPackage
            gdf.to_file(temp_path, driver="GPKG")
            logging.info(f"Temporary GeoPackage created at '{temp_path}'.")

            # Replace the original GeoPackage with the updated one
            shutil.copy(temp_path, geopackage_path)
            logging.info(f"GeoPackage '{geopackage_path}' has been updated with new lengths.")

            # Clean up the temporary file
            os.remove(temp_path)
            logging.info(f"Temporary file '{temp_path}' has been removed.")

        else:
            # If not overwriting, save to the specified output path
            gdf.to_file(output_geopackage_path, driver="GPKG")
            logging.info(f"Updated GeoPackage saved to '{output_geopackage_path}'.")

    except Exception as e:
        raise IOError(f"Error writing to GeoPackage: {e}")

    logging.info("Segment lengths update completed successfully.")

def plot_flow_vs_length_by_segment(segment_lengths, percentile, output_dir, plot=True):
    """
    Plots flow value vs length for each segment along with their first and second derivatives,
    and identifies the exact leveling out flow where the length's rate of change decreases significantly.

    Parameters:
        segment_lengths (dict): Nested dictionary mapping segment indices to flow values and lengths.
        percentile (float): The percentile used for length calculation.
        output_dir (str): Directory to save the plot images.
        plot (bool): Whether to generate and save plots.

    Returns:
        dict: Dictionary mapping segment indices to exact flow values where length levels out.
    """
    flow_value_dict = {}
    plots_dir = os.path.join(output_dir, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)

    for segment_idx, lengths in segment_lengths.items():
        # Sort the flow values and corresponding lengths
        sorted_flow_values = sorted(lengths.keys())
        sorted_lengths = [lengths[flow] for flow in sorted_flow_values]

        # Convert to numpy arrays for processing
        sorted_flow_values_np = np.array(sorted_flow_values)
        sorted_lengths_np = np.array(sorted_lengths)

        # Handle segments with all NaN lengths
        if np.all(np.isnan(sorted_lengths_np)):
            logging.warning(f"Segment {segment_idx}: All length values are NaN. Skipping.")
            continue

        # Replace NaNs with nearest valid value or interpolate
        if np.isnan(sorted_lengths_np).any():
            # Simple interpolation to fill NaNs
            nan_indices = np.isnan(sorted_lengths_np)
            valid_indices = ~nan_indices
            if valid_indices.sum() == 0:
                logging.warning(f"Segment {segment_idx}: No valid length data available after NaN removal. Skipping.")
                continue
            sorted_lengths_np[nan_indices] = np.interp(
                np.flatnonzero(nan_indices),
                np.flatnonzero(valid_indices),
                sorted_lengths_np[valid_indices]
            )

        # Smooth the lengths
        smoothed_lengths = smooth_data(sorted_lengths_np, smooth=True)

        # Find exact leveling out flow
        exact_flow = find_exact_leveling_out_point(sorted_flow_values_np, smoothed_lengths)

        # Fit a polynomial to the smoothed data
        poly = fit_polynomial(sorted_flow_values_np, smoothed_lengths)
        fitted_values = poly(sorted_flow_values_np)

        # Calculate derivatives
        first_derivative = np.gradient(smoothed_lengths, sorted_flow_values_np)
        second_derivative = np.gradient(first_derivative, sorted_flow_values_np)
        flow_value_dict[segment_idx] = exact_flow

        if plot:
            # Initialize the plot with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # ---- Top Subplot: Flow vs Length ----
            ax1.plot(
                sorted_flow_values_np, sorted_lengths_np, 
                marker='o', linestyle='-', label=f'Original Lengths ({percentile}th percentile)'
            )
            ax1.plot(
                sorted_flow_values_np, smoothed_lengths, 
                linestyle='-', label='Smoothed Lengths', color='orange'
            )
            ax1.plot(
                sorted_flow_values_np, fitted_values, 
                linestyle='--', label='Fitted Polynomial', color='green'
            )
            ax1.axvline(
                exact_flow, 
                color='red', linestyle='--', 
                label=f'Leveling Out at Flow = {exact_flow:.2f}'
            )

            ax1.set_title(f'Segment {segment_idx}: Flow vs Length ({percentile}th Percentile)')
            ax1.set_ylabel('Length')
            ax1.grid(True)
            ax1.legend()

            # ---- Bottom Subplot: Derivatives ----
            ax2.plot(sorted_flow_values_np, first_derivative, label='First Derivative', color='purple')
            ax2.plot(sorted_flow_values_np, second_derivative, label='Second Derivative', color='brown')
            ax2.axvline(
                exact_flow, 
                color='red', linestyle='--', 
                label=f'Leveling Out at Flow = {exact_flow:.2f}'
            )
            ax2.set_title('Derivatives of Flow vs Length')
            ax2.set_xlabel('Flow Value')
            ax2.set_ylabel('Derivative')
            ax2.grid(True)
            ax2.legend()

            plt.tight_layout()

            # Save the combined plot
            plot_path = os.path.join(plots_dir, f'segment_{segment_idx}_flow_length_derivatives_plot.png')
            plt.savefig(plot_path)
            plt.close()

            logging.info(f"Plot saved: {plot_path}")

    return flow_value_dict

def main():
    # Update input paths
    segmented_polygon_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Inputs\Segmented_Buffered_Valley_5m.gpkg"
    # Alternative path (uncomment if needed)
    # segmented_polygon_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Segmented Valleys\Bennett_Segmented_Valleys.gpkg"

    plot = False
    
    # Path to the GeoPackage to update
    geopackage_to_update = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_0o25cms.gpkg" # Assuming you want to update the same GeoPackage
    output_width_gpkg = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Results\Inflection Point Interpolated\Updated_Segmented_Buffered_Valley_5m.gpkg"
    # Updated flow_dict: Mapping flow values to GeoPackage paths containing input lines
    flow_dict = {
        0.25: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_0o25cms.gpkg",
        0.5:  r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_0o5cms.gpkg",
        1: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_1cms.gpkg",
        2: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_2cms.gpkg",
        3: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_3cms.gpkg",
        4: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_4cms.gpkg",
        5: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_5cms.gpkg",
        6: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_6cms.gpkg",
        7: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_7cms.gpkg",
        8: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_8cms.gpkg",
        9: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_9cms.gpkg",
        10: r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Lines\ME_5m\ME_10cms.gpkg",
    }
    scratch_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Scratch"
    output_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Results\Inflection Point Interpolated"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    percentile_array = [95]  # You can add more percentiles if needed
    for percentile in percentile_array:
        logging.info(f"Processing {percentile}th percentile...")

        try:
            # Extract lengths for each segment
            segmented_polygons_gdf = gpd.read_file(segmented_polygon_path)
            segment_lengths = extract_lengths_from_flow_dict(segmented_polygons_gdf, flow_dict, percentile, buffer_distance=1)  # Buffer distance added

            # Plot and get the exact flow values where length levels out for each segment
            flow_value_dict = plot_flow_vs_length_by_segment(segment_lengths, percentile, output_dir, plot=plot)

            # Initialize a dictionary to store both exact_flow and interpolated_length for each segment
            output_dict = {}

            # Interpolate lengths and create lines based on exact flow values
            lines_output_dir = os.path.join(output_dir, "Modified_Lines")
            os.makedirs(lines_output_dir, exist_ok=True)

            # Initialize a list to collect all modified GeoDataFrames
            all_modified_gdfs = []

            # Determine the CRS from the segmented polygons
            output_crs = segmented_polygons_gdf.crs

            for segment_idx, exact_flow in flow_value_dict.items():
                if np.isnan(exact_flow):
                    logging.warning(f"Segment {segment_idx}: Exact flow is NaN. Skipping line creation.")
                    continue

                # Interpolate length based on exact flow
                interpolated_length = interpolate_length(exact_flow, flow_dict)

                if np.isnan(interpolated_length):
                    logging.warning(f"Segment {segment_idx}: Interpolated length is NaN. Skipping.")
                    continue

                # Get the segment geometry
                segment_geom = segmented_polygons_gdf.geometry.iloc[segment_idx]


                logging.info(f"Segment {segment_idx}: Created line with length {interpolated_length}")

                # Add both exact_flow and interpolated_length to the output_dict
                output_dict[segment_idx] = {
                    'exact_flow': exact_flow,
                    'interpolated_length': interpolated_length
                }

            if all_modified_gdfs:
                # Concatenate all modified GeoDataFrames
                merged_modified_gdf = gpd.GeoDataFrame(pd.concat(all_modified_gdfs, ignore_index=True), crs=output_crs)

                # Define the output path for the merged GeoPackage
                merged_output_path = os.path.join(lines_output_dir, f"all_modified_lines_{percentile}th_percentile.gpkg")

                # Save the merged GeoDataFrame to a single GeoPackage
                merged_modified_gdf.to_file(merged_output_path, driver="GPKG", layer='modified_lines')
                logging.info(f"Merged modified lines saved to: {merged_output_path}")
            else:
                logging.warning("No modified lines were created. Merged GeoPackage will not be generated.")

            # Save the output_dict to a JSON file, including both exact_flow and interpolated_length
            flow_values_output_path = os.path.join(output_dir, "flow_values_with_lengths.json")
            # Convert dictionary keys to strings for JSON compatibility
            output_dict_str_keys = {str(k): v for k, v in output_dict.items()}
            with open(flow_values_output_path, 'w') as f:
                json.dump(output_dict_str_keys, f, indent=4)
            logging.info(f"Flow values with lengths saved to: {flow_values_output_path}")

            # Integrate the 'update_segment_lengths' function here
            # Specify the layer name in your GeoPackage that contains the segments

            update_segment_lengths(
                json_file_path=flow_values_output_path,
                geopackage_path=geopackage_to_update,
                output_geopackage_path= output_width_gpkg
            )

        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
