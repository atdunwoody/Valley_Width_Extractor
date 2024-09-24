#!/usr/bin/env python3
"""
plot_tertrain_vs_depth.py

This script reads a DEM and perpendicular lines, extracts cross-sectional elevations,
and plots them alongside horizontal lines representing selected flow depths.
Inputs are loaded from a configuration file.

Usage:
    python plot_tertrain_vs_depth.py --config path/to/config.yaml
"""

import argparse
import yaml
import logging
import os
import json

import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.ops import substring
from rasterio.features import geometry_mask

def setup_logging(log_file, level=logging.INFO):
    """
    Sets up logging for the script.
    
    Parameters:
        log_file (str): Path to the log file.
        level (int): Logging level.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """
    Loads the YAML configuration file.
    
    Parameters:
        config_path (str): Path to the configuration file.
    
    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_flow_values(flow_values_path):
    """
    Loads the flow values dictionary from a JSON file.
    
    Parameters:
        flow_values_path (str): Path to the JSON file containing flow values.
    
    Returns:
        dict: Mapping of segment indices to flow values.
    """
    with open(flow_values_path, 'r') as file:
        flow_values = json.load(file)
    return flow_values

def extract_elevation_along_line(dem_dataset, line, num_points=500):
    """
    Extracts elevation values from the DEM along the given line.
    
    Parameters:
        dem_dataset (rasterio.DatasetReader): Opened DEM raster dataset.
        line (shapely.geometry.LineString): The line along which to extract elevations.
        num_points (int): Number of points to sample along the line.
    
    Returns:
        tuple: (distances in meters, elevation values)
    """
    distances = np.linspace(0, line.length, num=num_points)
    points = [line.interpolate(distance) for distance in distances]
    elevations = []

    for point in points:
        row, col = dem_dataset.index(point.x, point.y)
        try:
            elevation = dem_dataset.read(1)[row, col]
            if dem_dataset.nodata is not None and elevation == dem_dataset.nodata:
                elevation = np.nan
        except IndexError:
            elevation = np.nan
        elevations.append(elevation)
    
    # Calculate cumulative distance
    cumulative_distance = np.linspace(0, line.length, num=num_points)

    return cumulative_distance, np.array(elevations)

def plot_cross_section(segment_idx, distances, elevations, flow_depth, output_dir):
    """
    Plots the cross-section elevation with a horizontal flow depth line and crops the x-axis
    to 50% more on either side of where the depth intersects with the elevation.
    
    Parameters:
        segment_idx (int): The segment index.
        distances (np.ndarray): Distances along the cross-section.
        elevations (np.ndarray): Elevation values along the cross-section.
        flow_depth (float): Selected flow depth for the segment.
        output_dir (str): Directory to save the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import logging

    # Add minimum elevation to flow depth to get the absolute flow depth
    min_elevation = np.nanmin(elevations)
    flow_depth_absolute = flow_depth + min_elevation

    # Initialize plot
    plt.figure(figsize=(12, 6))
    plt.plot(distances, elevations, label='Terrain Elevation', color='blue')

    if not np.isnan(flow_depth_absolute):
        plt.axhline(y=flow_depth_absolute, color='red', linestyle='--', label=f'Flow Depth: {flow_depth:.2f} m')
    else:
        logging.warning(f"Segment {segment_idx}: Flow depth is NaN. Skipping flow depth line.")

    # Title and labels
    plt.title(f'Segment {segment_idx} Cross-Section')
    plt.xlabel('Distance along cross-section (meters)')
    plt.ylabel('Elevation')
    plt.legend()
    plt.grid(True)

    # Find intersection points where elevation crosses the flow depth
    elevation_diff = elevations - flow_depth_absolute
    sign_changes = np.where(np.diff(np.sign(elevation_diff)))[0]

    if len(sign_changes) == 0:
        logging.warning(f"Segment {segment_idx}: No intersection between elevation and flow depth.")
        # Optionally, you can choose to plot the entire cross-section or apply a default cropping
        plt.tight_layout()
    else:
        # Estimate the exact intersection points using linear interpolation
        intersection_distances = []
        for idx_cross in sign_changes:
            x1, y1 = distances[idx_cross], elevation_diff[idx_cross]
            x2, y2 = distances[idx_cross + 1], elevation_diff[idx_cross + 1]
            if y2 - y1 != 0:
                x_intersect = x1 - y1 * (x2 - x1) / (y2 - y1)
                intersection_distances.append(x_intersect)
            else:
                intersection_distances.append(distances[idx_cross])

        # Define the cropping window based on intersection points
        min_intersect = min(intersection_distances)
        max_intersect = max(intersection_distances)

        # Calculate the window size and extend by 50% on each side
        window_size = max_intersect - min_intersect
        if window_size == 0:
            # If intersections are at the same point, define a default window
            window_size = np.max(distances) - np.min(distances)
        padding = 0.5 * window_size

        # Set new x-axis limits
        new_min = max(np.min(distances), min_intersect - padding)
        new_max = min(np.max(distances), max_intersect + padding)
        plt.xlim(new_min, new_max)

    # Save the plot
    plot_filename = os.path.join(output_dir, f'segment_{segment_idx}_cross_section.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    logging.info(f"Plot saved: {plot_filename}")


def main():
    # Argument parser for configuration file
    parser = argparse.ArgumentParser(description="Plot terrain cross-sections with flow depth lines.")
    parser.add_argument('--config', type=str, default='Hydraulic Valleys/plot terrain vs depth/config.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO').upper(), logging.INFO)
    log_file = config.get('logging', {}).get('log_file', 'plot_tertrain_vs_depth.log')
    setup_logging(log_file, level=log_level)
    logging.info("Logging is set up.")

    # Create output directory if it doesn't exist
    output_dir = config.get('output', {}).get('directory', 'output_plots')
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory is set to: {output_dir}")

    try:
        # Load DEM
        dem_path = config['dem']['path']
        logging.info(f"Loading DEM from: {dem_path}")
        dem = rasterio.open(dem_path)
    except Exception as e:
        logging.error(f"Failed to load DEM: {e}")
        return

    try:
        # Load perpendicular lines
        lines_path = config['perpendicular_lines']['path']
        logging.info(f"Loading perpendicular lines from: {lines_path}")
        lines_gdf = gpd.read_file(lines_path)
    except Exception as e:
        logging.error(f"Failed to load perpendicular lines: {e}")
        return

    try:
        # Load flow values
        flow_values_path = config['flow_values']['path']
        logging.info(f"Loading flow values from: {flow_values_path}")
        flow_value_dict = load_flow_values(flow_values_path)
    except Exception as e:
        logging.error(f"Failed to load flow values: {e}")
        return

    # Iterate over each perpendicular line segment
    for idx, row in lines_gdf.iterrows():
        segment_idx = idx  # Assuming index corresponds to segment identifier
        line_geom = row.geometry

        if not isinstance(line_geom, LineString):
            logging.warning(f"Segment {segment_idx}: Geometry is not a LineString. Skipping.")
            continue

        logging.info(f"Processing Segment {segment_idx}.")

        # Extract elevation along the line
        distances, elevations = extract_elevation_along_line(dem, line_geom)

        # Retrieve the selected flow depth for this segment
        flow_depth = flow_value_dict.get(str(segment_idx))  # Assuming JSON keys are strings
        if flow_depth is None:
            logging.warning(f"Segment {segment_idx}: No flow depth found. Setting as NaN.")
            flow_depth = np.nan

        # Plot cross-section with flow depth
        plot_cross_section(segment_idx, distances, elevations, flow_depth, output_dir)

    # Close the DEM dataset
    dem.close()
    logging.info("Processing completed successfully.")

if __name__ == "__main__":
    main()
