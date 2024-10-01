import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import json
from shapely.geometry import LineString, MultiLineString, Point
import os
import logging
import sys
import argparse

def sample_raster_along_line(line, raster, n_points=100):
    """
    Sample raster values along a LineString or MultiLineString.

    Parameters:
        line (LineString or MultiLineString): The line along which to sample.
        raster (rasterio.io.DatasetReader): The opened raster dataset.
        n_points (int): Number of points to sample along the line.

    Returns:
        distances (array): Distances along the line.
        elevations (array): Elevations sampled from the raster.
        points (list): Shapely Point objects of sampled points.
    """
    # Generate equally spaced distances along the line
    distances = np.linspace(0, line.length, n_points)
    points = [line.interpolate(distance) for distance in distances]

    xs = np.array([point.x for point in points])
    ys = np.array([point.y for point in points])

    # Prepare coordinates for sampling
    coords = [(x, y) for x, y in zip(xs, ys)]

    # Sample the raster at these coordinates
    try:
        elevations = [val[0] if val else np.nan for val in raster.sample(coords)]
    except Exception as e:
        logging.error(f"Error sampling raster: {e}")
        return [], [], []

    elevations = np.array(elevations)

    # Handle nodata values
    nodata = raster.nodata
    if nodata is not None:
        valid_data_mask = elevations != nodata
    else:
        valid_data_mask = ~np.isnan(elevations)

    distances = distances[valid_data_mask]
    elevations = elevations[valid_data_mask]
    points = [points[i] for i in range(len(points)) if valid_data_mask[i]]

    return distances, elevations, points

def main(lines_gpkg_path, raster_path, json_path, output_folder):
    # Read lines from GeoPackage
    try:
        gdf = gpd.read_file(lines_gpkg_path)
        logging.info(f"Read {len(gdf)} lines from {lines_gpkg_path}")
    except Exception as e:
        logging.error(f"Error reading lines GeoPackage: {e}")
        sys.exit(1)

    # Open raster DEM
    try:
        raster = rasterio.open(raster_path)
        logging.info(f"Opened raster {raster_path}")
    except Exception as e:
        logging.error(f"Error opening raster: {e}")
        sys.exit(1)

    # Read JSON file
    try:
        with open(json_path, 'r') as f:
            flow_depths = json.load(f)
        logging.info(f"Read flow depths from {json_path}")
    except Exception as e:
        logging.error(f"Error reading JSON file: {e}")
        sys.exit(1)

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created output folder {output_folder}")

    # Loop over lines
    for idx, row in gdf.iterrows():
        line = row.geometry
        if not isinstance(line, (LineString, MultiLineString)):
            logging.warning(f"Geometry at index {idx} is not a LineString or MultiLineString. Skipping.")
            continue
        # Sample raster along line
        try:
            distances, elevations, points = sample_raster_along_line(line, raster)
            if len(distances) == 0:
                logging.warning(f"No valid data for line at index {idx}. Skipping.")
                continue
        except Exception as e:
            logging.error(f"Error sampling raster along line at index {idx}: {e}")
            continue

        # Plot the terrain cross-section
        plt.figure()
        plt.plot(distances, elevations, label='Terrain Cross-Section', color='blue')

        # Get flow depth from JSON file
        segment_index = str(idx)
        if segment_index in flow_depths:
            flow_depth = float(flow_depths[segment_index])
            # Draw horizontal red line at elevation of flow depth
            plt.axhline(y=flow_depth, color='red', linestyle='--', label='Flow Depth')

            # Compute the difference between flow depth and minimum elevation
            min_elevation = np.min(elevations)
            depth_minus_min_elevation = flow_depth - min_elevation

            # Add text to the plot
            plt.text(0.05, 0.95, f'Depth: {depth_minus_min_elevation:.2f} m',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.5))
        else:
            # No flow depth for this segment
            logging.info(f"No flow depth for segment {segment_index}")
            flow_depth = None

        plt.xlabel('Distance along line (m)')
        plt.ylabel('Elevation (m)')
        plt.title(f'Terrain Cross-Section at Line {idx}')
        plt.legend()
        #place legend in lower left corner
        plt.legend(loc='lower left')
        #place text in upper right corner
        plt.text(0.95, 0.95, f'Elevation: {flow_depth:.2f}m', transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.5))
        
        plot_filename = os.path.join(output_folder, f'{idx}_cross_section.png')
        plt.savefig(plot_filename)
        plt.close()
        logging.info(f"Saved plot for line {idx} to {plot_filename}")

    logging.info("Cross-section plots have been generated successfully.")

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[
                            logging.StreamHandler()
                        ])

    parser = argparse.ArgumentParser(description='Plot terrain cross-sections with flow depths')
    parser.add_argument('lines_gpkg', help='Path to lines GeoPackage')
    parser.add_argument('dem_raster', help='Path to DEM raster')
    parser.add_argument('json_file', help='Path to JSON file with flow depths')
    parser.add_argument('output_folder', help='Folder to save output plots')
    args = parser.parse_args()

    main(args.lines_gpkg, args.dem_raster, args.json_file, args.output_folder)
