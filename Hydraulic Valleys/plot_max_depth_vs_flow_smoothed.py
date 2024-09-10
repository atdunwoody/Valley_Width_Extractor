import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import segment_stream as ss
import os

def extract_max_depths_by_point(shapefile, flow_raster_dict):
    if isinstance(shapefile, str):
        points_gdf = gpd.read_file(shapefile)
    elif isinstance(shapefile, gpd.GeoDataFrame):
        points_gdf = shapefile
    else:
        raise ValueError("shapefile must be a path to a shapefile or a GeoDataFrame.")
    point_depths = {}

    for idx, point in enumerate(points_gdf.geometry):
        point_depths[idx] = {}
        for flow_value, raster_path in flow_raster_dict.items():
            with rasterio.open(raster_path) as src:
                row, col = src.index(point.x, point.y)
                max_depth = src.read(1)[row, col]
                point_depths[idx][flow_value] = max_depth

    return point_depths

def smooth_data(data, window_length=5, polyorder=2):
    """
    Apply Savitzky-Golay filter for data smoothing.
    """
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)

def fit_polynomial(x, y, degree=3):
    """
    Fit a polynomial to the data.
    """
    coeffs = np.polyfit(x, y, degree)
    return np.poly1d(coeffs)

def find_leveling_out_point(flow_values, smoothed_depths):
    """
    Use first and second derivatives to find the leveling out point.
    """
    first_derivative = np.gradient(smoothed_depths, flow_values)
    second_derivative = np.gradient(first_derivative, flow_values)

    # Use both the first derivative approaching zero and second derivative being small
    leveling_out_index = np.argmin(np.abs(second_derivative))  # closest to zero

    return leveling_out_index

def plot_flow_vs_depth_by_point(point_depths):
    flow_value_dict = {}

    for point_idx, depths in point_depths.items():
        flow_values = list(depths.keys())
        max_depths = list(depths.values())

        # Smooth the max depths
        smoothed_depths = smooth_data(max_depths)

        # Find leveling out point
        leveling_out_index = find_leveling_out_point(flow_values, smoothed_depths)

        # Plot original and smoothed data
        plt.figure(figsize=(10, 6))
        plt.plot(flow_values, max_depths, marker='o', label='Original Max Depths')
        plt.plot(flow_values, smoothed_depths, label='Smoothed Depths', color='orange')
        plt.axvline(flow_values[leveling_out_index], color='red', linestyle='--', 
                    label=f'Leveling out at Flow = {flow_values[leveling_out_index]}')
        plt.title(f'Flow Value vs Max Depth for Point {point_idx}')
        plt.xlabel('Flow Value')
        plt.ylabel('Max Depth')
        plt.grid(True)
        plt.legend()

        plt.figure(figsize=(10, 6))
        first_derivative = np.gradient(smoothed_depths, flow_values)
        second_derivative = np.gradient(first_derivative, flow_values)
        plt.plot(flow_values, second_derivative, marker='o', color='green', label='Second Derivative')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'Second Derivative of Flow vs Max Depth for Point {point_idx}')
        plt.xlabel('Flow Value')
        plt.ylabel('Second Derivative')
        plt.grid(True)
        plt.legend()
        plt.show()
        flow_value_dict[point_idx] = flow_values[leveling_out_index]

    return flow_value_dict

centerline_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\MM_clipped.gpkg"
prefix = 'MM'
basefolder = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters"

segmented_poly_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Segmented Valleys\Bennett_Segmented_Valleys.gpkg"
scratch_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Clipped_Rasters"
output_dir = os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test", "Results", prefix)

flow_raster_dict = {
    0.25 : os.path.join(basefolder, prefix, f"{prefix}_0o25cms.tif"),
    0.5: os.path.join(basefolder, prefix, f"{prefix}_0o5cms.tif"),
    0.75:   os.path.join(basefolder, prefix, f"{prefix}_0o75cms.tif"),
    1: os.path.join(basefolder, prefix, f"{prefix}_1cms.tif"),
    1.5: os.path.join(basefolder, prefix, f"{prefix}_1o50cms.tif"),
    2: os.path.join(basefolder, prefix, f"{prefix}_2cms.tif"),
    3: os.path.join(basefolder, prefix, f"{prefix}_3cms.tif"),
    4: os.path.join(basefolder, prefix, f"{prefix}_4cms.tif"),
    5: os.path.join(basefolder, prefix, f"{prefix}_5cms.tif"),
    6: os.path.join(basefolder, prefix, f"{prefix}_6cms.tif"),
    7: os.path.join(basefolder, prefix, f"{prefix}_7cms.tif"),
    8: os.path.join(basefolder, prefix, f"{prefix}_8cms.tif"),
    9: os.path.join(basefolder, prefix, f"{prefix}_9cms.tif"),
    10: os.path.join(basefolder, prefix, f"{prefix}_10cms.tif"),
}

points_gdf = ss.create_points_along_line(centerline_path, spacing = 50)
max_depths = extract_max_depths_by_point(points_gdf, flow_raster_dict)
flow_value_dict = plot_flow_vs_depth_by_point(max_depths)

for point, flow in flow_value_dict.items():
    print(f"Point {point} levels out at flow value {flow}")

