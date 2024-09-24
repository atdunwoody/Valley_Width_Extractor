import geopandas as gpd
import rasterio
from rasterio.mask import mask
import os
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import rasterio
from rasterio.merge import merge
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

def plot_flow_vs_depth_by_point(point_depths, output_dir = None):
    flow_value_dict = {}

    for point_idx, depths in point_depths.items():
        flow_values = list(depths.keys())
        max_depths = list(depths.values())

        # Smooth the max depths
        smoothed_depths = smooth_data(max_depths)
    
        # Find leveling out point
        leveling_out_index = find_leveling_out_point(flow_values, smoothed_depths)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt_save_path = os.path.join(output_dir, f"point_{point_idx}_flow_vs_depth.png")
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
            plt.savefig(plt_save_path)

            second_der_path = os.path.join(output_dir, f"point_{point_idx}_second_derivative.png")
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
            plt.savefig(second_der_path)

        flow_value_dict[point_idx] = flow_values[leveling_out_index]

    return flow_value_dict


def clip_raster_to_polygon(raster_path, polygon, output_path):
    """
    Clip a raster to a polygon and save the output.
    """
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    return output_path

def stitch_rasters(raster_paths, output_raster_path):
    """
    Stitch multiple rasters together into a single raster.
    """
    src_files_to_mosaic = []
    for path in raster_paths:
        src = rasterio.open(path)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close all source files
    for src in src_files_to_mosaic:
        src.close()

def extract_max_depths_and_clip_rasters(shapefile, flow_raster_dict, segmented_polygons, flow_value_dict, output_dir):
    if isinstance(shapefile, str):
        points_gdf = gpd.read_file(shapefile)
    elif isinstance(shapefile, gpd.GeoDataFrame):
        points_gdf = shapefile
    else:
        raise ValueError("shapefile must be a path to a shapefile or a GeoDataFrame.")
    
    point_depths = {}
    raster_clips = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for idx, point in enumerate(points_gdf.geometry):
        point_depths[idx] = {}

        # Find the corresponding polygon segment
        for poly_idx, poly in segmented_polygons.iterrows():
            if poly['geometry'].contains(point):
                polygon = poly['geometry']
                break
        else:
            continue  # Skip if no polygon contains the point

        # Get the flow value for the leveling out point of the current point
        selected_flow_value = flow_value_dict.get(idx)

        if selected_flow_value is not None:
            # Get the corresponding raster for the selected flow value
            raster_path = flow_raster_dict[selected_flow_value]
            with rasterio.open(raster_path) as src:
                row, col = src.index(point.x, point.y)
                max_depth = src.read(1)[row, col]
                point_depths[idx][selected_flow_value] = max_depth

            # Clip the raster to the corresponding polygon
            clip_output_path = os.path.join(output_dir, f"point_{idx}_flow_{selected_flow_value}.tif")
            clipped_raster = clip_raster_to_polygon(raster_path, polygon, clip_output_path)
            raster_clips.append(clipped_raster)
    return point_depths, raster_clips


# Stitching the clipped rasters
def stitch_and_save_clipped_rasters(raster_clips, output_raster_path):
    """
    Stitch together the list of clipped rasters and save as a single output raster.
    """
    stitch_rasters(raster_clips, output_raster_path)

if __name__ == '__main__':
    
    prefix_list = ['ME', 'MM', 'MW', 'UM', 'UE', 'UW']
    spacing = 50
    basefolder = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters"
    segmented_poly_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Segmented Valleys\Bennett_Segmented_Valleys.gpkg"
    #segmented_poly_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Buffered Segmented Valleys\Bennett_Valleys_100m.gpkg"
    scratch_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Clipped_Rasters"
    
    for prefix in prefix_list:
        centerline_path = os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines", f"{prefix}_clipped.gpkg")
        output_dir = os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test", f"Results_{spacing}m", prefix)
        plot_dir = os.path.join(output_dir, 'plots')
        output_points = os.path.join(output_dir, f"{prefix}_points_{spacing}m.gpkg")
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
        if prefix == 'ME':
            #remove 0.75 and 1.5
            flow_raster_dict.pop(0.75)
            flow_raster_dict.pop(1.5)
        try:
            points_gdf = ss.create_points_along_line(centerline_path, spacing = spacing, output_path=output_points)
            max_depths = extract_max_depths_by_point(points_gdf, flow_raster_dict)
            flow_value_dict = plot_flow_vs_depth_by_point(max_depths, output_dir=plot_dir)

            for point, flow in flow_value_dict.items():
                print(f"Point {point} levels out at flow value {flow}")

            segmented_polygons = gpd.read_file(segmented_poly_path)
            output_dir_clips = os.path.join(scratch_dir, 'clipped_rasters')
            stitched_raster_output_path = os.path.join(output_dir, f"{prefix}_valley_bottom_{spacing}m.tif")

            max_depths, raster_clips = extract_max_depths_and_clip_rasters(points_gdf, flow_raster_dict, segmented_polygons, flow_value_dict, output_dir_clips)
            stitch_and_save_clipped_rasters(raster_clips, stitched_raster_output_path)
        except Exception as e:
            print(f"Error processing {prefix}: {e}")
            continue  
