import rasterio
import rasterio.mask  # Add this line to explicitly import the mask function
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import segment_stream as ss
import geopandas as gpd
from rasterio.merge import merge
import os

def extract_and_clip_selected_raster_by_segment(shapefile, flow_raster_dict, selected_flow_dict, output_dir):
    if isinstance(shapefile, str):
        segments_gdf = gpd.read_file(shapefile)
    elif isinstance(shapefile, gpd.GeoDataFrame):
        segments_gdf = shapefile
    else:
        raise ValueError("shapefile must be a path to a shapefile or a GeoDataFrame.")
    
    stitched_rasters = []
    clip_dir = os.path.join(output_dir, "Clipped_Rasters")
    os.makedirs(clip_dir, exist_ok=True)
    for idx, segment in enumerate(segments_gdf.geometry):
        # Get the selected flow value for this segment
        selected_flow_value = selected_flow_dict[idx]
        
        # Get the corresponding raster path for the selected flow value
        raster_path = flow_raster_dict.get(selected_flow_value)
        print(f"Segment {idx} - Flow Value: {selected_flow_value} - Raster Path: {raster_path}")
        if raster_path:
            with rasterio.open(raster_path) as src:
                #set raster transform to match segment
                src_transform = src.transform
                
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
                clipped_raster_path = f"{clip_dir}/segment_{idx}_flow_{selected_flow_value}.tif"
                with rasterio.open(clipped_raster_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                
                # Append to the list of clipped rasters for merging later
                stitched_rasters.append(clipped_raster_path)
        else:
            print(f"Flow value {selected_flow_value} not found in flow_raster_dict for segment {idx}")

    return stitched_rasters

def stitch_rasters(raster_paths, output_stitched_path):
    # Open all clipped raster files for merging
    src_files_to_mosaic = []
    for path in raster_paths:
        src = rasterio.open(path)
        src_files_to_mosaic.append(src)
    
    # Merge the rasters
    mosaic, out_trans = merge(src_files_to_mosaic)
    
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
    
    # Close all source files
    for src in src_files_to_mosaic:
        src.close()
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile

def extract_max_depths_by_segment(shapefile, flow_raster_dict, percentile=90):
    if isinstance(shapefile, str):
        segments_gdf = gpd.read_file(shapefile)
    elif isinstance(shapefile, gpd.GeoDataFrame):
        segments_gdf = shapefile
    else:
        raise ValueError("shapefile must be a path to a shapefile or a GeoDataFrame.")
    
    # Get the CRS from the shapefile
    shapefile_crs = segments_gdf.crs
    
    segment_depths = {}

    for flow_value, raster_path in flow_raster_dict.items():
        with rasterio.open(raster_path) as src:
            # Check if the CRS of the raster matches the shapefile CRS
            if src.crs != shapefile_crs:
                # Reproject raster to match shapefile CRS
                transform, width, height = calculate_default_transform(
                    src.crs, shapefile_crs, src.width, src.height, *src.bounds)
                
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': shapefile_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                
                # Create an empty array for the reprojected raster
                reprojected_raster = np.empty((src.count, height, width), dtype=src.dtypes[0])
                
                # Perform reprojection
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=reprojected_raster[i - 1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=shapefile_crs,
                        resampling=Resampling.nearest)
                
                # Create an in-memory raster with the reprojected data
                with MemoryFile() as memfile:
                    with memfile.open(**kwargs) as dataset:
                        dataset.write(reprojected_raster)
                        
                        # Now loop through each segment and mask the reprojected raster
                        for idx, segment in enumerate(segments_gdf.geometry):
                            segment_depths.setdefault(idx, {})
                            # Mask the reprojected raster with the polygon
                            out_image, out_transform = rasterio.mask.mask(dataset, [segment], crop=True)
                            # Get the 90th percentile depth, omitting nodata values
                            max_depth = np.percentile(out_image[out_image != src.nodata], percentile)
                            segment_depths[idx][flow_value] = max_depth
            else:
                # No reprojection needed, loop through each segment
                for idx, segment in enumerate(segments_gdf.geometry):
                    segment_depths.setdefault(idx, {})
                    # Mask raster with polygon directly
                    out_image, out_transform = rasterio.mask.mask(src, [segment], crop=True)
                    # Get the 90th percentile depth
                    max_depth = np.percentile(out_image[out_image != src.nodata], percentile)
                    segment_depths[idx][flow_value] = max_depth

    return segment_depths


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

def plot_flow_vs_depth_by_segment(segment_depths):
    flow_value_dict = {}

    for segment_idx, depths in segment_depths.items():
        flow_values = list(depths.keys())
        max_depths = list(depths.values())

        # Smooth the max depths
        smoothed_depths = smooth_data(max_depths)

        # Find leveling out point
        leveling_out_index = find_leveling_out_point(flow_values, smoothed_depths)

        # Fit a polynomial to the smoothed data
        poly = fit_polynomial(flow_values, smoothed_depths)
        fitted_values = poly(flow_values)

        # Plot original and smoothed data
        plt.figure(figsize=(10, 6))
        plt.plot(flow_values, max_depths, marker='o', label='Original Max Depths')
        plt.plot(flow_values, smoothed_depths, label='Smoothed Depths', color='orange')
        plt.plot(flow_values, fitted_values, label='Fitted Polynomial', color='green')
        plt.axvline(flow_values[leveling_out_index], color='red', linestyle='--', 
                    label=f'Leveling out at Flow = {flow_values[leveling_out_index]}')
        plt.title(f'Flow Value vs Max Depth for Segment {segment_idx}')
        plt.xlabel('Flow Value')
        plt.ylabel('Max Depth')
        plt.grid(True)
        plt.legend()
        # set axis limits
        plt.xlim(0, max(flow_values))
        plt.ylim(0, max(max_depths))
        plt.show()
        flow_value_dict[segment_idx] = flow_values[leveling_out_index]

    return flow_value_dict

if __name__ == "__main__":
    # Update input paths
    prefix = 'ME'
    # segmented_polygon_path = os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Segmented Valleys", 
    #                                       f"{prefix}_segmented_valley_5m.gpkg")
    segmented_polygon_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Segmented Valleys\Bennett_Segmented_Valleys.gpkg"
    
    basefolder = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Max Depth Rasters"
    
    scratch_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test\Clipped_Rasters"
    output_dir = os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leveling Test", "Results", prefix)
    percentile = 50
    
    flow_raster_dict = {
        0.25 : os.path.join(basefolder, prefix, f"{prefix}_0o25cms.tif"),
        0.5: os.path.join(basefolder, prefix, f"{prefix}_0o5cms.tif"),
        #0.75:   os.path.join(basefolder, prefix, f"{prefix}_0o75cms.tif"),
        1: os.path.join(basefolder, prefix, f"{prefix}_1cms.tif"),
       # 1.5: os.path.join(basefolder, prefix, f"{prefix}_1o50cms.tif"),
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


    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)
    

    stitched_raster_title = f"Stitched_Raster_{percentile}th_per_5m.tif"
    stitched_raster_output_path = os.path.join(output_dir, 
                                                stitched_raster_title)

        
    # Extract max depths for each segment
    segmented_polygons_gdf = gpd.read_file(segmented_polygon_path)
    max_depths = extract_max_depths_by_segment(segmented_polygons_gdf, flow_raster_dict, percentile)

    # Plot and get the flow values where depth levels out for each segment
    flow_value_dict = plot_flow_vs_depth_by_segment(max_depths)

    # Clip only the selected raster for each segment based on flow_value_dict
    clipped_rasters = extract_and_clip_selected_raster_by_segment(segmented_polygons_gdf, flow_raster_dict, flow_value_dict, output_dir)

    # Stitch clipped rasters into a single output
    stitch_rasters(clipped_rasters, stitched_raster_output_path)

    for segment, flow in flow_value_dict.items():
        print(f"Segment {segment} levels out at flow value {flow}")
