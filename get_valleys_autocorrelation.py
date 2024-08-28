import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
import os
from scipy.signal import savgol_filter
from statsmodels.tsa.stattools import acf

def sample_raster_along_line(line, raster, n_points=None, nodata_value=None):
    if n_points is None:
        n_points = int(line.length)
    distances = np.linspace(0, line.length, n_points)
    points = [line.interpolate(distance) for distance in distances]
    
    raster_values = []
    valid_points = []
    valid_distances = []

    # Get the raster dimensions
    raster_height, raster_width = raster.read(1).shape

    for distance, point in zip(distances, points):
        row, col = raster.index(point.x, point.y)
        
        # Check if the row and col are within the valid range
        if 0 <= row < raster_height and 0 <= col < raster_width:
            value = raster.read(1)[row, col]
            
            if value != nodata_value:
                raster_values.append(value)
                valid_points.append(point)
                valid_distances.append(distance)
        else:
            print(f"Point at distance {distance} falls outside raster bounds (row: {row}, col: {col}) and will be skipped.")

    return valid_distances, raster_values, valid_points

def compute_cross_sectional_area_trapezoidal(x, y, depth):
    y_adjusted = np.clip(depth - y, 0, None)
    area = np.trapz(y_adjusted, x)
    return area

def compute_wetted_perimeter(x, y, depth):
    perimeter = 0.0
    for i in range(1, len(x)):
        if y[i] < depth and y[i-1] < depth:
            segment_length = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
            perimeter += segment_length
        elif y[i] < depth:
            segment_length = np.sqrt((x[i] - x[i-1])**2 + (depth - y[i])**2)
            perimeter += segment_length
        elif y[i-1] < depth:
            segment_length = np.sqrt((x[i] - x[i-1])**2 + (depth - y[i-1])**2)
            perimeter += segment_length
    return perimeter

def determine_side_of_centerline(point, centerline):
    if isinstance(centerline, LineString):
        nearest_line = centerline
    elif isinstance(centerline, MultiLineString):
        # Find the closest LineString within the MultiLineString
        nearest_line = min(centerline.geoms, key=lambda line: line.distance(point))
    else:
        raise ValueError("Unsupported geometry type")

    nearest_point_on_centerline = nearest_line.interpolate(nearest_line.project(point))
    vector_to_point = np.array([point.x - nearest_point_on_centerline.x, point.y - nearest_point_on_centerline.y])
    reference_direction = np.array([nearest_line.coords[-1][0] - nearest_line.coords[0][0], nearest_line.coords[-1][1] - nearest_line.coords[0][1]])
    
    side = np.sign(np.cross(reference_direction, vector_to_point))
    return side


def smooth_ratio(ratio, window_length=11, polyorder=3):
    # Ensure window_length is appropriate given the size of the ratio array
    if len(ratio) < window_length:
        window_length = len(ratio)
        if window_length % 2 == 0:  # Window length must be odd
            window_length -= 1
    
    # Ensure polyorder is less than window_length
    if polyorder >= window_length:
        polyorder = window_length - 1
    
    return savgol_filter(ratio, window_length, polyorder)


def autocorrelation_analysis(data, lag=10):
    """Calculate the autocorrelation of the data for a given lag."""
    autocorr = acf(data, nlags=lag, fft=True)
    return autocorr

def find_leveling_point_autocorr(x, y, depth_increment=0.1, polynomial_order=4, 
                                smooth=True, window_length=11, polyorder=2, lag=10, threshold=0.5):
    depth = np.arange(min(y), max(y), depth_increment)
    ratio = []
    
    for d in depth:
        area = compute_cross_sectional_area_trapezoidal(x, y, d)
        perimeter = compute_wetted_perimeter(x, y, d)
        if perimeter > 0:
            ratio.append(area / perimeter)
        else:
            ratio.append(0)
    
    ratio = np.array(ratio)
    
    if smooth:
        ratio = smooth_ratio(ratio, window_length, polyorder)
    
    poly_coeffs = np.polyfit(depth, ratio, polynomial_order)
    poly_fit = np.polyval(poly_coeffs, depth)
    
    # First derivative
    first_derivative = np.gradient(poly_fit, depth)
    
    # Second derivative (to find inflection points)
    second_derivative = np.gradient(first_derivative, depth)
    
    # Perform autocorrelation analysis on the second derivative
    autocorr = autocorrelation_analysis(second_derivative, lag)
    
    # Detect significant drop in autocorrelation
    significant_drop_idx = np.where(np.abs(np.diff(autocorr)) > threshold)[0]
    
    if len(significant_drop_idx) > 0:
        leveling_out_elevation = depth[significant_drop_idx[0]]
    else:
        leveling_out_elevation = None
    
    return leveling_out_elevation

def plot_cross_section_area_to_wetted_perimeter_ratio(x, y, idx='', depth_increment=0.05, fig_output_path='', 
                                                      polynomial_order=4, smooth=True, window_length=11, polyorder=3, 
                                                      print_output=True, lag=10, threshold=0.01):
    
    leveling_out_elevation = find_leveling_point_autocorr(x, y, depth_increment, polynomial_order, 
                                                          smooth, window_length, polyorder, lag, threshold)

    depth = np.arange(min(y), max(y), depth_increment)
    ratio = []
    
    for d in depth:
        area = compute_cross_sectional_area_trapezoidal(x, y, d)
        perimeter = compute_wetted_perimeter(x, y, d)
        if perimeter > 0:
            ratio.append(area / perimeter ** 2)
        else:
            ratio.append(0)
    
    ratio = np.array(ratio)
    
    if smooth:
        ratio = smooth_ratio(ratio, window_length, polyorder)
    
    poly_coeffs = np.polyfit(depth, ratio, polynomial_order)
    poly_fit = np.polyval(poly_coeffs, depth)
    
    # First derivative for plotting
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

        fig_save_path = os.path.join(fig_output_path, f'cross_section_area_to_wetted_perimeter_ratio_{idx}.png')
        plt.savefig(fig_save_path)
        plt.close()

    return leveling_out_elevation


def main(gpkg_path, raster_path, output_folder, output_gpkg_path=None, 
         centerline_gpkg=None, second_derivative_num = 50, print_output=True,
         threshold=0.05):
    gdf = gpd.read_file(gpkg_path)
    centerline_gdf = gpd.read_file(centerline_gpkg)
    centerline = centerline_gdf.geometry.iloc[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if output_gpkg_path and os.path.exists(output_gpkg_path):
        os.remove(output_gpkg_path)
    
    left_points = []
    right_points = []

    with rasterio.open(raster_path) as raster:
        nodata_value = raster.nodata
        total_lines = len(gdf)
        print(f"Total lines to process: {total_lines}")
        for idx, row in gdf.iterrows():
            line = row.geometry
            if isinstance(line, (LineString, MultiLineString)):
                print(f'Processing line {idx} of {total_lines}')
                valid_distances, valid_raster_values, valid_points = sample_raster_along_line(line, raster, nodata_value=nodata_value)
                
                if len(valid_distances) == 0 or len(valid_points) == 0:
                    print(f"Skipping line {idx} due to all NoData values.")
                    continue

                #np.savetxt(f'{output_folder}/sampled_data_{idx}.csv', np.column_stack((valid_distances, valid_raster_values)), delimiter=',', header='Distance,Raster Value', comments='')
                
                leveling_out_elevation = plot_cross_section_area_to_wetted_perimeter_ratio(valid_distances, valid_raster_values, idx=idx, 
                                                                                           fig_output_path=output_folder, 
                                                                                           print_output=print_output, threshold=threshold)
                
                if leveling_out_elevation is not None:
                    sides = np.array([determine_side_of_centerline(point, centerline) for point in valid_points])
                    
                    left_side_indices = np.where(sides < 0)[0]
                    right_side_indices = np.where(sides > 0)[0]
                    
                    if len(left_side_indices) > 0 and len(right_side_indices) > 0:
                        left_side_values = np.array(valid_raster_values)[left_side_indices]
                        right_side_values = np.array(valid_raster_values)[right_side_indices]
                        
                        closest_left_index = left_side_indices[np.argmin(np.abs(left_side_values - leveling_out_elevation))]
                        closest_right_index = right_side_indices[np.argmin(np.abs(right_side_values - leveling_out_elevation))]
                        
                        closest_left_point = valid_points[closest_left_index]
                        closest_right_point = valid_points[closest_right_index]
                        
                        left_points.append(closest_left_point)
                        right_points.append(closest_right_point)
                        
                        print(f"Leveling out point on left side for line index {idx}: {closest_left_point} at elevation {leveling_out_elevation}")
                        print(f"Leveling out point on right side for line index {idx}: {closest_right_point} at elevation {leveling_out_elevation}")
            else:
                geometry_type = line.geom_type
                print(f'Skipping non-LineString geometry at index {idx} with geometry type: {geometry_type}')
    
    if left_points and right_points:
        polygon_coords = left_points + right_points[::-1] + [left_points[0]]
        polygon = Polygon([(point.x, point.y) for point in polygon_coords])
        
        polygon_gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs=gdf.crs)
        polygon_gdf.to_file(output_gpkg_path, driver="GPKG")
        print(f"Polygon GeoPackage created at: {output_gpkg_path}")


if __name__ == "__main__":
    perpendiculars_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Perpendiculars"
    raster_dir = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\Watershed_Clipped"
    centerlines_dir = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\Centerlines"
    
    perpendiculars_paths = [os.path.join(perpendiculars_dir, f) for f in os.listdir(perpendiculars_dir) if f.endswith('.gpkg')]
    raster_paths = [os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith('.tif')]
    centerline_paths = [os.path.join(centerlines_dir, f) for f in os.listdir(centerlines_dir) if f.endswith('.gpkg')]
    
    for perpendiculars_path, raster_path, centerline_gpkg in zip(perpendiculars_paths, raster_paths, centerline_paths):
        
        watershed_name = os.path.basename(perpendiculars_path).split('_')[0]
    
        output_folder = os.path.join(r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints", watershed_name)

        #create threshold array from 0.01 to 0.1 in 0.005 increments
        thresholds = np.arange(0.01, 0.1, 0.005)
        for threshold in thresholds:
            output_gpkg_name = f"Valley_Footprint_Full_autocorr_{threshold}.gpkg"
            output_gpkg_path = os.path.join(output_folder, output_gpkg_name)
            #skip if output file already exists
            if os.path.exists(output_gpkg_path):
                print(f"Output file already exists at: {output_gpkg_path}")
                continue
            print(f"Processing watershed: {watershed_name}")
            print(f"Output geopackage path: {output_gpkg_path}")
            main(perpendiculars_path, raster_path, output_folder, output_gpkg_path, 
                centerline_gpkg, threshold=threshold, print_output=False)
