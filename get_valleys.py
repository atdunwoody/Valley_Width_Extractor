import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point
import os

def sample_raster_along_line(line, raster, n_points=None):
    if n_points is None:
        n_points = int(line.length)
    distances = np.linspace(0, line.length, n_points)
    points = [line.interpolate(distance) for distance in distances]
    
    raster_values = []
    for point in points:
        row, col = raster.index(point.x, point.y)
        raster_values.append(raster.read(1)[row, col])
    
    return distances, raster_values, points

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

def find_leveling_point(x, y, depth_increment=0.1, polynomial_order=4):
    depth = np.arange(min(y), max(y), depth_increment)
    ratio = []
    
    for d in depth:
        area = compute_cross_sectional_area_trapezoidal(x, y, d)
        perimeter = compute_wetted_perimeter(x, y, d)
        if perimeter > 0:
            ratio.append(area / (perimeter ** 4))
        else:
            ratio.append(0)
    
    ratio = np.array(ratio)
    poly_coeffs = np.polyfit(depth, ratio, polynomial_order)
    poly_fit = np.polyval(poly_coeffs, depth)
    
    derivatives = np.gradient(poly_fit, depth)
    
    threshold = np.max(np.abs(derivatives)) * 0.4
    leveling_point_index = np.argmax(np.abs(derivatives) < threshold)
    
    if leveling_point_index is not None:
        leveling_out_elevation = depth[leveling_point_index]
    else:
        leveling_out_elevation = None

    return leveling_out_elevation

def determine_side_of_centerline(point, centerline):
    # A method to determine which side of the centerline the point lies on
    nearest_point_on_centerline = centerline.interpolate(centerline.project(point))
    vector_to_point = np.array([point.x - nearest_point_on_centerline.x, point.y - nearest_point_on_centerline.y])
    # The side is determined by the sign of the cross product with a reference direction along the centerline
    reference_direction = np.array([centerline.coords[-1][0] - centerline.coords[0][0], centerline.coords[-1][1] - centerline.coords[0][1]])
    side = np.sign(np.cross(reference_direction, vector_to_point))
    return side

def plot_cross_section_area_to_wetted_perimeter_ratio(x, y, idx='', depth_increment=0.05, fig_output_path='', polynomial_order= 4):
    leveling_out_elevation = find_leveling_point(x, y, depth_increment, polynomial_order)

    depth = np.arange(min(y), max(y), depth_increment)
    ratio = []
    
    for d in depth:
        area = compute_cross_sectional_area_trapezoidal(x, y, d)
        perimeter = compute_wetted_perimeter(x, y, d)
        if perimeter > 0:
            ratio.append(area / (perimeter ** 4))
        else:
            ratio.append(0)
    
    ratio = np.array(ratio)
    poly_coeffs = np.polyfit(depth, ratio, polynomial_order)
    poly_fit = np.polyval(poly_coeffs, depth)
    
    # Plotting for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(depth, ratio, marker='o', linestyle='-', label='Data')
    plt.plot(depth, poly_fit, 'g--', label=f'{polynomial_order}th Degree Polynomial Fit')
    if leveling_out_elevation is not None:
        plt.axvline(x=leveling_out_elevation, color='red', linestyle='--', label='Leveling-Out Point')
    plt.xlabel('Depth (m)')
    plt.ylabel('Cross-Sectional Area / Wetted Perimeter ** 4')
    plt.title(f'Cross-Sectional Area / Wetted Perimeter vs. Depth (Index: {idx})')
    plt.legend()
    plt.grid(True)

    fig_save_path = os.path.join(fig_output_path, f'cross_section_area_to_wetted_perimeter_ratio_{idx}.png')
    plt.savefig(fig_save_path)
    plt.close()

    return leveling_out_elevation

def main(gpkg_path, raster_path, output_folder, polygon_path=None, output_gpkg_path=None, centerline_gpkg=None):
    gdf = gpd.read_file(gpkg_path)
    centerline_gdf = gpd.read_file(centerline_gpkg)
    centerline = centerline_gdf.geometry.iloc[0]  # Assuming a single centerline

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if output_gpkg_path and os.path.exists(output_gpkg_path):
        os.remove(output_gpkg_path)  # Remove existing output GeoPackage to avoid conflicts
    
    points_data = []  # To store the points for GeoPackage

    with rasterio.open(raster_path) as raster:
        for idx, row in gdf.iterrows():
            line = row.geometry
            if isinstance(line, LineString):
                print(f'Processing line at index {idx}')
                distances, raster_values, points = sample_raster_along_line(line, raster)
                
                np.savetxt(f'{output_folder}/sampled_data_{idx}.csv', np.column_stack((distances, raster_values)), delimiter=',', header='Distance,Raster Value', comments='')
                
                leveling_out_elevation = plot_cross_section_area_to_wetted_perimeter_ratio(distances, raster_values, idx=idx, fig_output_path=output_folder)
                
                if leveling_out_elevation is not None:
                    # Find the points on either side of the centerline
                    sides = np.array([determine_side_of_centerline(point, centerline) for point in points])
                    
                    left_side_indices = np.where(sides < 0)[0]
                    right_side_indices = np.where(sides > 0)[0]
                    
                    if len(left_side_indices) > 0 and len(right_side_indices) > 0:
                        closest_left_index = left_side_indices[np.argmin(np.abs(np.array(raster_values)[left_side_indices] - leveling_out_elevation))]
                        closest_right_index = right_side_indices[np.argmin(np.abs(np.array(raster_values)[right_side_indices] - leveling_out_elevation))]
                        
                        closest_left_point = points[closest_left_index]
                        closest_right_point = points[closest_right_index]
                        
                        points_data.append({'geometry': closest_left_point, 'line_index': idx, 'side': 'left', 'leveling_out_elevation': leveling_out_elevation})
                        points_data.append({'geometry': closest_right_point, 'line_index': idx, 'side': 'right', 'leveling_out_elevation': leveling_out_elevation})
                        
                        print(f"Leveling out point on left side for line index {idx}: {closest_left_point} at elevation {leveling_out_elevation}")
                        print(f"Leveling out point on right side for line index {idx}: {closest_right_point} at elevation {leveling_out_elevation}")

    # Create a GeoDataFrame from the points data and save it to a GeoPackage
    if points_data:
        points_gdf = gpd.GeoDataFrame(points_data, crs=gdf.crs)
        points_gdf.to_file(output_gpkg_path, driver="GPKG")
        print(f"GeoPackage created at: {output_gpkg_path}")

if __name__ == "__main__":
    gpkg_path = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\perpendiculars_sparse.gpkg"
    raster_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\dem_2021_ME_clip_filtered.tif"
    output_folder = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\ME_clip"
    polygon_path = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\ME_Centerlines_EPSG26913_single.gpkg"
    output_gpkg_path = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\LevelingOffPoints.gpkg"
    centerline_gpkg = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\ME_Centerlines_EPSG26913_single.gpkg"
    
    main(gpkg_path, raster_path, output_folder, polygon_path, output_gpkg_path, centerline_gpkg)
