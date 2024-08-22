import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point
import time
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
    
    return distances, raster_values

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

def plot_cross_section_area_to_wetted_perimeter_ratio(x, y, idx='', depth_increment=0.1, fig_output_path='', polynomial_order=4):
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
    level_out_index = next((i for i, derivative in enumerate(np.abs(derivatives)) if derivative < threshold), None)
    
    if level_out_index is not None:
        leveling_out_elevation = depth[level_out_index]
    else:
        leveling_out_elevation = None

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

def main(gpkg_path, raster_path, output_folder, polygon_path=None, output_gpkg_path=None):
    gdf = gpd.read_file(gpkg_path)
    
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
                distances, raster_values = sample_raster_along_line(line, raster)
                
                np.savetxt(f'{output_folder}/sampled_data_{idx}.csv', np.column_stack((distances, raster_values)), delimiter=',', header='Distance,Raster Value', comments='')
                
                leveling_out_elevation = plot_cross_section_area_to_wetted_perimeter_ratio(distances, raster_values, idx=idx, fig_output_path=output_folder)
                
                if leveling_out_elevation is not None:
                    # Find the point along the line where the leveling off occurred
                    level_off_distance = np.interp(leveling_out_elevation, raster_values, distances)
                    level_off_point = line.interpolate(level_off_distance)
                    points_data.append({'geometry': level_off_point, 'line_index': idx, 'leveling_out_elevation': leveling_out_elevation})
                    print(f"Leveling out point for line index {idx}: {level_off_point} at elevation {leveling_out_elevation}")

    # Create a GeoDataFrame from the points data and save it to a GeoPackage
    if points_data:
        points_gdf = gpd.GeoDataFrame(points_data, crs=gdf.crs)
        points_gdf.to_file(output_gpkg_path, driver="GPKG")
        print(f"GeoPackage created at: {output_gpkg_path}")

if __name__ == "__main__":
    gpkg_path = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\perpendiculars_sparse.gpkg"
    raster_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\dem_2021_ME_clip.tif"
    output_folder = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\SlopevsDist"
    polygon_path = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\ME_Centerlines_EPSG26913_single.gpkg"
    output_gpkg_path = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\LevelingOffPoints.gpkg"
    
    main(gpkg_path, raster_path, output_folder, polygon_path, output_gpkg_path)
