import numpy as np
import matplotlib.pyplot as plt
import os

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

# read x,y from csv file
input_dir = r"Y:\ATD\GIS\Bennett\Channel Polygons\Centerlines_LSDTopo\Perpendiculars"
input_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
print("Input files:", input_files)
for input_file, idx in zip(input_files, range(len(input_files))):
    input_file_path = os.path.join(input_dir, input_file)
    data = np.loadtxt(input_file_path, delimiter=',', skiprows=1)
    distance = data[:, 0]
    elevation = data[:, 1]

    leveling_out_elevation = plot_cross_section_area_to_wetted_perimeter_ratio(distance, elevation, idx=idx, fig_output_path=input_dir)
    print(f"Leveling out elevation for file {input_file}: {leveling_out_elevation}")
