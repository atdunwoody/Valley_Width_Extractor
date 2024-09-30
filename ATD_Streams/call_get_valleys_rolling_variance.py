import os
from ATD_Streams.get_valleys_rolling_variance import main
import logging
from datetime import datetime

perpendiculars_list = [
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Perpendiculars\ME_clipped_perps_5m.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Perpendiculars\MM_clipped_perps_5m.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Perpendiculars\MW_clipped_perps_5m.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Perpendiculars\UE_clipped_perps_5m.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Perpendiculars\UM_clipped_perps_5m.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\Perpendiculars\UW_clipped_perps_5m.gpkg",
]

centerlines_list = [
    r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\ME_clipped.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\MM_clipped.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\MW_clipped.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\UE_clipped.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\UM_clipped.gpkg",
r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\UW_clipped.gpkg",
]

raster_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT\filled_dem.tif"
###############IMPORTANT################
# The centerline path must be a multiline string with collected geometries (e.g. only a single line)
output_dir = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\ATD_algorithm\Rolling Variance"

# Define depth increment
depth_increment = 0.01  # Depth increment in meters
# Define smoothing parameters
window_size = 11  # Must be odd and > poly_order
poly_order = 2    # Polynomial order for Savitzky-Golay filter
rolling_window_size = 100  # Window size for rolling window filter
min_consecutive_decreases = 7  # Minimum number of consecutive decreases to detect a valley
print_output = True

for centerline_path, perpendiculars_path in zip(centerlines_list, perpendiculars_list):
    # Define output folder
    output_folder = os.path.join(output_dir, os.path.basename(centerline_path).replace("_clipped.gpkg", ""))
    output_gpkg_path = os.path.join(output_folder, os.path.basename(centerline_path).replace(".gpkg", "") + "_valleys_variance.gpkg")
    print(f"Output geopackage path: {output_gpkg_path}")
    os.makedirs(output_folder, exist_ok=True)
    # Call the main function from ATD_streams.py
    try:
        
        start_time = datetime.now()
        # Execute main function
        json_path = main(
            gpkg_path=perpendiculars_path,
            raster_path=raster_path,
            output_folder=output_folder,
            output_gpkg_path=output_gpkg_path, 
            centerline_gpkg=centerline_path,
            depth_increment=depth_increment,
            print_output=print_output,
            window_size=window_size,
            poly_order=poly_order,
            rolling_window_size=rolling_window_size,
            min_consecutive_decreases=min_consecutive_decreases
        )
        print(f"Execution time: {datetime.now() - start_time}")
        from call_plot_cross_sections import run_cross_section_plotting
        run_cross_section_plotting(perpendiculars_path=perpendiculars_path, dem_raster=raster_path, json_file=json_path)

        print(f"Wavelet detection completed for {centerline_path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"Wavelet detection failed for {centerline_path}")