import os
import logging
from datetime import datetime
from get_valleys_wavelet import get_valleys
from get_perpendiculars import create_smooth_perpendicular_lines
from clip_perpendiculars_by_hillslope import clip_perpendiculars_by_hillslope
from smooth_valley_bottom_polygons import smooth_polygon

perpendiculars_list = [
#r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\LM2_perpendiculars_100m.gpkg",
r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\LPM_smooth_perpendiculars_20m.gpkg",
]

centerlines_list = [
#r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\LM2_centerline.gpkg",
r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\LPM_centerline.gpkg",
]

dem_path = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\WBT_Outputs_Low\filled_dem.tif"
hillslopes_path = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\WBT_Outputs_Low\hillslopes_100k.gpkg"
###############IMPORTANT################
# The centerline path must be a multiline string with collected geometries (e.g. only a single line)
output_dir = r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\Wavelets"

# Define depth increment
depth_increment = 0.01  # Depth increment in meters
# Define smoothing parameters
window_size = 11  # Must be odd and > poly_order
poly_order = 9    # Polynomial order for Savitzky-Golay filter
wavelet_threshold = 0.1
minimum_depth = 1  # Minimum depth threshold to ignore damping onset before this depth
print_output = False

for centerline_path, perpendiculars_path in zip(centerlines_list, perpendiculars_list):
    # Define output folder
    output_folder = os.path.join(output_dir, os.path.basename(centerline_path).replace("_centerline.gpkg", ""))
    output_folder = os.path.join(output_folder, datetime.now().strftime("%Y%m%d_%Hh%Mm"))
    output_gpkg_path = os.path.join(output_folder, os.path.basename(centerline_path).replace(".gpkg", "") + "_valleys_wavelets.gpkg")
    print(f"Output geopackage path: {output_gpkg_path}")
    os.makedirs(output_folder, exist_ok=True)
    
    #######################################################################
    ################### CLIP PERPENDICULARS BY HILLSLOPE ##################
    #######################################################################
    try:
        # Create smooth perpendicular lines
        clipped_perpendiculars_path = os.path.join(output_folder, os.path.basename(perpendiculars_path).replace(".gpkg", "_hillslopes.gpkg"))
        clip_perpendiculars_by_hillslope(   perpendiculars_path, 
                                            hillslopes_path, 
                                            centerline_path, 
                                            clipped_perpendiculars_path)
        print(f"Perpendciulars clipped by hillslopes for {perpendiculars_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"Perpendiculars clipping failed for {perpendiculars_path}")
        continue
    
    #######################################################################
    ############################# GET VALLEYS  ############################
    #######################################################################
    
    try:
        start_time = datetime.now()
        # Execute main function
        json_path = get_valleys(
            perpendiculars_path= clipped_perpendiculars_path,
            dem_path=dem_path,
            output_folder=output_folder,
            output_gpkg_path=output_gpkg_path, 
            centerline_gpkg=centerline_path,
            depth_increment=depth_increment,
            print_output= print_output,
            window_size=window_size,
            poly_order=poly_order,
            wavelet_threshold=wavelet_threshold,
            minimum_depth=minimum_depth
        )
        print(f"Execution time: {datetime.now() - start_time}")
        from call_plot_cross_sections import run_cross_section_plotting
        run_cross_section_plotting(perpendiculars_path=perpendiculars_path, dem_raster=dem_path, json_file=json_path)

        print(f"Wavelet detection completed for {centerline_path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"Wavelet detection failed for {centerline_path}")
    
    #######################################################################
    ################### SMOOTH VALLEY BOTTOM POLYGONS #####################
    #######################################################################
    try:
        smooth_polygon(output_gpkg_path, output_gpkg_path.replace("valleys_wavelets", "valleys_wavelets_smoothed"), smoothing_window=5)
        print(f"Valley polygons smoothed for {centerline_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"Valley polygon smoothing failed for {centerline_path}")
        continue