import os
import logging
from datetime import datetime
from get_valleys_wavelet import get_valleys
from get_perpendiculars import create_smooth_perpendicular_lines
from clip_perpendiculars_by_hillslope import clip_perpendiculars_by_hillslope
from smooth_valley_bottom_polygons import smooth_polygon

# Import tqdm for progress bar in case it's used in other modules
from tqdm import tqdm

# Optional: Configure logging for this script if needed
# This is useful if you want to capture logs from this script as well
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

perpendiculars_list = [
    r"Y:\ATD\GIS\Bennett\Valley Geometry\Perpendiculars\ME_smooth_perpendiculars_200m.gpkg"
    # r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Valley_smooth_perpendiculars_200m.gpkg"
    #r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\LM2_smooth_perpendiculars_10m.gpkg",
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\LPM_smooth_perpendiculars_10m.gpkg",
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\MM_smooth_perpendiculars_10m.gpkg",
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\MPM_smooth_perpendiculars_10m.gpkg",
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\UM1_smooth_perpendiculars_10m.gpkg",
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars\UM2_smooth_perpendiculars_10m.gpkg",
]

centerlines_list = [
    r"Y:\ATD\GIS\Bennett\Valley Geometry\Centerlines\ME_centerline.gpkg"
    # r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Valley_CL_single_part.gpkg"
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\LM2_centerline.gpkg",
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\LPM_centerline.gpkg",
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\MM_centerline.gpkg",
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\MPM_centerline.gpkg",
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\UM1_centerline.gpkg",
    # r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\UM2_centerline.gpkg",
]

# dem_path = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\WBT_Outputs_Low\filled_dem.tif"
# dem_path = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Inputs\Terrain\WBT_Outputs\filled_dem.tif"
dem_path = r"Y:\ATD\GIS\Bennett\DEMs\LIDAR\OT 2021\WBT_Outputs\filled_dem.tif"

hillslopes_path = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\WBT_Outputs_Low\hillslopes_100k.gpkg"

###############IMPORTANT################
# The centerline path must be a multiline string with collected geometries (e.g., only a single line)
output_dir = r"Y:\ATD\GIS\Bennett\Valley Bottoms\ATD_Algorithm\Wavelets"
# output_dir = r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\Wavelets"
# output_dir = r"Y:\ATD\GIS\Valley Bottom Testing\Control Valleys\Outputs\Wavelets"

# Define depth increment
depth_increment = 0.01  # Depth increment in meters

# Define smoothing parameters
wavelet_threshold = 0.1
minimum_depth = 1 # Minimum depth threshold to ignore damping onset before this depth
print_output = True

for centerline_path, perpendiculars_path in zip(centerlines_list, perpendiculars_list):
    # Define output folder
    base_output_folder = os.path.join(
        output_dir,
        os.path.basename(centerline_path).replace("_centerline.gpkg", "")
    )
    # Append current datetime to ensure unique folder per run
    output_folder = os.path.join(
        base_output_folder,
        datetime.now().strftime("%Y%m%d_%Hh%Mm")
    )
    print(f"Output directory: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    #######################################################################
    ################### CLIP PERPENDICULARS BY HILLSLOPE ##################
    #######################################################################
    # try:
    #     # Create smooth perpendicular lines
    #     clipped_perpendiculars_path = os.path.join(
    #         output_folder,
    #         os.path.basename(perpendiculars_path).replace(".gpkg", "_hillslopes.gpkg")
    #     )
    #     clip_perpendiculars_by_hillslope(
    #         perpendiculars_path,
    #         hillslopes_path,
    #         centerline_path,
    #         clipped_perpendiculars_path
    #     )
    #     print(f"Perpendiculars clipped by hillslopes for {perpendiculars_path}")
    # except Exception as e:
    #     logging.error(f"An error occurred during clipping perpendiculars: {e}")
    #     print(f"Perpendiculars clipping failed for {perpendiculars_path}")
    #     continue  # Skip to the next iteration if clipping fails

    #######################################################################
    ############################# GET VALLEYS  ############################
    #######################################################################

    try:
        start_time = datetime.now()
        # Execute get_valleys function with the correct output directory
        damping_polygon_path, damping_onset_json_path = get_valleys(
            perpendiculars_path=perpendiculars_path,  # Use clipped perpendiculars
            centerlines_path=centerline_path,
            dem_path=dem_path,
            output_dir=output_folder,  # Pass the specific output_folder
            wavelet_threshold=wavelet_threshold,
            minimum_depth=minimum_depth,
            print_output=print_output
        )

        print(f"Execution time for get_valleys: {datetime.now() - start_time}")

        # Check if the returned paths are valid before proceeding
        if damping_polygon_path and damping_onset_json_path:
            from call_plot_cross_sections import run_cross_section_plotting
            run_cross_section_plotting(
                perpendiculars_path=perpendiculars_path,  # Use clipped perpendiculars
                dem_raster=dem_path,
                json_file=damping_onset_json_path
            )
            print(f"Wavelet detection completed for {centerline_path}")
        else:
            print(f"Wavelet detection incomplete for {centerline_path}. Check logs for details.")

    except Exception as e:
        logging.error(f"An error occurred during get_valleys: {e}")
        print(f"Wavelet detection failed for {centerline_path}")
        continue  # Skip to the next iteration if get_valleys fails

    #######################################################################
    ################### SMOOTH VALLEY BOTTOM POLYGONS #####################
    #######################################################################
    try:
        # Ensure that damping_polygon_path exists before attempting to smooth
        if damping_polygon_path:
            smooth_poly_path = os.path.join(
                output_folder,
                os.path.basename(centerline_path).replace("_centerline.gpkg", "_valleys_wavelets_smoothed.gpkg")
            )
            smooth_polygon(damping_polygon_path, smooth_poly_path, smoothing_window=5)
            print(f"Valley polygons smoothed for {centerline_path}")
        else:
            print(f"No damping polygon to smooth for {centerline_path}")
    except Exception as e:
        logging.error(f"An error occurred during smoothing polygons: {e}")
        print(f"Valley polygon smoothing failed for {centerline_path}")
        continue  # Continue to the next iteration even if smoothing fails
